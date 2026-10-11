# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for the vLLM metrics monitor's wiring into the v1 PPOTrainer (verl/trainer/ppo/v1/trainer_base.py).

The monitor itself is tested in tests/workers/rollout/test_vllm_metrics_on_cpu.py. Here a fake with the monitor's hold
semantics records when the trainer polls: in sync mode only while a training batch generates (from dispatch until the
batch is collected, before on_sample_end sleeps the replicas), never during validation, collected once per step before
logging, and stopped at every exit of fit.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_base import PPOTrainer

MODULE = "verl.trainer.ppo.v1.trainer_base"


class _FakeMonitor:
    """VllmMetricsMonitor's pause/resume/collect/stop contract, logging into a shared event list."""

    def __init__(self, events: list, active: bool = False):
        self.events = events
        self._holds = set() if active else {"default"}
        self.collected = 0
        self.stopped = False

    @property
    def active(self) -> bool:
        return not self._holds

    def resume(self, reason: str = "default"):
        self.events.append(("resume", reason))
        self._holds.discard(reason)

    def pause(self, reason: str = "default"):
        self.events.append(("pause", reason))
        self._holds.add(reason)

    def collect(self) -> dict:
        self.collected += 1
        self.events.append(("collect",))
        return {"vllm/num_preemptions": 3.0, "vllm/samples": 7.0}

    def stop(self):
        self.stopped = True
        self.events.append(("stop",))


class _SampleEnd(Exception):
    """Raised by on_sample_end to stop _step_once right after the generation bracket."""


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        self.events.append(("on_sample_end", self.vllm_metrics_monitor.active))
        raise _SampleEnd


def _trainer(trainer_mode: str = "sync", monitor_active: bool = False) -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.events = []
    trainer.trainer_mode = trainer_mode
    trainer.parameter_sync_step = 1
    trainer.global_steps = 1
    trainer.config = OmegaConf.create(
        {"data": {"train_batch_size": 8}, "actor_rollout_ref": {"rollout": {"temperature": 1.0}}}
    )
    trainer.vllm_metrics_monitor = _FakeMonitor(trainer.events, active=monitor_active)

    def dispatch():
        trainer.events.append(("dispatch", trainer.vllm_metrics_monitor.active))

    trainer._add_batch_to_generate = dispatch

    class Buffer:
        fail = False

        def sample(self, global_steps, partition_id, batch_size):
            trainer.events.append(("sample", trainer.vllm_metrics_monitor.active))
            if self.fail:
                raise RuntimeError("generation failed")
            return SimpleNamespace(extra_info={}), {}

    trainer.replay_buffer = Buffer()
    return trainer


class TestSyncGenerationWindow:
    def test_window_spans_dispatch_to_collection(self):
        trainer = _trainer()
        with pytest.raises(_SampleEnd):
            trainer.step({}, {})
        assert trainer.events == [
            ("resume", "default"),
            ("dispatch", True),
            ("sample", True),
            ("pause", "default"),
            ("on_sample_end", False),  # replicas sleep with the monitor already paused
        ]

    def test_window_closes_when_sampling_fails(self):
        trainer = _trainer()
        trainer.replay_buffer.fail = True
        with pytest.raises(RuntimeError, match="generation failed"):
            trainer.step({}, {})
        assert trainer.events[-1] == ("pause", "default")
        assert not trainer.vllm_metrics_monitor.active

    @pytest.mark.parametrize("mode", ["colocate_async", "separate_async"])
    def test_async_modes_poll_continuously(self, mode):
        trainer = _trainer(trainer_mode=mode, monitor_active=True)
        with pytest.raises(_SampleEnd):
            trainer.step({}, {})
        assert [e for e in trainer.events if e[0] in ("resume", "pause")] == []
        assert ("dispatch", True) in trainer.events and ("on_sample_end", True) in trainer.events

    def test_without_monitor(self):
        trainer = _trainer()
        trainer.vllm_metrics_monitor = None
        trainer._open_vllm_metrics_window()
        trainer._close_vllm_metrics_window()
        trainer._stop_vllm_metrics_monitor()
        assert PPOTrainer.vllm_metrics_monitor is None


class TestValidation:
    def _with_validate(self, trainer, fail=False):
        trainer.on_validate_begin = lambda: trainer.events.append(("validate_begin",))
        trainer.on_validate_end = lambda: trainer.events.append(("validate_end",))

        def validate():
            trainer.events.append(("validate", trainer.vllm_metrics_monitor.active))
            if fail:
                raise RuntimeError("validation failed")
            return {"val-core/acc": 0.5}

        trainer._validate = validate
        return trainer

    @pytest.mark.parametrize("mode,active", [("sync", False), ("colocate_async", True)])
    def test_validation_is_held(self, mode, active):
        trainer = self._with_validate(_trainer(trainer_mode=mode, monitor_active=active))
        assert trainer._run_validation() == {"val-core/acc": 0.5}
        assert trainer.events == [
            ("validate_begin",),
            ("pause", "validation"),
            ("validate", False),
            ("resume", "validation"),
            ("validate_end",),
        ]
        assert trainer.vllm_metrics_monitor.active is active  # the hold does not change the mode's own state

    def test_hold_released_when_validation_fails(self):
        trainer = self._with_validate(_trainer(trainer_mode="colocate_async", monitor_active=True), fail=True)
        with pytest.raises(RuntimeError, match="validation failed"):
            trainer._run_validation()
        assert trainer.events[-1] == ("resume", "validation")
        assert trainer.vllm_metrics_monitor.active

    def test_without_monitor(self):
        trainer = self._with_validate(_trainer())
        trainer.vllm_metrics_monitor = None
        trainer._validate = lambda: {"val-core/acc": 1.0}
        assert trainer._run_validation() == {"val-core/acc": 1.0}


class TestAddresses:
    def test_hybrid_replicas(self):
        trainer = _trainer()
        trainer.llm_server_manager = SimpleNamespace(server_addresses=["10.0.0.1:1", "10.0.0.1:2"])
        assert trainer._vllm_metrics_addresses() == ["10.0.0.1:1", "10.0.0.1:2"]

    def test_standalone_replicas_of_separate_async(self):
        trainer = _trainer(trainer_mode="separate_async")
        trainer.llm_server_manager = SimpleNamespace(server_addresses=["a:1"])
        trainer.standalone_server_manager = SimpleNamespace(server_addresses=["b:1", "b:2"])
        assert trainer._vllm_metrics_addresses() == ["a:1", "b:1", "b:2"]

    def test_addresses_are_read_lazily(self):
        trainer = _trainer()
        trainer.llm_server_manager = SimpleNamespace(server_addresses=["a:1"])
        trainer.llm_server_manager.server_addresses.append("a:2")
        assert trainer._vllm_metrics_addresses() == ["a:1", "a:2"]


def _fit_trainer(total_steps: int, val_only: bool = False) -> _StubTrainer:
    trainer = _trainer()
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "p",
                "experiment_name": "e",
                "logger": ["console"],
                "val_before_train": True,
                "val_only": val_only,
                "total_epochs": 1,
                "save_freq": -1,
                "test_freq": 2,
                "rollout_data_dir": None,
            },
            "global_profiler": {"steps": None, "profile_continuous_steps": False},
        }
    )
    trainer.global_steps = 0
    trainer.steps_per_epoch = 100
    trainer.total_training_steps = total_steps
    trainer.logged = []

    def validate():
        monitor = trainer.vllm_metrics_monitor
        trainer.events.append(("validate", monitor.active if monitor is not None else None))
        return {"val-core/acc": 0.5}

    def step(metrics, timing_raw):
        trainer.events.append(("step", trainer.global_steps))
        return SimpleNamespace(keys=[], partition_id="train")

    trainer._validate = validate
    trainer.step = step
    trainer._compute_metrics = lambda *args, **kwargs: None
    trainer._shutdown_dump_executor = lambda: trainer.events.append(("shutdown_dump",))
    return trainer


@pytest.fixture
def fit_env():
    tracking = MagicMock()
    with (
        patch(f"{MODULE}.SkipManager"),
        patch(f"{MODULE}.Tracking", return_value=tracking),
        patch(f"{MODULE}.ValidationGenerationsLogger"),
        patch(f"{MODULE}.DapoFilteredRewardTableLogger"),
        patch(f"{MODULE}.tq"),
    ):
        yield tracking


class TestFit:
    def test_each_step_logs_its_window_and_the_monitor_stops(self, fit_env):
        trainer = _fit_trainer(total_steps=2)
        monitor = trainer.vllm_metrics_monitor
        trainer.fit(agent_loop_manager=None)

        logged = [c.kwargs["data"] for c in fit_env.log.call_args_list]
        assert len(logged) == 3  # initial validation + 2 steps
        assert "vllm/num_preemptions" not in logged[0]  # the initial validation is not a generation window
        assert all(d["vllm/num_preemptions"] == 3.0 for d in logged[1:])
        assert monitor.collected == 2
        assert [e for e in trainer.events if e[0] == "validate"] == [("validate", False)] * 2
        assert monitor.stopped and trainer.vllm_metrics_monitor is None
        # collect happens after the step's validation (step 2 validates), right before logging
        tail = [e[0] for e in trainer.events if e[0] in ("step", "validate", "collect", "stop")]
        assert tail == ["validate", "step", "collect", "step", "validate", "collect", "stop"]

    def test_val_only_stops_the_monitor(self, fit_env):
        trainer = _fit_trainer(total_steps=2, val_only=True)
        monitor = trainer.vllm_metrics_monitor
        trainer.fit(agent_loop_manager=None)
        assert monitor.stopped and monitor.collected == 0
        assert ("step", 1) not in trainer.events

    def test_fit_without_monitor(self, fit_env):
        trainer = _fit_trainer(total_steps=1)
        trainer.vllm_metrics_monitor = None
        trainer.fit(agent_loop_manager=None)
        logged = [c.kwargs["data"] for c in fit_env.log.call_args_list]
        assert not any(k.startswith("vllm/") for d in logged for k in d)
