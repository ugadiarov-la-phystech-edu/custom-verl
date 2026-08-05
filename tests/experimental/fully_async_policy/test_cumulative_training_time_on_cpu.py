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
"""Unit tests for the fully-async training/train_time_s metric (exact virtual-timeline
cumulative training time) and its fully_async/timing/* component tags:
- RolloutSample carries the virtual-timeline stamps through (cloud)pickle
- FullyAsyncRollouter anchors first_sample_time once, accumulates validation-pause
  time only after the anchor exists, and stamps samples at enqueue
- FullyAsyncLLMServerClient holds train requests at the load-balancer gate during
  validation windows while validation requests bypass it
- FullyAsyncTrainer replays the pipeline schedule on a virtual timeline that
  matches the wall clock of a run with neither validation nor checkpointing, in
  rollout-bound and trainer-bound regimes alike
- timing state survives a checkpoint round-trip (timing_state.json), so a resumed
  run continues cumulative_training_time instead of restarting at zero

Run: pytest tests/experimental/fully_async_policy/test_cumulative_training_time_on_cpu.py
"""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import ray.cloudpickle
from omegaconf import OmegaConf

import verl.experimental.fully_async_policy.fully_async_rollouter as rollouter_module
from verl.experimental.fully_async_policy.detach_utils import RolloutSample
from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncLLMServerClient,
)
from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncRollouter as _RollouterActor,
)
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer as _TrainerActor
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput


def _unwrap_ray_actor_class(actor_cls):
    """Both classes are @ray.remote ActorClass wrappers; tests need the plain class."""
    return actor_cls.__ray_metadata__.modified_class if hasattr(actor_cls, "__ray_metadata__") else actor_cls


FullyAsyncRollouter = _unwrap_ray_actor_class(_RollouterActor)
FullyAsyncTrainer = _unwrap_ray_actor_class(_TrainerActor)

TIMING_PREFIX = "fully_async/timing/"


class _FakeBatch:
    """Minimal picklable stand-in for the DataProto held by RolloutSample."""

    def __init__(self):
        self.non_tensor_batch = {}

    def __len__(self):
        return 1


def _make_rollouter(first_sample_time=None):
    """Minimal FullyAsyncRollouter with only the attributes the timing paths touch.
    No llm_server_manager attribute: pause/resume then skip the gate/abort layer."""
    r = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
    r.lock = asyncio.Lock()
    r._resume_event = asyncio.Event()
    r._resume_event.set()
    r.paused = False
    r.first_sample_time = first_sample_time
    r.cumulative_validation_time = 0.0
    r.cumulative_checkpoint_pause = 0.0
    r._validation_pause_start = None
    return r


def _make_trainer(first_sample_time=None, cumulative_validation_time=0.0, cumulative_save_time=0.0):
    """Minimal FullyAsyncTrainer with only the attributes under test."""
    t = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
    t.rollouter_first_sample_time = first_sample_time
    t.rollouter_cumulative_validation_time = cumulative_validation_time
    t.rollouter_cumulative_checkpoint_pause = 0.0
    t.cumulative_save_time = cumulative_save_time
    t.timing_wall_offset = 0.0
    t.timing_validation_offset = 0.0
    t.timing_save_offset = 0.0
    t.virtual_free_time = None
    t.virtual_training_time_offset = 0.0
    t._step_virtual_start = None
    t._step_actual_start = None
    t._step_validate_time = 0.0
    t._step_save_time = 0.0
    t.current_param_version = 0
    t.last_ckpt_version = 0
    t.max_steps_duration = 0
    return t


def _sample(enqueue_time, validation_pause_before, checkpoint_pause_before=0.0):
    return SimpleNamespace(
        enqueue_time=enqueue_time,
        validation_pause_before=validation_pause_before,
        checkpoint_pause_before=checkpoint_pause_before,
    )


def _run_step(trainer, consumer_end, samples, step_end, validate_time=0.0, save_time=0.0):
    """Drive one trainer step through the production virtual-clock hooks."""
    trainer._step_validate_time = validate_time
    trainer._step_save_time = save_time
    trainer._open_virtual_step(consumer_end, samples)
    trainer._advance_virtual_clock(now=step_end)
    return trainer.virtual_free_time


# ---------------------------------------------------------------- RolloutSample stamps


def test_rollout_sample_stamp_defaults_and_pickle_roundtrip():
    sample = RolloutSample(full_batch=_FakeBatch(), sample_id="s0", epoch=0, rollout_status={})
    assert sample.enqueue_time is None
    assert sample.validation_pause_before == 0.0
    assert sample.checkpoint_pause_before == 0.0

    sample.enqueue_time = 123.0
    sample.validation_pause_before = 4.5
    restored = ray.cloudpickle.loads(ray.cloudpickle.dumps(sample))
    assert restored.enqueue_time == 123.0
    assert restored.validation_pause_before == 4.5
    assert restored.checkpoint_pause_before == 0.0


# ---------------------------------------------------------------- rollouter accounting


def test_rollouter_pause_resume_accumulates_after_anchor():
    rollouter = _make_rollouter(first_sample_time=100.0)

    async def _pause_validate_resume():
        await rollouter.pause_generation_for_validation()
        assert rollouter.paused is True
        assert not rollouter._resume_event.is_set()
        await asyncio.sleep(0.03)  # the validation window
        return await rollouter.resume_generation_after_validation()

    state = asyncio.run(_pause_validate_resume())
    assert rollouter.paused is False
    assert rollouter._resume_event.is_set()
    assert rollouter.cumulative_validation_time >= 0.03
    assert state["first_sample_time"] == 100.0
    assert state["cumulative_validation_time"] == rollouter.cumulative_validation_time

    before = rollouter.cumulative_validation_time
    asyncio.run(_pause_validate_resume())
    assert rollouter.cumulative_validation_time >= before + 0.03, "must accumulate across windows"


def test_rollouter_pause_resume_ignores_windows_before_anchor():
    rollouter = _make_rollouter(first_sample_time=None)

    async def _window():
        await rollouter.pause_generation_for_validation()
        await asyncio.sleep(0.02)
        return await rollouter.resume_generation_after_validation()

    state = asyncio.run(_window())
    assert rollouter.cumulative_validation_time == 0.0
    assert state["first_sample_time"] is None
    # the pause start must not leak into a later post-anchor window
    assert rollouter._validation_pause_start is None


def test_rollouter_resume_without_pause_is_safe():
    rollouter = _make_rollouter(first_sample_time=100.0)
    state = asyncio.run(rollouter.resume_generation_after_validation())
    assert rollouter.cumulative_validation_time == 0.0
    assert state["cumulative_validation_time"] == 0.0


def test_feed_samples_anchor_is_set_once():
    rollouter = _make_rollouter()
    rollouter.config = OmegaConf.create({})
    rollouter.global_steps = 1
    rollouter.total_rollout_steps = 10
    rollouter.pending_queue = asyncio.Queue()
    rollouter._create_continuous_iterator = lambda: iter([(0, {"a": 1}), (0, {"a": 2})])

    with patch.object(rollouter_module, "prepare_single_generation_data", lambda batch_dict, config: _FakeBatch()):
        asyncio.run(rollouter._feed_samples())
    assert rollouter.first_sample_time is not None
    anchor = rollouter.first_sample_time

    # a second feed pass (resume scenarios) must not move the anchor
    rollouter._create_continuous_iterator = lambda: iter([(1, {"a": 3})])
    with patch.object(rollouter_module, "prepare_single_generation_data", lambda batch_dict, config: _FakeBatch()):
        asyncio.run(rollouter._feed_samples())
    assert rollouter.first_sample_time == anchor


def test_process_single_sample_stamps_enqueue_and_pauses():
    rollouter = _make_rollouter(first_sample_time=100.0)
    rollouter.cumulative_validation_time = 7.5
    rollouter.cumulative_checkpoint_pause = 1.25
    rollouter.total_generated_samples = 0
    rollouter.dropped_stale_samples = 0
    rollouter.processed_sample_count = 0

    fake_batch = _FakeBatch()

    async def _generate(batch):
        return fake_batch

    async def _stats():
        return {}

    async def _put_sample(sample):
        rollouter._last_put = sample
        return True

    rollouter.async_rollout_manager = SimpleNamespace(generate_sequences_single=_generate)
    rollouter.get_statistics = _stats
    rollouter.message_queue_client = SimpleNamespace(put_sample=lambda sample: _put_sample(sample))

    sample = RolloutSample(full_batch=_FakeBatch(), sample_id="s1", epoch=0, rollout_status={})
    before = time.time()
    asyncio.run(rollouter._process_single_sample_streaming(sample))
    after = time.time()

    stamped = ray.cloudpickle.loads(rollouter._last_put)
    assert before <= stamped.enqueue_time <= after
    assert stamped.validation_pause_before == 7.5
    assert stamped.checkpoint_pause_before == 1.25


# ---------------------------------------------------------------- client train gate


class _StubLoadBalancer:
    """Emulates the GlobalRequestLoadBalancer handle: .remote() returns an awaitable."""

    def __init__(self, paused_flags):
        self._flags = list(paused_flags)  # consumed in order; last value repeats
        self.calls = 0
        self.is_train_generation_paused = SimpleNamespace(remote=self._remote)

    def _remote(self):
        idx = min(self.calls, len(self._flags) - 1)
        self.calls += 1

        async def _value():
            return self._flags[idx]

        return _value()


def _token_output(stop_reason="completed"):
    return TokenOutput(token_ids=[1], log_probs=[0.0], num_preempted=0, stop_reason=stop_reason)


def _make_client(load_balancer, partial_rollout=True):
    client = FullyAsyncLLMServerClient.__new__(FullyAsyncLLMServerClient)
    client.config = OmegaConf.create({"async_training": {"partial_rollout": partial_rollout}})
    client._load_balancer = load_balancer
    return client


def test_client_gate_holds_train_requests_while_paused():
    lb = _StubLoadBalancer([True, True, False])
    client = _make_client(lb)
    base_calls = []

    async def _base_generate(self, request_id, **kwargs):
        base_calls.append(kwargs["sampling_params"].copy())
        return _token_output()

    with patch.object(LLMServerClient, "generate", _base_generate):
        out = asyncio.run(
            client.generate(request_id="r1", prompt_ids=[1, 2], sampling_params={"max_tokens": 8, "temperature": 1.0})
        )
    assert out.stop_reason == "completed"
    assert len(base_calls) == 1
    assert lb.calls == 3, "train request must poll the gate until it opens"


def test_client_gate_bypassed_by_validation_and_marker_kept_for_next_turn():
    lb = _StubLoadBalancer([True])  # gate closed: a train request would hang here
    client = _make_client(lb)

    async def _base_generate(self, request_id, **kwargs):
        return _token_output()

    sampling_params = {"max_tokens": 8, "verl_validate": True}
    with patch.object(LLMServerClient, "generate", _base_generate):
        asyncio.run(client.generate(request_id="r2", prompt_ids=[1], sampling_params=sampling_params))
    assert lb.calls == 0, "validation requests must not consult the gate"
    # Multi-turn agent loops reuse this dict on every turn: the marker must survive so
    # later turns of a validation request also bypass the gate (a popped marker would
    # deadlock them against the closed gate mid-validation).
    assert sampling_params.get("verl_validate") is True


def test_base_client_strips_marker_from_engine_params_without_mutating_caller():
    engine_kwargs = {}

    class _EngineGenerate:
        def remote(self, **kwargs):
            engine_kwargs.update(kwargs)

            async def _c():
                return _token_output()

            return _c()

    server_stub = SimpleNamespace(generate=_EngineGenerate())

    class _Acquire:
        def remote(self, request_id):
            async def _c():
                return ("s0", server_stub)

            return _c()

    class _Release:
        def remote(self, server_id):
            return None

    client = LLMServerClient.__new__(LLMServerClient)
    client.config = OmegaConf.create({})
    client._load_balancer = SimpleNamespace(acquire_server=_Acquire(), release_server=_Release())

    sampling_params = {"max_tokens": 8, "verl_validate": True}
    asyncio.run(client.generate(request_id="r4", prompt_ids=[1], sampling_params=sampling_params))
    assert "verl_validate" not in engine_kwargs["sampling_params"], "marker must never reach the engine"
    assert sampling_params["verl_validate"] is True, "caller's dict must not be mutated"


def test_client_gate_holds_partial_rollout_resume():
    # First generate is aborted (validation pause preempted it); the resume must poll the
    # gate (closed once) before re-entering the engine.
    lb = _StubLoadBalancer([False, True, False])
    client = _make_client(lb)
    stop_reasons = iter(["aborted", "completed"])
    base_calls = []

    async def _base_generate(self, request_id, **kwargs):
        base_calls.append(kwargs["prompt_ids"])
        return _token_output(stop_reason=next(stop_reasons))

    with patch.object(LLMServerClient, "generate", _base_generate):
        out = asyncio.run(client.generate(request_id="r3", prompt_ids=[1], sampling_params={"max_tokens": 8}))
    assert out.stop_reason == "completed"
    assert len(base_calls) == 2
    assert lb.calls == 3


# ---------------------------------------------------------------- trainer virtual clock


def test_virtual_clock_rollout_bound_matches_no_validation_run():
    # Reference run (no validation): batches ready at t=10, 20 (R=10); trainer
    # busy U=6 per step -> steps end at 16 and 26.
    # Run with validation: an 8s validation pauses generation, so batch 2
    # arrives at t=28 stamped (28, pause=8) -> virtual ready 20. The trainer
    # idles 16..28 waiting; the metric must not count that induced wait.
    trainer = _make_trainer(first_sample_time=0.0)
    end1 = _run_step(trainer, consumer_end=10.0, samples=[_sample(10.0, 0.0)], step_end=16.0)
    assert end1 == 16.0
    end2 = _run_step(trainer, consumer_end=28.0, samples=[_sample(28.0, 8.0)], step_end=34.0)
    assert end2 == 26.0, "virtual step 2 must end where the no-validation run ends"

    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=34.0)
    assert step_data["training/train_time_s"] == 26.0


def test_virtual_clock_trainer_bound_matches_no_validation_run():
    # Reference run: batches ready at t=6, 12 (R=6); trainer busy U=10 -> steps
    # end at 16 and 26; the trainer is the bottleneck throughout.
    # Run with validation: batch 2 was enqueued at t=12 before a validation window;
    # the trainer's step-2 compute is unchanged, so wall time is unchanged — and so
    # must the metric be (a naive wall - validation subtraction would over-subtract).
    trainer = _make_trainer(first_sample_time=0.0, cumulative_validation_time=8.0)
    _run_step(trainer, consumer_end=6.0, samples=[_sample(6.0, 0.0)], step_end=16.0)
    end2 = _run_step(trainer, consumer_end=16.0, samples=[_sample(12.0, 0.0)], step_end=26.0)
    assert end2 == 26.0

    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=26.0)
    assert step_data["training/train_time_s"] == 26.0


def test_virtual_clock_excludes_validation_stall():
    # A 3s validation stall inside the step must not advance the virtual clock:
    # 10s of measured step time -> 7s of busy.
    trainer = _make_trainer(first_sample_time=0.0)
    end = _run_step(trainer, consumer_end=10.0, samples=[_sample(10.0, 0.0)], step_end=20.0, validate_time=3.0)
    assert end == 17.0


def test_virtual_clock_excludes_checkpoint_save_time():
    # A 5s checkpoint save inside the step is not training and must not advance
    # the virtual clock: 15s of measured step time -> 10s of busy.
    trainer = _make_trainer(first_sample_time=0.0)
    end = _run_step(trainer, consumer_end=10.0, samples=[_sample(10.0, 0.0)], step_end=25.0, save_time=5.0)
    assert end == 20.0


def test_open_virtual_step_takes_last_sample_and_handles_missing_stamps():
    trainer = _make_trainer(first_sample_time=0.0)
    trainer.virtual_free_time = 5.0
    # batch ready = max over samples of (enqueue - val pause - ckpt pause)
    #             = max(4, 9, 6) = 9
    samples = [_sample(10.0, 6.0), _sample(11.0, 2.0), _sample(12.0, 5.0, checkpoint_pause_before=1.0)]
    trainer._open_virtual_step(30.0, samples)
    assert trainer._step_virtual_start == 9.0
    assert trainer._step_actual_start == 30.0

    # samples stamped before the checkpoint_pause field existed: pause = 0
    trainer._open_virtual_step(35.0, [SimpleNamespace(enqueue_time=33.0, validation_pause_before=4.0)])
    assert trainer._step_virtual_start == 29.0

    # old-format samples without any stamps: fall back to the actual ready time
    trainer._open_virtual_step(40.0, [SimpleNamespace()])
    assert trainer._step_virtual_start == 40.0


def test_advance_virtual_clock_noop_between_steps():
    trainer = _make_trainer(first_sample_time=0.0)
    trainer.virtual_free_time = 26.0
    trainer._advance_virtual_clock(now=99.0)  # no open step: must not move
    assert trainer.virtual_free_time == 26.0


def test_add_cumulative_time_metrics_math():
    trainer = _make_trainer(first_sample_time=100.0, cumulative_validation_time=5.0, cumulative_save_time=2.0)
    trainer.virtual_free_time = 140.0
    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=150.0)
    assert step_data[TIMING_PREFIX + "wall_time_since_first_sample"] == 50.0
    assert step_data[TIMING_PREFIX + "cumulative_validation_time"] == 5.0
    assert step_data[TIMING_PREFIX + "cumulative_save_time"] == 2.0
    assert step_data["training/train_time_s"] == 40.0


def test_add_cumulative_time_metrics_noop_without_anchor():
    trainer = _make_trainer(first_sample_time=None)
    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=150.0)
    assert step_data == {}


def test_cache_virtual_time_state():
    trainer = _make_trainer()
    trainer._cache_virtual_time_state(
        {"first_sample_time": 100.0, "cumulative_validation_time": 3.0, "cumulative_checkpoint_pause": 0.5}
    )
    assert trainer.rollouter_first_sample_time == 100.0
    assert trainer.rollouter_cumulative_validation_time == 3.0
    assert trainer.rollouter_cumulative_checkpoint_pause == 0.5
    # a state without an anchor must not clear an existing one
    trainer._cache_virtual_time_state({"first_sample_time": None, "cumulative_validation_time": 4.0})
    assert trainer.rollouter_first_sample_time == 100.0
    assert trainer.rollouter_cumulative_validation_time == 4.0


# ---------------------------------------------------------------- timing state persistence


def test_timing_state_checkpoint_roundtrip(tmp_path):
    trainer = _make_trainer(first_sample_time=100.0, cumulative_validation_time=5.0, cumulative_save_time=2.0)
    trainer.virtual_free_time = 140.0
    trainer._save_timing_state(str(tmp_path), save_start=150.0)

    saved = json.loads((tmp_path / "timing_state.json").read_text())
    assert saved == {
        "wall_time_since_first_sample": 50.0,
        "cumulative_validation_time": 5.0,
        "cumulative_save_time": 2.0,
        "cumulative_training_time": 40.0,
    }

    resumed = _make_trainer()
    resumed._restore_timing_state(str(tmp_path))
    assert resumed.timing_wall_offset == 50.0
    assert resumed.timing_validation_offset == 5.0
    assert resumed.timing_save_offset == 2.0
    assert resumed.virtual_training_time_offset == 40.0

    # after resume, the restored offsets keep the metrics continuous
    resumed.rollouter_first_sample_time = 1000.0
    resumed.virtual_free_time = 1030.0
    step_data = {}
    resumed._add_cumulative_time_metrics(step_data, now=1040.0)
    assert step_data[TIMING_PREFIX + "wall_time_since_first_sample"] == 90.0
    assert step_data["training/train_time_s"] == 70.0


def test_timing_state_save_carries_offsets_forward_without_anchor(tmp_path):
    # A save before the rollouter reports its first draw (e.g. resumed run that
    # checkpoints immediately) must persist the restored offsets unchanged.
    trainer = _make_trainer(first_sample_time=None)
    trainer.timing_wall_offset = 50.0
    trainer.timing_validation_offset = 5.0
    trainer.timing_save_offset = 2.0
    trainer.virtual_training_time_offset = 40.0
    trainer._save_timing_state(str(tmp_path), save_start=999.0)
    saved = json.loads((tmp_path / "timing_state.json").read_text())
    assert saved["wall_time_since_first_sample"] == 50.0
    assert saved["cumulative_training_time"] == 40.0


def test_timing_state_second_resume_chains_totals(tmp_path):
    # run 1 -> checkpoint
    run1 = _make_trainer(first_sample_time=0.0, cumulative_validation_time=4.0, cumulative_save_time=1.0)
    run1.virtual_free_time = 80.0
    run1._save_timing_state(str(tmp_path), save_start=100.0)

    # run 2 resumes, trains 50s more of wall with 3s validation, checkpoints again
    run2 = _make_trainer()
    run2._restore_timing_state(str(tmp_path))
    run2.rollouter_first_sample_time = 1000.0
    run2.rollouter_cumulative_validation_time = 3.0
    run2.cumulative_save_time = 0.5
    run2.virtual_free_time = 1045.0
    run2._save_timing_state(str(tmp_path), save_start=1050.0)

    saved = json.loads((tmp_path / "timing_state.json").read_text())
    assert saved["wall_time_since_first_sample"] == 150.0
    assert saved["cumulative_validation_time"] == 7.0
    assert saved["cumulative_save_time"] == 1.5
    assert saved["cumulative_training_time"] == 80.0 + 45.0


def test_restore_timing_state_missing_file_keeps_zero_offsets(tmp_path):
    trainer = _make_trainer()
    trainer._restore_timing_state(str(tmp_path))
    assert trainer.timing_wall_offset == 0.0
    assert trainer.virtual_training_time_offset == 0.0


def test_restore_timing_state_old_format_falls_back_to_naive(tmp_path):
    # Checkpoints written before the virtual-clock metric may lack the
    # cumulative_training_time key entirely: fall back to wall - val - save.
    (tmp_path / "timing_state.json").write_text(
        json.dumps(
            {"wall_time_since_first_sample": 50.0, "cumulative_validation_time": 5.0, "cumulative_save_time": 2.0}
        )
    )
    trainer = _make_trainer()
    trainer._restore_timing_state(str(tmp_path))
    assert trainer.virtual_training_time_offset == 43.0


def test_restore_timing_state_seeds_from_legacy_train_time(tmp_path):
    # Checkpoints from before timing_state.json existed carry only train_time.pt (the old
    # naive clock): its value seeds the virtual clock so the metric stays continuous.
    trainer = _make_trainer()
    trainer._restore_timing_state(str(tmp_path), legacy_train_time_s=78624.0)
    assert trainer.virtual_training_time_offset == 78624.0
    # a present timing_state.json always wins over the legacy value
    (tmp_path / "timing_state.json").write_text(json.dumps({"cumulative_training_time": 40.0}))
    trainer2 = _make_trainer()
    trainer2._restore_timing_state(str(tmp_path), legacy_train_time_s=78624.0)
    assert trainer2.virtual_training_time_offset == 40.0


# ---------------------------------------------------------------- trainer save accounting


def test_fit_save_checkpoint_accumulates_save_time():
    trainer = _make_trainer(first_sample_time=0.0)
    trainer.config = OmegaConf.create({"trainer": {"save_freq": 5, "esi_redundant_time": 0}})
    trainer.current_param_version = 5
    trainer.last_ckpt_version = 0
    trainer.timing_raw = {}
    trainer._save_checkpoint = lambda: time.sleep(0.02)
    trainer._open_virtual_step(10.0, [_sample(10.0, 0.0)])

    trainer._fit_save_checkpoint()

    assert trainer.cumulative_save_time >= 0.02
    assert trainer._step_save_time >= 0.02
    assert trainer.last_ckpt_version == 5
    # the excluded save time keeps the virtual clock still while wall time advances
    virtual_before_save = 10.0 + (time.time() - 10.0) - trainer._step_save_time
    assert abs(trainer._virtual_now(time.time()) - virtual_before_save) < 0.5
