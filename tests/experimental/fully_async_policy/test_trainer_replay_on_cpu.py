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
"""Trainer side of the replay-buffer recipe (no GPU, no Ray cluster).

- __init__ validation: the replay arm's required objective / sync settings, unsupported custom_vcpo keys,
  trainer.total_training_steps as an update cap
- acquisition: watermark, the smaller all-fresh first mini-batch, the fresh-share gate and its waivers,
  termination on the rollouter's end signal
- the training batch built from frozen statistics, buffer maintenance order, replay/* metrics
- one replay step end to end with the worker calls stubbed: exact mini-batch size, weight sync per update,
  metrics landing at the version the update produced
- the actor-update mini-batch override and the main loop cancelling the rollouter once the trainer is done
"""

import asyncio
import os
from types import SimpleNamespace

import numpy as np
import pytest
import ray.cloudpickle
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy import fully_async_main as main_mod
from verl.experimental.fully_async_policy.detach_utils import RolloutSample
from verl.experimental.fully_async_policy.fully_async_trainer import (
    FullyAsyncTrainer as _TrainerActor,
)
from verl.experimental.fully_async_policy.fully_async_trainer import (
    TrainingStopException,
    check_unsupported_async_keys,
    parse_max_train_steps,
)
from verl.experimental.fully_async_policy.replay_buffer import ReplayBuffer
from verl.protocol import DataProto
from verl.trainer.ppo import ray_trainer as ray_trainer_mod


def _unwrap(actor_cls):
    return actor_cls.__ray_metadata__.modified_class if hasattr(actor_cls, "__ray_metadata__") else actor_cls


FullyAsyncTrainer = _unwrap(_TrainerActor)

CONFIG_DIR = os.path.abspath("verl/experimental/fully_async_policy/config")
N = 4
PROMPT_LEN = 3
RESP_LEN = 6

REPLAY_ARM = [
    "actor_rollout_ref.hybrid_engine=False",
    "trainer.logger=[console]",
    "algorithm.adv_estimator=grpo",
    "actor_rollout_ref.actor.ppo_mini_batch_size=33",
    "actor_rollout_ref.rollout.n=16",
    "trainer.n_gpus_per_node=3",
    "async_training.trigger_parameter_sync_step=1",
    "async_training.replay_buffer.enable=True",
    "async_training.replay_buffer.requires_mini_batches=0.5",
    "async_training.replay_buffer.min_fresh_ratio=0.5",
    "async_training.replay_buffer.reuse_halflife=1",
    "algorithm.rollout_correction.bypass_mode=True",
    "algorithm.rollout_correction.loss_type=reinforce",
    "algorithm.rollout_correction.rollout_is=token",
    "actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode",
    "+actor_rollout_ref.actor.policy_loss.rollout_correction=${algorithm.rollout_correction}",
]


def _compose(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="fully_async_ppo_megatron_trainer", overrides=list(overrides))


def _init_trainer(overrides=(), base=REPLAY_ARM):
    cfg = _compose([*base, *overrides])
    return FullyAsyncTrainer(config=cfg, tokenizer=None, role_worker_mapping={}, resource_pool_manager=None)


# ------------------------------------------------------------------ __init__ validation


class TestInit:
    def test_replay_arm(self):
        t = _init_trainer()
        assert t.replay_enable
        assert t.replay_first_mini_size == 18  # 0.5 x 33 -> 17 -> 18 so 18 x 16 splits over dp=3
        assert t.replay_min_fresh_ratio == 0.5
        assert t.replay_buffer.tau == 8.0 and t.replay_buffer.staleness_threshold == 32
        assert t.replay_buffer.reuse_halflife == 1.0
        assert t.max_train_steps is None

    def test_off_by_default(self):
        t = _init_trainer(base=["actor_rollout_ref.hybrid_engine=False", "trainer.logger=[console]"])
        assert not t.replay_enable
        assert not hasattr(t, "replay_buffer")

    @pytest.mark.parametrize(
        "override, match",
        [
            ("async_training.trigger_parameter_sync_step=2", "trigger_parameter_sync_step"),
            ("async_training.require_batches=2", "require_batches"),
            ("actor_rollout_ref.actor.ppo_epochs=2", "ppo_epochs"),
            ("algorithm.adv_estimator=rloo", "GRPO"),
            ("algorithm.use_kl_in_reward=True", "use_kl_in_reward"),
            ("algorithm.rollout_correction.bypass_mode=False", "bypass_mode=True"),
            ("actor_rollout_ref.actor.policy_loss.loss_mode=vanilla", "loss_mode=bypass_mode"),
            ("algorithm.rollout_correction.loss_type=ppo_clip", "reinforce"),
            ("async_training.use_dynamic_resource_scheduling=True", "dynamic_resource"),
        ],
    )
    def test_rejects_settings_the_replay_loop_does_not_support(self, override, match):
        with pytest.raises(AssertionError, match=match):
            _init_trainer([override])

    def test_rejects_missing_worker_side_rollout_correction(self):
        base = [o for o in REPLAY_ARM if not o.startswith("+actor_rollout_ref.actor.policy_loss.rollout_correction")]
        with pytest.raises(AssertionError, match="reinforce"):
            _init_trainer(base=base)

    def test_min_fresh_ratio_range(self):
        with pytest.raises(AssertionError, match="min_fresh_ratio"):
            _init_trainer(["async_training.replay_buffer.min_fresh_ratio=1.5"])

    def test_step_cap_from_total_training_steps(self):
        t = _init_trainer(["trainer.total_training_steps=5"])
        assert t.max_train_steps == 5
        t.set_total_train_steps(100)
        assert t.total_train_steps == 5
        t.set_total_train_steps(3)
        assert t.total_train_steps == 3

    def test_unsupported_key_rejected_at_init(self):
        with pytest.raises(ValueError, match="dynamic_filtering"):
            _init_trainer(["+async_training.dynamic_filtering.enable=True"])


class TestUnsupportedKeys:
    @pytest.mark.parametrize(
        "cfg",
        [
            {},
            {"dynamic_filtering": {"enable": False, "min_buffered_batches": 1.0}},
            {"opportunistic_epochs": {"enable": False, "max_extra_epochs": 0}},
            {"ppo_epochs": None, "save_queue_state": False, "resumable_ckpts_to_keep": None},
            {"replay_buffer": {"enable": True, "save_state": False}},
            {"resumable_ckpts_to_keep": 0},
        ],
    )
    def test_off_values_pass(self, cfg):
        check_unsupported_async_keys(OmegaConf.create(cfg))

    @pytest.mark.parametrize(
        "cfg, key",
        [
            ({"dynamic_filtering": {"enable": True}}, "dynamic_filtering.enable"),
            ({"opportunistic_epochs": {"enable": True}}, "opportunistic_epochs.enable"),
            ({"ppo_epochs": 2}, "ppo_epochs"),
            ({"save_queue_state": True}, "save_queue_state"),
            ({"replay_buffer": {"save_state": True}}, "replay_buffer.save_state"),
            ({"resumable_ckpts_to_keep": 1}, "resumable_ckpts_to_keep"),
            ({"bsz_per_dp_rank": 33}, "bsz_per_dp_rank"),
        ],
    )
    def test_enabled_values_raise(self, cfg, key):
        with pytest.raises(ValueError, match=key):
            check_unsupported_async_keys(OmegaConf.create(cfg))


class TestParseMaxTrainSteps:
    def test_values(self):
        assert parse_max_train_steps(None) is None
        assert parse_max_train_steps(3) == 3
        assert parse_max_train_steps("7") == 7

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive(self, bad):
        with pytest.raises(ValueError, match="total_training_steps"):
            parse_max_train_steps(bad)


# ------------------------------------------------------------------ a bare trainer with a fake queue


class _Queue:
    """The async MessageQueueClient surface the replay loop uses; ``arrivals`` feed get_sample one by one."""

    def __init__(self, available=(), arrivals=()):
        self.available = list(available)
        self.arrivals = list(arrivals)
        self.drains = 0

    async def get_available_samples(self):
        self.drains += 1
        out, self.available = self.available, []
        return out

    async def get_sample(self):
        if not self.arrivals:
            return None
        return self.arrivals.pop(0), len(self.arrivals)


def _sample(version=0, scores=(1.0, 0.0, 0.0, 1.0)):
    """A pickled RolloutSample as the rollouter enqueues it (after the insertion gate)."""
    n = len(scores)
    lengths = [RESP_LEN, 2, 4, 1][:n]
    mask = torch.zeros(n, RESP_LEN, dtype=torch.long)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    prompts = torch.ones(n, PROMPT_LEN, dtype=torch.long)
    batch = DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": torch.ones(n, RESP_LEN, dtype=torch.long),
            "input_ids": torch.ones(n, PROMPT_LEN + RESP_LEN, dtype=torch.long),
            "attention_mask": torch.cat([torch.ones(n, PROMPT_LEN, dtype=torch.long), mask], dim=1),
            "response_mask": mask,
            "rollout_log_probs": -torch.rand(n, RESP_LEN),
        },
        non_tensors={
            "uid": np.array([f"u{version}"] * n, dtype=object),
            "min_global_steps": np.array([version] * n, dtype=object),
            "max_global_steps": np.array([version] * n, dtype=object),
        },
        meta_info={"metrics": [{"generate_sequences": 0.1, "tool_calls": 0.0}] * n},
    )
    s = np.asarray(scores, dtype=np.float32)
    batch.non_tensor_batch["reward_scalar"] = s
    batch.non_tensor_batch["advantage_scalar"] = ((s - s.mean()) / (s.std(ddof=1) + 1e-6)).astype(np.float32)
    rs = RolloutSample(full_batch=batch, sample_id=f"s{version}", epoch=0, rollout_status={}, group_version=version)
    return ray.cloudpickle.dumps(rs)


def _bare_trainer(mini=4, first=None, rmb=1.0, fresh=0.0, timeout=3600.0, queue=None):
    t = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
    t.required_samples = mini
    t.replay_first_mini_size = first
    t.replay_requires_mini_batches = rmb
    t.replay_min_fresh_ratio = fresh
    t.replay_min_fresh_wait_timeout_s = timeout
    t.replay_fresh_poll_interval_s = 0.0
    t.replay_buffer = ReplayBuffer(tau=8.0, staleness_threshold=32, seed=0)
    t.replay_updates_done = 0
    t.rollout_done = False
    t.current_param_version = 0
    t._replay_fresh_wait_s = 0.0
    t._replay_fresh_floor_waived = 0
    t.message_queue_client = queue or _Queue()
    t.virtual_free_time = None
    t._step_virtual_start = None
    t._step_actual_start = None
    t._step_wait_valid_time = 0.0
    t._step_save_time = 0.0
    t.cumulative_save_time = 0.0
    t.config = OmegaConf.create(
        {
            "trainer": {"balance_batch": False},
            "actor_rollout_ref": {"rollout": {"temperature": 0.7, "n": N, "multi_turn": {"enable": False}}},
        }
    )
    t.tokenizer = None
    return t


def _run(coro):
    return asyncio.run(coro)


class TestAcquire:
    def test_first_minibatch_is_small_and_all_fresh_then_full_size(self):
        q = _Queue(available=[_sample(0) for _ in range(3)], arrivals=[_sample(0) for _ in range(10)])
        t = _bare_trainer(mini=4, first=2, rmb=0.5, queue=q)
        entries, info = _run(t._acquire_replay_minibatch())
        assert len(entries) == 2 and info["n_new"] == 2 and info["n_replayed"] == 0
        t.replay_updates_done = 1
        t._replay_post_update_maintenance(entries, 1)
        entries, info = _run(t._acquire_replay_minibatch())
        assert len(entries) == 4  # full size from the second update on

    def test_first_size_only_before_any_update(self):
        t = _bare_trainer(mini=4, first=2, rmb=0.5)
        assert t._replay_minibatch_size() == 2
        t.replay_updates_done = 1
        assert t._replay_minibatch_size() == 4
        t2 = _bare_trainer(mini=4, first=None)
        assert t2._replay_minibatch_size() == 4

    def test_waits_for_the_watermark(self):
        q = _Queue(available=[_sample(0)], arrivals=[_sample(0) for _ in range(7)])
        t = _bare_trainer(mini=2, rmb=2.0, queue=q)
        entries, _ = _run(t._acquire_replay_minibatch())
        assert t.replay_buffer.size() == 4  # waited for 2 x 2 groups before composing
        assert len(entries) == 2
        assert len(q.arrivals) == 4

    def test_stops_on_the_end_signal_below_the_watermark(self):
        q = _Queue(available=[_sample(0), None])
        t = _bare_trainer(mini=4, queue=q)
        assert _run(t._acquire_replay_minibatch()) == (None, None)
        assert t.rollout_done

    def test_drains_the_tail_after_the_end_signal(self):
        q = _Queue(available=[_sample(0) for _ in range(5)] + [None])
        t = _bare_trainer(mini=4, queue=q)
        entries, _ = _run(t._acquire_replay_minibatch())
        assert len(entries) == 4 and t.rollout_done

    def test_groups_are_added_at_the_current_version(self):
        q = _Queue(available=[_sample(3)])
        t = _bare_trainer(mini=1, queue=q)
        t.current_param_version = 5
        entries, info = _run(t._acquire_replay_minibatch())
        assert info["staleness"] == [2]
        assert entries[0].score == pytest.approx(2.0 ** (-2 / 8))  # scored at the version it arrived under


class TestFreshGate:
    def _primed(self, mini=4, fresh=0.5, **kw):
        t = _bare_trainer(mini=mini, fresh=fresh, **kw)
        for _ in range(mini):
            t.replay_buffer.add(ray.cloudpickle.loads(_sample(0)), 0)
        t.replay_buffer.compose_minibatch(mini, 0)  # consume their freshness
        return t

    @pytest.mark.parametrize(
        "ratio, mini, expected", [(0.0, 33, 0), (0.5, 33, 17), (1 / 3, 3, 1), (0.34, 3, 2), (1.0, 33, 33)]
    )
    def test_floor(self, ratio, mini, expected):
        assert _bare_trainer(fresh=ratio)._replay_min_fresh_groups(mini) == expected

    def test_waits_until_enough_groups_arrived(self):
        t = self._primed()
        arrivals = [[_sample(0)], [], [_sample(0)]]

        async def drain():
            return arrivals.pop(0) if arrivals else []

        t.message_queue_client.get_available_samples = drain
        entries, info = _run(t._acquire_replay_minibatch())
        assert info["n_new"] == 2  # floor ceil(0.5 x 4)
        assert t._replay_fresh_floor_waived == 0

    def test_off_never_polls(self):
        t = self._primed(fresh=0.0)
        _run(t._acquire_replay_minibatch())
        assert t.message_queue_client.drains == 1  # the acquire's own drain only

    def test_waived_when_the_rollouter_is_done(self):
        # the cap is a safety bound: the end signal must waive the floor at once, not after the cap
        t = self._primed(timeout=2.0)
        t.message_queue_client.available = [None]
        entries, info = _run(t._acquire_replay_minibatch())
        assert info["n_new"] == 0 and t._replay_fresh_floor_waived == 1
        assert t._replay_fresh_wait_s < 1.0

    def test_waived_on_the_wall_clock_cap(self):
        t = self._primed(timeout=1e-9)
        polls = []

        async def drain():
            # safety bound: end the rollout after 200 polls; the cap must waive long before that
            polls.append(1)
            return [None] if len(polls) >= 200 else []

        t.message_queue_client.get_available_samples = drain
        _run(t._acquire_replay_minibatch())
        assert t._replay_fresh_floor_waived == 1
        assert len(polls) < 200


class TestBatchAndMaintenance:
    def test_batch_uses_frozen_statistics(self):
        t = _bare_trainer()
        entries = [SimpleNamespace(sample=ray.cloudpickle.loads(_sample(0, scores=(1.0, 0.0, 0.0, 1.0))))]
        batch = t._build_replay_batch(entries)
        mask = batch.batch["response_mask"].float()
        adv = torch.as_tensor(batch.non_tensor_batch["advantage_scalar"])
        torch.testing.assert_close(batch.batch["advantages"], adv[:, None] * mask)
        torch.testing.assert_close(batch.batch["returns"], batch.batch["advantages"])
        scores = batch.batch["token_level_scores"]
        # the reward sits on the last response token of each trajectory, nowhere else
        assert scores.sum(-1).tolist() == [1.0, 0.0, 0.0, 1.0]
        assert scores[0, RESP_LEN - 1] == 1.0 and scores[3, 0] == 1.0
        torch.testing.assert_close(batch.batch["token_level_rewards"], scores)
        assert batch.meta_info["temperature"] == 0.7
        assert "old_log_probs" not in batch.batch  # set by bypass mode, not recomputed

    def test_a_group_can_be_assembled_again(self):
        t = _bare_trainer()
        entry = SimpleNamespace(sample=ray.cloudpickle.loads(_sample(0)))
        first = t._build_replay_batch([entry])
        second = t._build_replay_batch([entry])
        torch.testing.assert_close(first.batch["advantages"], second.batch["advantages"])

    def test_maintenance_marks_before_evicting(self):
        t = _bare_trainer()
        t.replay_buffer = ReplayBuffer(tau=8.0, staleness_threshold=1, seed=0)
        e = t.replay_buffer.add(SimpleNamespace(group_version=0), 0)
        t._replay_post_update_maintenance([e], new_version=2)
        assert t.replay_buffer.evicted_unseen_total == 0  # trained, then evicted
        assert t.replay_buffer.evicted_trained_once_total == 1

    def test_maintenance_rescores_at_the_new_version(self):
        t = _bare_trainer()
        e = t.replay_buffer.add(SimpleNamespace(group_version=0), 0)
        t._replay_post_update_maintenance([], new_version=8)
        assert e.score == pytest.approx(0.5)

    def test_metrics(self):
        t = _bare_trainer(mini=3, fresh=0.5)
        for v in (5, 3, 4):
            t.replay_buffer.add(SimpleNamespace(group_version=v), 5)
        t._replay_fresh_wait_s, t._replay_fresh_floor_waived = 2.5, 1
        t.replay_updates_done = 7
        info = {
            "n_new": 2,
            "n_replayed": 1,
            "staleness": [1, 2, 4],
            "fresh_staleness": [1, 2],
            "times_trained": [0, 0, 3],
        }
        m = {}
        t._add_replay_metrics(m, info, new_version=6)
        assert m["replay/minibatch_size"] == 3 and m["replay/minibatch_new"] == 2
        assert m["replay/minibatch_new_ratio"] == pytest.approx(2 / 3)
        assert m["replay/minibatch_staleness_mean"] == pytest.approx(7 / 3)
        assert m["replay/minibatch_staleness_p50"] == 2
        assert m["replay/minibatch_fresh_staleness_max"] == 2
        assert m["replay/minibatch_replayed_staleness_mean"] == 4
        assert m["replay/minibatch_times_trained_max"] == 3
        assert m["replay/buffer_size"] == 3 and m["replay/buffer_max_staleness"] == 3
        assert m["replay/fresh_floor"] == 2 and m["replay/fresh_wait_s"] == 2.5 and m["replay/fresh_floor_waived"] == 1
        assert m["replay/updates_done"] == 7
        assert all(isinstance(v, int | float) for v in m.values())  # MetricsAggregator drops anything else


# ------------------------------------------------------------------ one replay step


class TestReplayStep:
    def _trainer(self, entries=True):
        t = _bare_trainer(mini=2)
        calls = []
        t.calls = calls
        t.global_steps = 1
        t.epoch = 0
        t.local_trigger_step = 1
        t.trigger_parameter_sync_step = 1
        t.config = OmegaConf.merge(t.config, {"global_profiler": {"steps": None}})
        samples = [ray.cloudpickle.loads(_sample(0)) for _ in range(2)]
        info = {"n_new": 2, "n_replayed": 0, "staleness": [0, 0], "fresh_staleness": [0, 0], "times_trained": [0, 0]}

        async def acquire():
            if not entries:
                return None, None
            chosen = [t.replay_buffer.add(s, 0) for s in samples]
            return chosen, info

        t._acquire_replay_minibatch = acquire
        t._collect_metrics_from_samples = lambda batch, metrics: None
        t._fit_start_profile = lambda **k: None
        t._fit_stop_profile = lambda **k: None
        t._fit_compute_log_prob = lambda batch: calls.append("log_prob") or batch

        def update_actor(batch, mini_batch_size=None):
            calls.append(("update_actor", len(batch), mini_batch_size))
            return DataProto(meta_info={"metrics": {"actor/pg_loss": [0.1]}})

        t._update_actor = update_actor

        async def update_weights():
            calls.append(("update_weights", t.current_param_version))
            return {"x": 1}

        t._fit_update_weights = update_weights
        t._fit_dump_data = lambda batch: None
        t._record_train_resource_utilization = lambda allocated_time: None

        async def validate():
            calls.append("validate")

        t._fit_validate = validate
        t._fit_save_checkpoint = lambda: calls.append("save")
        t._fit_collect_metrics = lambda batch: None
        t._fit_postprocess_step = lambda: calls.append(("postprocess", dict(t.metrics)))
        t._fit_log_aggregated_training_metrics = lambda timing: calls.append("log")
        return t

    def test_one_update(self):
        t = self._trainer()
        _run(t._fit_replay_step())
        names = [c if isinstance(c, str) else c[0] for c in t.calls]
        assert names == ["log_prob", "update_actor", "update_weights", "validate", "save", "postprocess", "log"]
        assert t.calls[1] == ("update_actor", 2 * N, 2 * N)  # the whole batch is one exact mini-batch
        assert t.calls[2] == ("update_weights", 1)  # synced after the update, at its version
        assert t.current_param_version == 1 and t.replay_updates_done == 1
        assert all(e.times_trained == 1 for e in t.replay_buffer.entries)
        # maintenance ran at the version the update produced (1), not the one it trained from (0)
        assert all(e.score == pytest.approx(2.0 ** (-1 / 8)) for e in t.replay_buffer.entries)
        metrics = t.calls[5][1]
        assert metrics["actor/pg_loss"] == pytest.approx(0.1)
        assert metrics["replay/minibatch_size"] == 2

    def test_end_signal_stops_training(self):
        t = self._trainer(entries=False)
        with pytest.raises(TrainingStopException):
            _run(t._fit_replay_step())
        assert t.calls == []


class TestFitLoop:
    def _trainer(self, cap, replay=True, stop_after=None):
        t = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
        t.message_queue_client = object()
        t.rollouter = object()
        t.global_steps = 0
        t.max_train_steps = cap
        t.stopped_by_step_cap = False
        t.replay_enable = replay
        t.current_param_version = 0
        t.local_trigger_step = 1
        t.config = OmegaConf.create({"trainer": {"test_freq": 1}})
        t.progress_bar = SimpleNamespace(close=lambda: None)
        t.steps = []

        async def step():
            if stop_after is not None and len(t.steps) >= stop_after:
                raise TrainingStopException()
            t.steps.append("replay" if replay else "fifo")
            t.global_steps += 1
            t.current_param_version += 1

        t._fit_replay_step = step
        t.fit_step = step
        t.saved = []
        t._fit_save_checkpoint = lambda force=False: t.saved.append(force)
        return t

    def test_cap_stops_after_exactly_n_updates(self):
        t = self._trainer(cap=3)
        _run(t.fit())
        assert t.steps == ["replay"] * 3
        assert t.stopped_by_step_cap
        assert t.saved == [True]  # final save forced

    def test_no_cap_runs_until_the_end_signal(self):
        t = self._trainer(cap=None, stop_after=5)
        _run(t.fit())
        assert t.steps == ["replay"] * 5 and not t.stopped_by_step_cap

    def test_fifo_path_when_replay_is_off(self):
        t = self._trainer(cap=2, replay=False)
        _run(t.fit())
        assert t.steps == ["fifo"] * 2


# ------------------------------------------------------------------ actor update override and main loop


class TestUpdateActorMiniBatch:
    def _run(self, monkeypatch, mini_batch_size):
        monkeypatch.setattr(ray_trainer_mod, "left_right_2_no_padding", lambda td: td)
        captured = {}

        class _WG:
            def update_actor(self, td):
                captured.update(global_batch_size=td["global_batch_size"], mini_batch_size=td["mini_batch_size"])
                return ray_trainer_mod.tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {"mfu": 0.5}})

        t = ray_trainer_mod.RayPPOTrainer.__new__(ray_trainer_mod.RayPPOTrainer)
        t.actor_rollout_wg = _WG()
        t.config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {"n": 16, "temperature": 1.0, "multi_turn": {"enable": False}},
                    "actor": {
                        "ppo_mini_batch_size": 33,
                        "ppo_epochs": 1,
                        "data_loader_seed": 0,
                        "shuffle": False,
                        "calculate_entropy": False,
                        "entropy_coeff": 0.0,
                    },
                }
            }
        )
        batch = DataProto.from_dict(tensors={"x": torch.zeros(288, 1)})
        t._update_actor(batch, mini_batch_size=mini_batch_size)
        return captured

    def test_default_is_the_configured_mini_batch(self, monkeypatch):
        assert self._run(monkeypatch, None) == {"global_batch_size": 528, "mini_batch_size": 528}

    def test_override(self, monkeypatch):
        assert self._run(monkeypatch, 288) == {"global_batch_size": 288, "mini_batch_size": 288}


class TestMainLoopCancelsTheRollouter:
    def _run(self, monkeypatch, order):
        """``order``: which component finishes first."""
        cancelled = []
        rollouter_f, trainer_f = "rollouter_future", "trainer_future"
        pending = [rollouter_f, trainer_f]

        def wait(futures, num_returns=1, timeout=None):
            name = order.pop(0) if order else futures[0]
            done = [f for f in futures if f.startswith(name)]
            return done, [f for f in futures if f not in done]

        monkeypatch.setattr(main_mod.ray, "wait", wait)
        monkeypatch.setattr(main_mod.ray, "get", lambda f: None)
        monkeypatch.setattr(main_mod.ray, "cancel", lambda f: cancelled.append(f))

        async def clear():
            return None

        runner_cls = _unwrap(main_mod.FullyAsyncTaskRunner)
        runner = runner_cls.__new__(runner_cls)
        runner.components = {
            "rollouter": SimpleNamespace(fit=SimpleNamespace(remote=lambda: rollouter_f)),
            "trainer": SimpleNamespace(fit=SimpleNamespace(remote=lambda: trainer_f)),
            "message_queue_client": SimpleNamespace(clear_queue=clear),
        }
        runner._run_training_loop()
        return cancelled, pending

    def test_trainer_first_cancels_the_rollouter(self, monkeypatch):
        cancelled, _ = self._run(monkeypatch, ["trainer"])
        assert cancelled == ["rollouter_future"]

    def test_rollouter_first_is_the_normal_end(self, monkeypatch):
        cancelled, _ = self._run(monkeypatch, ["rollouter", "trainer"])
        assert cancelled == []
