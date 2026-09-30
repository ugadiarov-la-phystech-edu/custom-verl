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
"""Rollouter side of the replay-buffer recipe (no GPU, no Ray cluster).

- insertion gate: frozen GRPO statistics match compute_grpo_outcome_advantage, degenerate groups are
  dropped and counted, the staleness quota slot is released, group_version stamping
- concurrency ramp wired into the dispatch cap
- stop-the-world pauses (serialized validation, checkpoint saves): freeze order, nesting, and that
  neither the monitor loop nor reset_staleness resumes generation underneath them
- MessageQueue.get_available_samples non-blocking drain
- the async_training keys and their off defaults, and the __init__ validation
"""

import asyncio
import inspect
import os
from types import SimpleNamespace

import numpy as np
import pytest
import ray.cloudpickle
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy import fully_async_rollouter as rollouter_mod
from verl.experimental.fully_async_policy.detach_utils import RolloutSample
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter as _RollouterActor
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer as _TrainerActor
from verl.experimental.fully_async_policy.message_queue import MessageQueue as _MessageQueueActor
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def _unwrap(actor_cls):
    return actor_cls.__ray_metadata__.modified_class if hasattr(actor_cls, "__ray_metadata__") else actor_cls


FullyAsyncRollouter = _unwrap(_RollouterActor)
FullyAsyncTrainer = _unwrap(_TrainerActor)
MessageQueue = _unwrap(_MessageQueueActor)

CONFIG_DIR = os.path.abspath("verl/experimental/fully_async_policy/config")
N = 4
RESP_LEN = 5


def _compose(config_name="fully_async_ppo_trainer", overrides=()):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


# ------------------------------------------------------------------ fixtures


class _Events(list):
    pass


class _Remote:
    """``obj.method.remote(...)`` returning an awaitable, recording the call."""

    def __init__(self, events, name, result=None):
        self.events, self.name, self.result = events, name, result

    def remote(self, *args, **kwargs):
        self.events.append((self.name, *args))

        async def _done():
            return self.result

        return _done()


class _Replica:
    def __init__(self, events, idx):
        self.events, self.idx = events, idx

    async def abort_all_requests(self):
        self.events.append(("abort", self.idx))

    async def resume_generation(self):
        self.events.append(("resume_generation", self.idx))


def _server_manager(events, n_replicas=2):
    lb = SimpleNamespace(set_hold=_Remote(events, "set_hold"))
    replicas = [_Replica(events, i) for i in range(n_replicas)]
    return SimpleNamespace(
        global_load_balancer=lb,
        get_replicas=lambda: replicas,
        get_active_server_count=lambda: n_replicas,
    )


class _Queue:
    def __init__(self, size=0):
        self.put = []
        self.size = size

    async def put_sample(self, sample):
        self.put.append(sample)
        return True

    async def get_queue_size(self):
        return self.size

    async def get_statistics(self):
        return {"queue_size": self.size}


def _bare_rollouter(events=None, replay=True, norm=True, ramp=(), serialize=False, save_pause=False):
    """A rollouter without __init__: only the state the tested methods touch."""
    r = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
    events = _Events() if events is None else events
    r.events = events
    r.replay_mode = replay
    r.norm_adv_by_std_in_grpo = norm
    r.concurrency_ramp = list(ramp)
    r.ramp_first_size = 18
    r.required_samples = 33
    r.concurrent_samples_per_replica = 33
    r.max_concurrent_samples = 165
    r.max_required_samples = 1089
    r.staleness_threshold = 32
    r.max_queue_size = 1089
    r._ramp_cap_logged = None
    r.serialize_validation = serialize
    r.pause_generation_during_save = save_pause
    r._hard_pause_reasons = set()
    r.first_sample_time = None
    r.cumulative_validation_time = 0.0
    r.cumulative_checkpoint_pause = 0.0
    r._save_pause_start = None
    for name in (
        "groups_completed_total",
        "all_correct_groups_total",
        "all_wrong_groups_total",
        "groups_completed_window",
        "all_correct_groups_window",
        "all_wrong_groups_window",
        "filtered_degenerate_groups",
        "total_generated_samples",
        "staleness_samples",
        "dropped_stale_samples",
        "processed_sample_count",
        "_step_generated_samples",
    ):
        setattr(r, name, 0)
    r.active_tasks = set()
    r.pending_queue = asyncio.Queue()
    r.paused = False
    r.running = True
    r.llm_server_manager = _server_manager(events)
    r.message_queue_client = _Queue()
    r._init_async_objects()
    return r


def _group(scores, start_versions=None, with_rewards=True):
    """A completed group as the agent loop returns it: sparse rm_scores on the last response token."""
    n = len(scores)
    batch = {"response_mask": torch.ones(n, RESP_LEN)}
    if with_rewards:
        rm = torch.zeros(n, RESP_LEN)
        rm[:, -1] = torch.tensor(scores, dtype=torch.float32)
        batch["rm_scores"] = rm
    non_tensor = {"uid": np.array(["g"] * n, dtype=object)}
    if start_versions is not None:
        non_tensor["min_global_steps"] = np.array(start_versions, dtype=object)
    return DataProto.from_dict(tensors=batch, non_tensors=non_tensor)


def _rollout_sample(scores, start_versions=None, with_rewards=True):
    return RolloutSample(
        full_batch=_group(scores, start_versions, with_rewards), sample_id="s0", epoch=0, rollout_status={}
    )


# ------------------------------------------------------------------ insertion gate


class TestInsertionGate:
    @pytest.mark.parametrize("norm", [True, False])
    def test_frozen_advantages_match_grpo(self, norm):
        scores = [1.0, 0.0, 0.0, 1.0, 0.5]
        r = _bare_rollouter(norm=norm)
        rs = _rollout_sample(scores, start_versions=[3] * 5)
        assert r._prepare_replay_group(rs)
        rewards = torch.zeros(5, RESP_LEN)
        rewards[:, -1] = torch.tensor(scores)
        ref_adv, _ = compute_grpo_outcome_advantage(
            token_level_rewards=rewards,
            response_mask=torch.ones(5, RESP_LEN),
            index=np.array(["g"] * 5, dtype=object),
            norm_adv_by_std_in_grpo=norm,
        )
        nt = rs.full_batch.non_tensor_batch
        np.testing.assert_allclose(nt["advantage_scalar"], ref_adv[:, 0].numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(nt["reward_scalar"], scores)
        assert nt["advantage_scalar"].dtype == np.float32
        assert nt["reward_scalar"].dtype == np.float32

    @pytest.mark.parametrize(
        "scores, correct, wrong", [([1.0] * N, 1, 0), ([0.0] * N, 0, 1), ([-1.0] * N, 0, 1), ([2.5] * N, 1, 0)]
    )
    def test_degenerate_groups_are_dropped_and_classified(self, scores, correct, wrong):
        r = _bare_rollouter()
        assert not r._prepare_replay_group(_rollout_sample(scores))
        assert (r.all_correct_groups_total, r.all_wrong_groups_total) == (correct, wrong)
        assert (r.all_correct_groups_window, r.all_wrong_groups_window) == (correct, wrong)
        assert r.groups_completed_total == r.groups_completed_window == 1

    def test_kept_group_counts_as_completed_only(self):
        r = _bare_rollouter()
        assert r._prepare_replay_group(_rollout_sample([1.0, 0.0, 0.0, 0.0]))
        assert r.groups_completed_total == 1
        assert r.all_correct_groups_total == r.all_wrong_groups_total == 0

    def test_group_without_rewards_is_dropped(self):
        r = _bare_rollouter()
        assert not r._prepare_replay_group(_rollout_sample([1.0, 0.0], with_rewards=False))
        assert r.groups_completed_total == 1

    def test_single_trajectory_group_is_dropped(self):
        assert not _bare_rollouter()._prepare_replay_group(_rollout_sample([1.0]))

    @pytest.mark.parametrize(
        "versions, expected", [([5, 3, 4, 5], 3), ([7, 7, 7, 7], 7), ([None, 2, 2, 2], 0), (None, 0)]
    )
    def test_group_version_is_the_oldest_start(self, versions, expected):
        rs = _rollout_sample([1.0, 0.0, 1.0, 0.0], start_versions=versions)
        assert _bare_rollouter()._prepare_replay_group(rs)
        assert rs.group_version == expected

    def test_rewards_are_summed_over_tokens(self):
        # rm_scores may spread a reward over tokens; the scalar is the per-trajectory sum
        rs = _rollout_sample([1.0, 0.0, 0.0, 0.0])
        rs.full_batch.batch["rm_scores"][0, 0] = 0.5
        assert _bare_rollouter()._prepare_replay_group(rs)
        np.testing.assert_allclose(rs.full_batch.non_tensor_batch["reward_scalar"], [1.5, 0, 0, 0])

    def test_rollout_sample_default_group_version(self):
        assert RolloutSample(full_batch=None, sample_id="x", epoch=0, rollout_status={}).group_version == 0


class _AgentLoop:
    def __init__(self, scores):
        self.scores = scores

    async def generate_sequences_single(self, batch):
        return _group(self.scores, start_versions=[2] * len(self.scores))


class TestProcessSample:
    def _run(self, r, scores):
        r.async_rollout_manager = _AgentLoop(scores)
        rs = RolloutSample(full_batch=_group(scores), sample_id="s1", epoch=0, rollout_status={})
        asyncio.run(r._process_single_sample_streaming(rs))
        return rs

    def test_dropped_group_releases_its_quota_slot(self):
        r = _bare_rollouter()
        r.staleness_samples = 10
        self._run(r, [1.0] * N)
        assert r.message_queue_client.put == []
        assert r.staleness_samples == 9
        assert r.filtered_degenerate_groups == 1
        assert r.processed_sample_count == 1
        assert r.total_generated_samples == 0

    def test_kept_group_is_enqueued_with_frozen_statistics(self):
        r = _bare_rollouter()
        r.staleness_samples = 10
        self._run(r, [1.0, 0.0, 0.0, 0.0])
        assert len(r.message_queue_client.put) == 1
        sent = ray.cloudpickle.loads(r.message_queue_client.put[0])
        assert sent.group_version == 2
        assert "advantage_scalar" in sent.full_batch.non_tensor_batch
        assert r.staleness_samples == 10
        assert r.total_generated_samples == 1
        assert r.filtered_degenerate_groups == 0

    def test_off_enqueues_degenerate_groups(self):
        r = _bare_rollouter(replay=False)
        self._run(r, [1.0] * N)
        assert len(r.message_queue_client.put) == 1
        assert r.groups_completed_total == 0

    def test_statistics_report_filtered_groups_and_the_cap(self):
        r = _bare_rollouter(ramp=[4, 8])
        r.filtered_degenerate_groups = 3
        stats = asyncio.run(r.get_statistics())
        assert stats["count/filtered_degenerate_groups"] == 3
        assert stats["concurrency_cap"] == 8  # 4 per replica x 2 replicas


class TestGroupOutcomeMetrics:
    def test_ratios_and_window_reset(self):
        r = _bare_rollouter()
        for scores in ([1.0] * N, [0.0] * N, [0.0] * N, [1.0, 0.0, 0.0, 0.0]):
            r._prepare_replay_group(_rollout_sample(scores))
        m = r._pop_group_outcome_metrics()
        assert m["fully_async/groups/completed_total"] == 4
        assert m["fully_async/groups/all_correct_ratio"] == pytest.approx(0.25)
        assert m["fully_async/groups/all_wrong_ratio"] == pytest.approx(0.5)
        assert m["fully_async/groups/all_correct_ratio_total"] == pytest.approx(0.25)
        r._prepare_replay_group(_rollout_sample([1.0] * N))
        m = r._pop_group_outcome_metrics()
        assert m["fully_async/groups/all_correct_ratio"] == pytest.approx(1.0)  # window: 1 group
        assert m["fully_async/groups/all_correct_ratio_total"] == pytest.approx(0.4)  # 2 of 5

    def test_empty_window_omits_window_ratios(self):
        m = _bare_rollouter()._pop_group_outcome_metrics()
        assert m == {"fully_async/groups/completed_total": 0}

    @pytest.mark.parametrize("replay", [True, False])
    def test_reset_staleness_reports_them_in_replay_mode(self, replay):
        r = _bare_rollouter(replay=replay)
        r.idle_start_time = r.step_start_time = 0.0
        r._completed_steps = 1
        r._step_samples_history = []
        r._STEP_HISTORY_SIZE = 10
        r._active_count_history = [(0.0, 0, 1)]
        r._prepare_replay_group(_rollout_sample([0.0] * N))
        timing = asyncio.run(r.reset_staleness())
        assert ("fully_async/groups/all_wrong_ratio" in timing) is replay


# ------------------------------------------------------------------ concurrency ramp


class TestConcurrencyCap:
    def test_off_is_max_concurrent_samples(self):
        r = _bare_rollouter()
        r.total_generated_samples = 0
        assert r._concurrency_cap() == 165

    @pytest.mark.parametrize("delivered, expected", [(0, 10), (17, 10), (18, 24), (50, 24), (51, 40), (84, 165)])
    def test_follows_deliveries(self, delivered, expected):
        r = _bare_rollouter(ramp=[5, 12, 20])
        r.total_generated_samples = delivered
        assert r._concurrency_cap() == expected

    def test_bounded_by_the_staleness_quota(self):
        r = _bare_rollouter(ramp=[5, 12, 20])
        r.max_required_samples = 7
        assert r._concurrency_cap() == 7

    def test_dispatch_loop_uses_the_dynamic_cap(self):
        src = inspect.getsource(FullyAsyncRollouter._processor_worker)
        assert "len(self.active_tasks) >= self._concurrency_cap()" in src
        assert ">= self.max_concurrent_samples" not in src

    def test_follows_dynamic_max_concurrent_samples(self):
        r = _bare_rollouter(ramp=[5])
        r.total_generated_samples = 100
        r.max_concurrent_samples = 99
        assert r._concurrency_cap() == 99


# ------------------------------------------------------------------ stop-the-world pauses


class TestHardPause:
    def test_freeze_order(self):
        r = _bare_rollouter()
        asyncio.run(r.begin_save_pause())
        # hold first, so no aborted request can resubmit before the hold is visible
        assert list(r.events) == [
            ("set_hold", True),
            ("abort", 0),
            ("abort", 1),
            ("resume_generation", 0),
            ("resume_generation", 1),
        ]
        assert r.paused and not r._resume_event.is_set()
        assert r.is_hard_paused()

    def test_end_lifts_the_hold_and_resumes(self):
        r = _bare_rollouter()
        asyncio.run(r.begin_save_pause())
        r.events.clear()
        asyncio.run(r.end_save_pause())
        assert list(r.events) == [("set_hold", False)]
        assert not r.paused and r._resume_event.is_set()
        assert not r.is_hard_paused()

    def test_nested_pauses_freeze_once_and_resume_after_the_last(self):
        r = _bare_rollouter()

        async def scenario():
            await r._begin_hard_pause("validation")
            await r._begin_hard_pause("save")
            n_after_begin = len(r.events)
            await r._end_hard_pause("validation")
            still = r.paused and r.is_hard_paused()
            await r._end_hard_pause("save")
            return n_after_begin, still

        n_after_begin, still = asyncio.run(scenario())
        assert n_after_begin == 5  # one freeze
        assert still
        assert not r.paused
        assert r.events.count(("set_hold", False)) == 1

    def test_reset_staleness_does_not_resume_under_a_hard_pause(self):
        r = _bare_rollouter()
        r.idle_start_time = r.step_start_time = 0.0
        r._completed_steps = 1
        r._step_samples_history = []
        r._STEP_HISTORY_SIZE = 10
        r._active_count_history = [(0.0, 0, 1)]
        asyncio.run(r.begin_save_pause())
        asyncio.run(r.reset_staleness())
        assert r.paused and not r._resume_event.is_set()

    def test_monitor_loop_does_not_resume_under_a_hard_pause(self, monkeypatch):
        r = _bare_rollouter()
        asyncio.run(r.begin_save_pause())

        async def one_tick(_delay):
            r.running = False  # stop after this iteration

        monkeypatch.setattr(rollouter_mod.asyncio, "sleep", one_tick)

        async def never_pause():
            return False

        r._should_pause_generation = never_pause
        asyncio.run(r._async_monitor_loop())
        assert r.paused

    def test_monitor_loop_still_resumes_ordinary_pauses(self, monkeypatch):
        r = _bare_rollouter()
        r.paused = True

        async def one_tick(_delay):
            r.running = False

        monkeypatch.setattr(rollouter_mod.asyncio, "sleep", one_tick)

        async def never_pause():
            return False

        r._should_pause_generation = never_pause
        asyncio.run(r._async_monitor_loop())
        assert not r.paused

    @pytest.mark.parametrize("serialize", [True, False])
    def test_validation_is_serialized_only_when_enabled(self, serialize):
        r = _bare_rollouter(serialize=serialize)
        seen = {}

        def validate():
            seen["hard_paused"] = r.is_hard_paused()
            seen["held"] = ("set_hold", True) in r.events
            return {"val/score": 1.0}

        r._validate = validate
        out = asyncio.run(r.do_validate())
        assert out["val/score"] == 1.0
        assert "rollouter/validate_time" in out
        assert seen == {"hard_paused": serialize, "held": serialize}
        assert not r.is_hard_paused()
        assert not r.paused

    def test_validation_failure_still_lifts_the_pause(self):
        r = _bare_rollouter(serialize=True)

        def boom():
            raise RuntimeError("validation failed")

        r._validate = boom
        with pytest.raises(RuntimeError, match="validation failed"):
            asyncio.run(r.do_validate())
        assert not r.is_hard_paused()
        assert ("set_hold", False) in r.events


class TestTrainerSavePause:
    def _trainer(self, enabled, fail=False):
        t = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
        events = []
        t.config = OmegaConf.create({"async_training": {"pause_generation_during_save": enabled}})

        class _Rollouter:
            begin_save_pause = SimpleNamespace(remote=lambda: events.append("begin"))
            end_save_pause = SimpleNamespace(remote=lambda: events.append("end"))

        t.rollouter = _Rollouter()

        def inner():
            events.append("save")
            if fail:
                raise RuntimeError("disk full")

        t._save_checkpoint_inner = inner
        return t, events

    def test_off(self, monkeypatch):
        monkeypatch.setattr("ray.get", lambda x: x)
        t, events = self._trainer(False)
        t._save_checkpoint()
        assert events == ["save"]

    def test_brackets_the_whole_save(self, monkeypatch):
        monkeypatch.setattr("ray.get", lambda x: x)
        t, events = self._trainer(True)
        t._save_checkpoint()
        assert events == ["begin", "save", "end"]

    def test_resumes_after_a_failed_save(self, monkeypatch):
        monkeypatch.setattr("ray.get", lambda x: x)
        t, events = self._trainer(True, fail=True)
        with pytest.raises(RuntimeError, match="disk full"):
            t._save_checkpoint()
        assert events == ["begin", "save", "end"]


# ------------------------------------------------------------------ message queue drain


class TestGetAvailableSamples:
    def _queue(self):
        return MessageQueue(OmegaConf.create({}), max_queue_size=10)

    def test_drains_in_order_including_the_sentinel(self):
        q = self._queue()

        async def scenario():
            for x in (b"a", b"b"):
                await q.put_sample(x)
            await q.put_sample(None)
            drained = await q.get_available_samples()
            again = await q.get_available_samples()
            return drained, again, await q.get_queue_size()

        drained, again, size = asyncio.run(scenario())
        assert drained == [b"a", b"b", None]
        assert again == []
        assert size == 0
        assert q.total_consumed == 3

    def test_empty_queue_returns_immediately(self):
        assert asyncio.run(self._queue().get_available_samples()) == []


# ------------------------------------------------------------------ config and __init__ validation


@pytest.mark.parametrize("config_name", ["fully_async_ppo_trainer", "fully_async_ppo_megatron_trainer"])
def test_new_async_training_keys_default_off(config_name):
    at = _compose(config_name).async_training
    assert at.concurrency_ramp is None
    assert at.serialize_validation is False
    assert at.pause_generation_during_save is False
    assert OmegaConf.to_container(at.replay_buffer) == {
        "enable": False,
        "tau": 8.0,
        "staleness_threshold": 32,
        "requires_mini_batches": 1.0,
        "sampling_seed": 1234,
        "reuse_halflife": None,
        "min_fresh_ratio": 0.0,
        "min_fresh_wait_timeout_s": 3600.0,
    }


def _init_rollouter(monkeypatch, overrides):
    """Run the real __init__ with the dataset/dataloader construction stubbed out."""
    monkeypatch.setattr(rollouter_mod, "create_rl_dataset", lambda *a, **k: [0] * 8)
    monkeypatch.setattr(rollouter_mod, "create_rl_sampler", lambda *a, **k: None)
    monkeypatch.setattr(
        FullyAsyncRollouter,
        "_create_dataloader",
        lambda self, *a, **k: setattr(self, "train_dataloader", [0] * 8),
    )
    monkeypatch.setattr(FullyAsyncRollouter, "_init_dump_executor", lambda self: None)
    base = [
        "actor_rollout_ref.hybrid_engine=False",
        "data.train_batch_size=0",
        "actor_rollout_ref.actor.ppo_mini_batch_size=33",
        "actor_rollout_ref.rollout.n=16",
        "trainer.n_gpus_per_node=3",
        "trainer.nnodes=1",
        "trainer.total_epochs=1",
        "async_training.concurrent_samples_per_replica=33",
    ]
    cfg = _compose("fully_async_ppo_megatron_trainer", base + list(overrides))
    return FullyAsyncRollouter(config=cfg, tokenizer=None)


class TestInit:
    def test_defaults(self, monkeypatch):
        r = _init_rollouter(monkeypatch, [])
        assert r.replay_mode is False
        assert r.concurrency_ramp == []
        assert r.serialize_validation is False and r.pause_generation_during_save is False
        assert r.ramp_first_size == 33

    def test_replay_arm(self, monkeypatch):
        r = _init_rollouter(
            monkeypatch,
            [
                "async_training.replay_buffer.enable=True",
                "async_training.replay_buffer.requires_mini_batches=0.5",
                "async_training.concurrency_ramp=[5,12,20]",
                "async_training.serialize_validation=True",
                "async_training.pause_generation_during_save=True",
                "algorithm.norm_adv_by_std_in_grpo=True",
            ],
        )
        assert r.replay_mode is True
        assert r.concurrency_ramp == [5, 12, 20]
        assert r.ramp_first_size == 18  # 0.5 x 33 -> 17 -> 18 (18 x 16 splits over dp=3)
        assert r.serialize_validation and r.pause_generation_during_save

    def test_ramp_stage_above_the_per_replica_cap_is_rejected(self, monkeypatch):
        with pytest.raises(AssertionError, match="concurrency_ramp"):
            _init_rollouter(monkeypatch, ["async_training.concurrency_ramp=[5,40]"])

    @pytest.mark.parametrize("key", ["serialize_validation", "pause_generation_during_save"])
    def test_stop_the_world_requires_partial_rollout(self, monkeypatch, key):
        with pytest.raises(AssertionError, match="partial_rollout"):
            _init_rollouter(monkeypatch, [f"async_training.{key}=True", "async_training.partial_rollout=False"])
