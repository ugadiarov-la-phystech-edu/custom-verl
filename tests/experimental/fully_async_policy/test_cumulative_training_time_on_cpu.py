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
"""fully_async/timing/cumulative_training_time: the virtual (no-validation, no-checkpoint) timeline.

The trainer replays the pipeline schedule with validation- and save-caused delays deleted: each step
starts at max(trainer free, batch virtual-ready time) -- a sample is virtually ready at its enqueue time
minus the rollouter's validation and save pauses before it -- and advances by its busy time minus the
validation wait and the checkpoint save. The rollouter anchors the clock at its first training draw and
stamps every enqueued sample. Every checkpoint records the totals in global_step_N/timing_state.json.
"""

import asyncio
import contextlib
import json
import os
import time
from datetime import datetime
from types import SimpleNamespace

import pytest
import ray.cloudpickle
from omegaconf import OmegaConf

from tests.experimental.fully_async_policy.test_rollouter_replay_on_cpu import _bare_rollouter, _group
from tests.experimental.fully_async_policy.test_trainer_replay_on_cpu import _bare_trainer, _Queue, _sample
from verl.experimental.fully_async_policy import fully_async_rollouter as rollouter_mod
from verl.experimental.fully_async_policy.detach_utils import RolloutSample


def _stamped(enqueue, validation_before=0.0, checkpoint_before=0.0):
    return SimpleNamespace(
        enqueue_time=enqueue, validation_pause_before=validation_before, checkpoint_pause_before=checkpoint_before
    )


def _step(t, open_at, samples, close_at, wait_valid=0.0, save=0.0):
    t._step_wait_valid_time = wait_valid
    t._step_save_time = save
    t._open_virtual_step(open_at, samples)
    t._advance_virtual_clock(now=close_at)


def _training_time(t, now, first=0.0, validation=0.0, pause=0.0):
    data = {}
    timing = {
        "first_sample_time": first,
        "cumulative_validation_time": validation,
        "cumulative_checkpoint_pause": pause,
    }
    t._add_cumulative_time_metrics(data, timing, now=now)
    return data


class TestVirtualClock:
    def test_no_validation_no_save_equals_wall_time(self):
        t = _bare_trainer()
        _step(t, 10, [_stamped(10)], close_at=15)  # waited for data until 10, busy 5
        _step(t, 30, [_stamped(30)], close_at=36)  # rollout-bound: idle until 30
        _step(t, 36, [_stamped(33)], close_at=40)  # trainer-bound: data was already there
        assert t.virtual_free_time == 40
        assert _training_time(t, now=40)["fully_async/timing/cumulative_training_time"] == 40

    def test_validation_is_deleted(self):
        t = _bare_trainer()
        # step 1 validates for 20 s at its end; generation was held for those 20 s as well
        _step(t, 10, [_stamped(10)], close_at=35, wait_valid=20)
        assert t.virtual_free_time == 15
        # the next batch arrived at 40, 20 s late because of the pause: virtually ready at 20
        _step(t, 40, [_stamped(40, validation_before=20)], close_at=45)
        assert t.virtual_free_time == 25
        m = _training_time(t, now=45, validation=20)
        assert m["fully_async/timing/cumulative_training_time"] == 25  # = 45 wall - 20 validation
        assert m["fully_async/timing/wall_time_since_first_sample"] == 45

    def test_checkpoint_save_is_deleted(self):
        t = _bare_trainer()
        _step(t, 10, [_stamped(10)], close_at=45, save=30)
        assert t.virtual_free_time == 15
        _step(t, 50, [_stamped(50, checkpoint_before=30)], close_at=52)
        assert t.virtual_free_time == 22

    def test_trainer_bound_step_starts_when_the_trainer_is_free(self):
        t = _bare_trainer()
        t.virtual_free_time = 20.0
        t._open_virtual_step(25, [_stamped(5)])
        assert t._step_virtual_start == 20.0

    def test_rollout_bound_step_starts_when_the_last_sample_is_ready(self):
        t = _bare_trainer()
        t.virtual_free_time = 20.0
        t._open_virtual_step(40, [_stamped(22), _stamped(38), _stamped(30)])
        assert t._step_virtual_start == 38

    def test_pure_replay_step_starts_when_the_trainer_is_free(self):
        # a mini-batch without fresh groups waits for no arrival, even if the wall clock moved on
        t = _bare_trainer()
        t.virtual_free_time = 20.0
        t._open_virtual_step(100, [])
        assert t._step_virtual_start == 20.0

    def test_first_step_without_stamps_starts_at_the_actual_time(self):
        t = _bare_trainer()
        t._open_virtual_step(12, [SimpleNamespace()])
        assert t._step_virtual_start == 12

    def test_mid_step_position(self):
        t = _bare_trainer()
        t._step_wait_valid_time = 3.0
        t._open_virtual_step(10, [_stamped(8)])
        assert t._virtual_now(20) == pytest.approx(8 + 10 - 3)
        assert t._virtual_now(20) == _training_time(t, now=20)["fully_async/timing/cumulative_training_time"]

    def test_advance_without_an_open_step_is_a_no_op(self):
        t = _bare_trainer()
        t.virtual_free_time = 5.0
        t._advance_virtual_clock(now=100)
        assert t.virtual_free_time == 5.0


class TestMetrics:
    def test_nothing_before_the_first_training_sample(self):
        t = _bare_trainer()
        data = {}
        t._add_cumulative_time_metrics(
            data, {"first_sample_time": None, "cumulative_validation_time": 0.0, "cumulative_checkpoint_pause": 0.0}
        )
        assert data == {}

    def test_keys(self):
        t = _bare_trainer()
        t.cumulative_save_time = 7.0
        _step(t, 110, [_stamped(110)], close_at=150)
        m = _training_time(t, now=160, first=100, validation=4, pause=6)
        assert m == {
            "fully_async/timing/wall_time_since_first_sample": 60,
            "fully_async/timing/cumulative_validation_time": 4,
            "fully_async/timing/cumulative_checkpoint_pause": 6,
            "fully_async/timing/cumulative_save_time": 7.0,
            "fully_async/timing/cumulative_training_time": 50,
        }

    def test_logged_with_every_sync(self, monkeypatch):
        t = _bare_trainer()
        t.metrics_aggregator = SimpleNamespace(get_aggregated_metrics=lambda **k: {"x": 1.0}, reset=lambda: None)
        logged = []
        t.logger = SimpleNamespace(log=lambda data, step: logged.append(data))
        t.rollouter = SimpleNamespace(
            get_timing_state=SimpleNamespace(
                remote=lambda: {
                    "first_sample_time": time.time() - 10,
                    "cumulative_validation_time": 1.0,
                    "cumulative_checkpoint_pause": 0.0,
                }
            )
        )
        monkeypatch.setattr("ray.get", lambda x: x)
        t._fit_log_aggregated_training_metrics({})
        assert logged[0]["x"] == 1.0
        assert logged[0]["fully_async/timing/wall_time_since_first_sample"] == pytest.approx(10, abs=1)


class TestTrainerWiring:
    def test_replay_acquire_is_gated_by_the_fresh_prefix_only(self):
        t = _bare_trainer(mini=2)
        t.virtual_free_time = 0.0
        old = ray.cloudpickle.loads(_sample(0))
        old.enqueue_time = 500.0  # a replayed group: its stamp must not gate the step
        t.replay_buffer.add(old, 0)
        t.replay_buffer.compose_minibatch(1, 0)  # consume its freshness
        fresh = ray.cloudpickle.loads(_sample(0))
        fresh.enqueue_time = 50.0
        t.message_queue_client = _Queue(available=[ray.cloudpickle.dumps(fresh)])
        entries, info = asyncio.run(t._acquire_replay_minibatch())
        assert info["n_new"] == 1 and len(entries) == 2
        assert t._step_virtual_start == 50.0

    def test_fifo_collection_opens_the_step(self, monkeypatch):
        from verl.experimental.fully_async_policy import fully_async_trainer as trainer_mod

        t = _bare_trainer(mini=2)
        t.dynamic_schedule_enabled = False
        t._step_wait_times, t._step_wait_samples = [], []
        samples = []
        for enqueue in (30.0, 35.0):
            s = ray.cloudpickle.loads(_sample(0))
            s.enqueue_time = enqueue
            samples.append(ray.cloudpickle.dumps(s))
        t.message_queue_client = _Queue(arrivals=samples)
        monkeypatch.setattr(
            trainer_mod, "assemble_batch_from_rollout_samples", lambda *a, **k: SimpleNamespace(meta_info={})
        )
        asyncio.run(t._get_samples_from_queue())
        assert t._step_virtual_start == 35.0

    def test_validation_wait_is_excluded(self):
        t = _bare_trainer()
        t.local_trigger_step = 1
        t.current_param_version = 2
        t.config = OmegaConf.create({"trainer": {"test_freq": 2}, "async_training": {"use_trainer_do_validate": False}})
        t.logger = SimpleNamespace(log=lambda **k: None)

        async def validate():
            await asyncio.sleep(0.2)
            return {}

        t.rollouter = SimpleNamespace(do_validate=SimpleNamespace(remote=validate))
        asyncio.run(t._fit_validate())
        assert t._step_wait_valid_time == pytest.approx(0.2, abs=0.1)

    def test_save_time_is_excluded(self):
        t = _bare_trainer()
        t.current_param_version = 3
        t.last_ckpt_version = 0
        t.max_steps_duration = 0
        t.timing_raw = {}
        t.config = OmegaConf.create({"trainer": {"save_freq": 3, "esi_redundant_time": 0}})
        t._save_checkpoint = lambda: time.sleep(0.1)
        t._fit_save_checkpoint()
        assert t._step_save_time == pytest.approx(t.timing_raw["save_checkpoint"])
        assert t.cumulative_save_time == pytest.approx(t.timing_raw["save_checkpoint"])
        assert t._step_save_time >= 0.1

    @pytest.mark.parametrize("method", ["_fit_replay_step", "fit_step"])
    def test_steps_close_the_clock_after_the_save(self, method):
        import inspect

        from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

        src = inspect.getsource(getattr(FullyAsyncTrainer.__ray_metadata__.modified_class, method))
        save, advance = src.index("self._fit_save_checkpoint()"), src.index("self._advance_virtual_clock()")
        assert save < advance
        assert "self._step_wait_valid_time = 0.0" in src and "self._step_save_time = 0.0" in src


class TestRollouterSide:
    def test_first_training_draw_anchors_the_clock(self, monkeypatch):
        r = _bare_rollouter()
        r.global_steps = 1
        r.total_rollout_steps = 3
        r.config = None

        def draws():
            for i in range(3):
                yield 0, {"i": i}

        r._create_continuous_iterator = draws
        monkeypatch.setattr(rollouter_mod, "prepare_single_generation_data", lambda batch, config: batch)
        before = time.time()
        asyncio.run(r._feed_samples())
        first = r.first_sample_time
        assert before <= first <= time.time()
        r.global_steps = 1
        asyncio.run(r._feed_samples())
        assert r.first_sample_time == first  # set once

    @pytest.mark.parametrize("anchored", [True, False])
    def test_validation_time_counts_after_the_anchor_only(self, anchored):
        r = _bare_rollouter()
        r.first_sample_time = 1.0 if anchored else None
        r._validate = lambda: time.sleep(0.05) or {}
        asyncio.run(r.do_validate())
        assert (r.cumulative_validation_time >= 0.05) is anchored

    @pytest.mark.parametrize("serialize", [True, False])
    def test_validation_accounts_the_frozen_window_when_serialized(self, serialize):
        # serialized: generation is frozen from the freeze to the unfreeze, and the trainer excludes all of it,
        # so the shift of later samples must cover the freeze too (as end_save_pause does for saves)
        r = _bare_rollouter(serialize=serialize)
        r.first_sample_time = 1.0
        freeze = r._freeze_inflight_generation

        async def slow_freeze():
            await freeze()
            await asyncio.sleep(0.2)

        r._freeze_inflight_generation = slow_freeze
        r._validate = lambda: time.sleep(0.05) or {}
        out = asyncio.run(r.do_validate())
        assert out["rollouter/validate_time"] < 0.2  # the validation alone
        assert (r.cumulative_validation_time >= 0.25) is serialize
        assert r.cumulative_validation_time >= 0.05

    def test_failed_validation_is_not_accounted(self):
        r = _bare_rollouter(serialize=True)
        r.first_sample_time = 1.0

        def boom():
            raise RuntimeError("validation failed")

        r._validate = boom
        with pytest.raises(RuntimeError):
            asyncio.run(r.do_validate())
        assert r.cumulative_validation_time == 0.0
        assert not r.is_hard_paused()

    @pytest.mark.parametrize("anchored", [True, False])
    def test_save_pause_counts_after_the_anchor_only(self, anchored):
        r = _bare_rollouter()
        r.first_sample_time = 1.0 if anchored else None

        async def scenario():
            await r.begin_save_pause()
            await asyncio.sleep(0.05)
            await r.end_save_pause()

        asyncio.run(scenario())
        assert (r.cumulative_checkpoint_pause >= 0.05) is anchored

    def test_enqueued_samples_are_stamped(self):
        r = _bare_rollouter(replay=False)
        r.cumulative_validation_time, r.cumulative_checkpoint_pause = 12.0, 3.0

        class _AgentLoop:
            async def generate_sequences_single(self, batch):
                return _group([1.0, 0.0])

        r.async_rollout_manager = _AgentLoop()
        before = time.time()
        rs = RolloutSample(full_batch=_group([1.0, 0.0]), sample_id="s", epoch=0, rollout_status={})
        asyncio.run(r._process_single_sample_streaming(rs))
        sent = ray.cloudpickle.loads(r.message_queue_client.put[0])
        assert before <= sent.enqueue_time <= time.time()
        assert (sent.validation_pause_before, sent.checkpoint_pause_before) == (12.0, 3.0)

    def test_timing_state(self):
        r = _bare_rollouter()
        r.first_sample_time, r.cumulative_validation_time, r.cumulative_checkpoint_pause = 5.0, 2.0, 1.0
        assert r.get_timing_state() == {
            "first_sample_time": 5.0,
            "cumulative_validation_time": 2.0,
            "cumulative_checkpoint_pause": 1.0,
        }

    def test_unstamped_sample_defaults(self):
        rs = RolloutSample(full_batch=None, sample_id="x", epoch=0, rollout_status={})
        assert (rs.enqueue_time, rs.validation_pause_before, rs.checkpoint_pause_before) == (None, 0.0, 0.0)


# --------------------------------------------------------------------------- timing_state.json in checkpoints

FMT = "%Y-%m-%d %H:%M:%S"
TIMING_KEYS = {
    "wall_time_since_first_sample",
    "cumulative_validation_time",
    "cumulative_save_time",
    "cumulative_training_time",
    "first_sample_datetime",
    "checkpoint_saved_datetime",
}


def _rollouter_timing(first=100.0, validation=5.0, pause=0.0):
    return {"first_sample_time": first, "cumulative_validation_time": validation, "cumulative_checkpoint_pause": pause}


def _saved(t, folder, save_start, timing):
    t._save_timing_state(str(folder), save_start, timing)
    return json.loads((folder / "timing_state.json").read_text())


class TestTimingStateFile:
    def test_mid_step_save(self, tmp_path):
        # the values of custom_vcpo's test_timing_state_checkpoint_roundtrip
        t = _bare_trainer()
        t.cumulative_save_time = 2.0
        t._step_virtual_start, t._step_actual_start, t._step_wait_valid_time = 120.0, 130.0, 5.0
        assert _saved(t, tmp_path, 150.0, _rollouter_timing()) == {
            "wall_time_since_first_sample": 50.0,
            "cumulative_validation_time": 5.0,
            "cumulative_save_time": 2.0,
            "cumulative_training_time": 35.0,  # 120 + (150 - 130) - 5 - 100
            "first_sample_datetime": datetime.fromtimestamp(100.0).strftime(FMT),
            "checkpoint_saved_datetime": datetime.fromtimestamp(150.0).strftime(FMT),
        }

    def test_save_between_steps_uses_where_the_last_step_ended(self, tmp_path):
        t = _bare_trainer()
        _step(t, 110, [_stamped(110)], close_at=150)
        state = _saved(t, tmp_path, 400.0, _rollouter_timing())
        assert state["cumulative_training_time"] == 50.0  # virtual_free_time 150 - first 100
        assert state["wall_time_since_first_sample"] == 300.0

    def test_equals_the_logged_metric_at_the_same_moment(self, tmp_path):
        t = _bare_trainer()
        t.cumulative_save_time = 7.0
        _step(t, 110, [_stamped(110)], close_at=150, wait_valid=4)
        t._open_virtual_step(160, [_stamped(158)])
        timing = _rollouter_timing(first=100.0, validation=4.0, pause=6.0)
        logged = {}
        t._add_cumulative_time_metrics(logged, timing, now=170.0)
        state = _saved(t, tmp_path, 170.0, timing)
        for key in ("wall_time_since_first_sample", "cumulative_validation_time", "cumulative_save_time"):
            assert state[key] == logged[f"fully_async/timing/{key}"], key
        assert state["cumulative_training_time"] == logged["fully_async/timing/cumulative_training_time"]

    def test_before_the_first_training_sample_everything_is_zero(self, tmp_path):
        t = _bare_trainer()
        t.cumulative_save_time = 3.0  # no anchor yet: not counted, as in the metrics
        state = _saved(t, tmp_path, 50.0, _rollouter_timing(first=None, validation=0.0))
        assert state == {
            "wall_time_since_first_sample": 0.0,
            "cumulative_validation_time": 0.0,
            "cumulative_save_time": 0.0,
            "cumulative_training_time": 0.0,
            "first_sample_datetime": None,
            "checkpoint_saved_datetime": datetime.fromtimestamp(50.0).strftime(FMT),
        }

    def test_anchored_but_no_step_opened_yet(self, tmp_path):
        state = _saved(_bare_trainer(), tmp_path, 130.0, _rollouter_timing())
        assert state["cumulative_training_time"] == 0.0 and state["wall_time_since_first_sample"] == 30.0

    def test_creates_the_folder_and_overwrites(self, tmp_path):
        t = _bare_trainer()
        folder = tmp_path / "global_step_3"
        _saved(t, folder, 120.0, _rollouter_timing())
        assert _saved(t, folder, 140.0, _rollouter_timing())["wall_time_since_first_sample"] == 40.0
        assert [p.name for p in folder.iterdir()] == ["timing_state.json"]


def _checkpointing_trainer(tmp_path, monkeypatch, timing, actor_save_s=30.0, step=12):
    """A bare trainer whose _save_checkpoint_inner runs for real on a fake clock: the actor save takes
    actor_save_s seconds, the rollouter returns `timing`."""
    from verl.experimental.fully_async_policy import fully_async_trainer as trainer_mod

    clock = {"now": 1000.0}
    monkeypatch.setattr(trainer_mod.time, "time", lambda: clock["now"])
    monkeypatch.setattr(trainer_mod.ray, "get", lambda x: x)

    @contextlib.contextmanager
    def fake_timer(name, timing_raw, **kwargs):  # marked_timer on the fake clock
        start = clock["now"]
        yield
        timing_raw[name] = timing_raw.get(name, 0.0) + clock["now"] - start

    monkeypatch.setattr(trainer_mod, "marked_timer", fake_timer)
    calls = []

    def actor_save(local_path, remote_path, version, max_ckpt_to_keep=None):
        calls.append(("actor", version))
        clock["now"] += actor_save_s

    def rollouter_save(folder):
        calls.append(("rollouter", folder))
        os.makedirs(folder, exist_ok=True)

    def get_timing_state():
        calls.append(("timing", clock["now"]))
        return dict(timing)

    t = _bare_trainer()
    t.current_param_version = step
    t.use_critic = False
    t.actor_rollout_wg = SimpleNamespace(save_checkpoint=actor_save)
    t.rollouter = SimpleNamespace(
        save_checkpoint=SimpleNamespace(remote=rollouter_save),
        get_timing_state=SimpleNamespace(remote=get_timing_state),
    )
    t.config = OmegaConf.create(
        {
            "trainer": {"default_local_dir": str(tmp_path), "default_hdfs_dir": None, "max_actor_ckpt_to_keep": None},
            "async_training": {"pause_generation_during_save": False},
        }
    )
    return t, clock, calls


class TestTimingStateInCheckpoints:
    def test_written_next_to_the_checkpoint(self, tmp_path, monkeypatch):
        t, _, calls = _checkpointing_trainer(tmp_path, monkeypatch, _rollouter_timing(first=900.0))
        t._save_checkpoint_inner()
        state = json.loads((tmp_path / "global_step_12" / "timing_state.json").read_text())
        assert set(state) == TIMING_KEYS
        assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "12"
        assert [c[0] for c in calls] == ["timing", "actor", "rollouter"]

    def test_snapshot_excludes_the_saves_own_duration(self, tmp_path, monkeypatch):
        t, clock, _ = _checkpointing_trainer(tmp_path, monkeypatch, _rollouter_timing(first=900.0), actor_save_s=30.0)
        t._step_virtual_start, t._step_actual_start = 950.0, 960.0
        t.cumulative_save_time = 4.0
        t._save_checkpoint_inner()
        assert clock["now"] == 1030.0  # the save took 30 s
        state = json.loads((tmp_path / "global_step_12" / "timing_state.json").read_text())
        assert state["wall_time_since_first_sample"] == 100.0  # 1000 - 900, at the start of the save
        assert state["cumulative_training_time"] == 90.0  # 950 + (1000 - 960) - 900
        assert state["cumulative_save_time"] == 4.0  # earlier saves only
        assert state["checkpoint_saved_datetime"] == datetime.fromtimestamp(1000.0).strftime(FMT)

    def test_each_checkpoint_counts_the_earlier_saves(self, tmp_path, monkeypatch):
        t, clock, _ = _checkpointing_trainer(tmp_path, monkeypatch, _rollouter_timing(first=900.0), actor_save_s=30.0)
        t.last_ckpt_version, t.max_steps_duration, t.timing_raw = 0, 0, {}
        t.config.trainer.save_freq, t.config.trainer.esi_redundant_time = 12, 0
        # step 12: data at 900, busy 100 s, then a 30 s save; the clock closes after the save (as in fit_step)
        t._open_virtual_step(900.0, [_stamped(900.0)])
        t._fit_save_checkpoint()
        t._advance_virtual_clock()
        assert t.virtual_free_time == 1000.0
        # step 24: its batch arrived right after the stop-the-world save, i.e. 30 s late; busy 50 s
        t.current_param_version = 24
        t._step_save_time = 0.0
        t._open_virtual_step(clock["now"], [_stamped(clock["now"], checkpoint_before=30.0)])
        clock["now"] += 50.0
        t._fit_save_checkpoint()
        first = json.loads((tmp_path / "global_step_12" / "timing_state.json").read_text())
        second = json.loads((tmp_path / "global_step_24" / "timing_state.json").read_text())
        assert first["cumulative_save_time"] == 0.0 and second["cumulative_save_time"] == 30.0
        assert first["cumulative_training_time"] == 100.0
        # the first save's 30 s is not training time: the second step starts where the first one ended
        assert second["cumulative_training_time"] == 150.0
        assert second["wall_time_since_first_sample"] == 180.0

    def test_stop_the_world_save_writes_it_too(self, tmp_path, monkeypatch):
        t, _, calls = _checkpointing_trainer(tmp_path, monkeypatch, _rollouter_timing(first=900.0))
        t.config.async_training.pause_generation_during_save = True
        t.rollouter.begin_save_pause = SimpleNamespace(remote=lambda: calls.append(("begin",)))
        t.rollouter.end_save_pause = SimpleNamespace(remote=lambda: calls.append(("end",)))
        t._save_checkpoint()
        assert [c[0] for c in calls] == ["begin", "timing", "actor", "rollouter", "end"]
        assert (tmp_path / "global_step_12" / "timing_state.json").exists()
