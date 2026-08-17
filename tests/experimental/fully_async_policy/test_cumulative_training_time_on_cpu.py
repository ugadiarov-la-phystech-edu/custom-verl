# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Unit tests for the fully_async/timing/cumulative_training_time metric
(VCPO port) and the stop-the-world accounting:
- the trainer's virtual (no-validation-no-save) clock: rollout-bound,
  trainer-bound, stall/save exclusion, stamp handling
- timing state survives a checkpoint round-trip (timing_state.json), chaining
  across multiple restarts
- the rollouter's pause brackets (serialized validation, begin/end_save_pause)
  accumulate into the pause totals stamped onto samples

Run: pytest tests/experimental/fully_async_policy/test_cumulative_training_time_on_cpu.py
"""

import asyncio
import json
import time
from types import SimpleNamespace

from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter as _RollouterActor
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer as _TrainerActor


def _unwrap_ray_actor_class(actor_cls):
    """Both classes are @ray.remote ActorClass wrappers; tests need the plain class."""
    return actor_cls.__ray_metadata__.modified_class if hasattr(actor_cls, "__ray_metadata__") else actor_cls


FullyAsyncRollouter = _unwrap_ray_actor_class(_RollouterActor)
FullyAsyncTrainer = _unwrap_ray_actor_class(_TrainerActor)

TIMING_PREFIX = "fully_async/timing/"


def _make_trainer(first_sample_time=None, cumulative_validation_time=0.0, cumulative_save_time=0.0):
    """Minimal FullyAsyncTrainer with only the attributes under test."""
    t = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
    t.rollouter_first_sample_time = first_sample_time
    t.cumulative_validation_time = cumulative_validation_time
    t.cumulative_save_time = cumulative_save_time
    t.timing_wall_offset = 0.0
    t.timing_validation_offset = 0.0
    t.timing_save_offset = 0.0
    t.virtual_free_time = None
    t.virtual_training_time_offset = 0.0
    t._step_virtual_start = None
    t._step_actual_start = None
    t._step_valid_time = 0.0
    t._step_save_time = 0.0
    t.current_param_version = 0
    t.last_ckpt_version = 0
    t.max_steps_duration = 0
    return t


# ---------------------------------------------------------------- trainer math


def test_add_cumulative_time_metrics_math():
    trainer = _make_trainer(first_sample_time=100.0, cumulative_validation_time=5.0, cumulative_save_time=2.0)
    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=150.0)

    assert step_data[TIMING_PREFIX + "wall_time_since_first_sample"] == 50.0
    assert step_data[TIMING_PREFIX + "cumulative_validation_time"] == 5.0
    assert step_data[TIMING_PREFIX + "cumulative_save_time"] == 2.0
    # no batch has opened the virtual clock yet -> no exact metric
    assert TIMING_PREFIX + "cumulative_training_time" not in step_data

    # mid-step: started at virtual 120 / actual 130, 5s stalled on validation
    trainer._step_virtual_start = 120.0
    trainer._step_actual_start = 130.0
    trainer._step_valid_time = 5.0
    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=150.0)
    assert step_data[TIMING_PREFIX + "cumulative_training_time"] == 120.0 + (150.0 - 130.0) - 5.0 - 100.0


def test_add_cumulative_time_metrics_noop_without_anchor():
    trainer = _make_trainer(first_sample_time=None)
    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=150.0)
    assert step_data == {}


# ---------------------------------------------------------------- resume (timing_state.json)


def test_timing_state_checkpoint_roundtrip(tmp_path):
    saver = _make_trainer(first_sample_time=100.0, cumulative_validation_time=5.0, cumulative_save_time=2.0)
    # mid-step save: virtual 120 / actual 130, 5s stalled on validation
    saver._step_virtual_start = 120.0
    saver._step_actual_start = 130.0
    saver._step_valid_time = 5.0
    saver._save_timing_state(str(tmp_path), save_start=150.0)

    state = json.loads((tmp_path / "timing_state.json").read_text())
    assert state == {
        "wall_time_since_first_sample": 50.0,
        "cumulative_validation_time": 5.0,
        "cumulative_save_time": 2.0,
        "cumulative_training_time": 35.0,  # virtual: 120 + (150-130) - 5 - 100
    }

    resumed = _make_trainer()
    resumed._restore_timing_state(str(tmp_path))
    assert resumed.timing_wall_offset == 50.0
    assert resumed.timing_validation_offset == 5.0
    assert resumed.timing_save_offset == 2.0
    assert resumed.virtual_training_time_offset == 35.0

    # resumed segment: 30s wall, 4s validation, 1s saving; a step opened at
    # virtual 1010 / actual 1012 -> every metric continues from the totals.
    resumed.rollouter_first_sample_time = 1000.0
    resumed.cumulative_validation_time = 4.0
    resumed.cumulative_save_time = 1.0
    resumed._step_virtual_start = 1010.0
    resumed._step_actual_start = 1012.0
    step_data = {}
    resumed._add_cumulative_time_metrics(step_data, now=1030.0)
    assert step_data[TIMING_PREFIX + "wall_time_since_first_sample"] == 80.0
    assert step_data[TIMING_PREFIX + "cumulative_validation_time"] == 9.0
    assert step_data[TIMING_PREFIX + "cumulative_save_time"] == 3.0
    assert step_data[TIMING_PREFIX + "cumulative_training_time"] == (1010.0 + 18.0 - 1000.0) + 35.0


def test_timing_state_save_carries_offsets_forward_without_anchor(tmp_path):
    # A resumed run may checkpoint again before the rollouter reports its first
    # sample; the previous run's totals must pass through unchanged.
    trainer = _make_trainer(first_sample_time=None)
    trainer.timing_wall_offset = 50.0
    trainer.timing_validation_offset = 5.0
    trainer.timing_save_offset = 2.0
    trainer.virtual_training_time_offset = 35.0
    trainer._save_timing_state(str(tmp_path), save_start=999.0)

    state = json.loads((tmp_path / "timing_state.json").read_text())
    assert state["wall_time_since_first_sample"] == 50.0
    assert state["cumulative_validation_time"] == 5.0
    assert state["cumulative_save_time"] == 2.0
    assert state["cumulative_training_time"] == 35.0


def test_timing_state_second_resume_chains_totals(tmp_path):
    # run 1 -> checkpoint -> run 2 (with offsets) -> checkpoint -> run 3:
    # totals must chain across multiple restarts, not just one.
    run2 = _make_trainer(first_sample_time=200.0, cumulative_validation_time=3.0, cumulative_save_time=1.0)
    run2.timing_wall_offset = 50.0
    run2.timing_validation_offset = 5.0
    run2.timing_save_offset = 2.0
    run2.virtual_training_time_offset = 35.0
    run2.virtual_free_time = 230.0  # between steps: last step ended at virtual 230
    run2._save_timing_state(str(tmp_path), save_start=240.0)

    run3 = _make_trainer()
    run3._restore_timing_state(str(tmp_path))
    assert run3.timing_wall_offset == 90.0
    assert run3.timing_validation_offset == 8.0
    assert run3.timing_save_offset == 3.0
    assert run3.virtual_training_time_offset == (230.0 - 200.0) + 35.0


def test_restore_timing_state_missing_file_keeps_zero_offsets(tmp_path):
    trainer = _make_trainer()
    trainer._restore_timing_state(str(tmp_path))
    assert trainer.timing_wall_offset == 0.0
    assert trainer.timing_validation_offset == 0.0
    assert trainer.timing_save_offset == 0.0
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


# ------------------------------------------------- virtual (no-validation) clock


def _sample(enqueue_time, validation_pause_before=0.0, checkpoint_pause_before=0.0):
    return SimpleNamespace(
        enqueue_time=enqueue_time,
        validation_pause_before=validation_pause_before,
        checkpoint_pause_before=checkpoint_pause_before,
    )


def _run_step(trainer, consumer_end, samples, step_end, valid_time=0.0, save_time=0.0):
    """Drive one trainer step through the production virtual-clock hooks."""
    trainer._step_valid_time = valid_time
    trainer._step_save_time = save_time
    trainer._open_virtual_step(consumer_end, samples)
    trainer._advance_virtual_clock(now=step_end)
    return trainer.virtual_free_time


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
    assert step_data[TIMING_PREFIX + "cumulative_training_time"] == 26.0


def test_virtual_clock_trainer_bound_matches_no_validation_run():
    # Reference run: batches ready at t=6, 12 (R=6); trainer busy U=10 -> steps
    # end at 16 and 26; the trainer is the bottleneck throughout.
    # Run with validation: batch 2 was enqueued at t=12 before an 8s validation
    # (16..24); the trainer trains straight through it from backlog, so wall
    # time is unchanged — and so must the metric be (a naive wall - validation
    # subtraction would wrongly report 26 - 8 = 18 here).
    trainer = _make_trainer(first_sample_time=0.0, cumulative_validation_time=8.0)
    _run_step(trainer, consumer_end=6.0, samples=[_sample(6.0, 0.0)], step_end=16.0)
    end2 = _run_step(trainer, consumer_end=16.0, samples=[_sample(12.0, 0.0)], step_end=26.0)
    assert end2 == 26.0

    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=26.0)
    assert step_data[TIMING_PREFIX + "cumulative_training_time"] == 26.0


def test_virtual_clock_excludes_validation_stall():
    # A 3s awaited validation inside the step is validation-caused and must
    # not advance the virtual clock: 10s of measured step time -> 7s of busy.
    trainer = _make_trainer(first_sample_time=0.0)
    end = _run_step(trainer, consumer_end=10.0, samples=[_sample(10.0, 0.0)], step_end=20.0, valid_time=3.0)
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


# ------------------------------------------------- rollouter pause brackets


def _make_pause_rollouter(first_sample_time=100.0, serialize_validation=True, validate_duration=0.02):
    r = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
    r.lock = asyncio.Lock()
    r._resume_event = asyncio.Event()
    r._resume_event.set()
    r.paused = False
    r.active_tasks = set()
    r.first_sample_time = first_sample_time
    r.cumulative_validation_time = 0.0
    r.cumulative_checkpoint_pause = 0.0
    r._save_pause_start = None
    r.serialize_validation = serialize_validation
    r._validate = lambda: (time.sleep(validate_duration), {"val-core/acc": 1.0})[1]
    return r


def test_do_validate_serialized_pauses_and_accumulates():
    rollouter = _make_pause_rollouter()

    async def run():
        # an in-flight task the drain must wait for
        started = asyncio.Event()

        async def in_flight():
            started.set()
            await asyncio.sleep(0.03)

        task = asyncio.get_event_loop().create_task(in_flight())
        rollouter.active_tasks.add(task)
        await started.wait()
        result = await rollouter.do_validate()
        return result, task.done()

    result, in_flight_finished = asyncio.run(run())
    assert in_flight_finished, "serialized validation must drain in-flight generation first"
    assert result["val-core/acc"] == 1.0
    assert "rollouter/validate_time" in result
    assert rollouter.cumulative_validation_time >= 0.02
    assert rollouter.paused is False, "generation must resume after validation"
    assert rollouter._resume_event.is_set()


def test_do_validate_unserialized_does_not_pause_or_accumulate():
    rollouter = _make_pause_rollouter(serialize_validation=False)
    result = asyncio.run(rollouter.do_validate())
    assert result["val-core/acc"] == 1.0
    assert rollouter.cumulative_validation_time == 0.0
    assert rollouter.paused is False


def test_do_validate_pre_anchor_pause_not_accumulated():
    rollouter = _make_pause_rollouter(first_sample_time=None)
    asyncio.run(rollouter.do_validate())
    assert rollouter.cumulative_validation_time == 0.0
    assert rollouter.paused is False, "resume must happen even without the anchor"


def test_save_pause_brackets_accumulate_post_anchor():
    rollouter = _make_pause_rollouter()

    async def run():
        await rollouter.begin_save_pause()
        assert rollouter.paused is True
        await asyncio.sleep(0.02)
        await rollouter.end_save_pause()

    asyncio.run(run())
    assert rollouter.cumulative_checkpoint_pause >= 0.02
    assert rollouter.paused is False
    assert rollouter._resume_event.is_set()

    # pre-anchor: window runs but nothing accumulates
    rollouter = _make_pause_rollouter(first_sample_time=None)

    async def run_pre():
        await rollouter.begin_save_pause()
        await rollouter.end_save_pause()

    asyncio.run(run_pre())
    assert rollouter.cumulative_checkpoint_pause == 0.0
    assert rollouter.paused is False


def test_get_first_sample_time_getter():
    rollouter = _make_pause_rollouter(first_sample_time=123.5)
    assert rollouter.get_first_sample_time() == 123.5


# ------------------------------------------------- monitor-loop pause guard


def _async_const(value):
    async def fn(*args, **kwargs):
        return value

    return fn


def test_monitor_never_resumes_hard_pause():
    """Regression: the 10s monitor loop must not undo a stop-the-world save /
    validation pause even when the staleness quota would allow generation."""
    rollouter = _make_pause_rollouter()
    rollouter._should_pause_generation = _async_const(False)

    async def run():
        await rollouter.begin_save_pause()
        assert rollouter.paused is True and rollouter._hard_paused is True
        await rollouter._maybe_auto_resume()  # one monitor tick
        assert rollouter.paused is True, "monitor tick resumed a hard pause"
        assert not rollouter._resume_event.is_set()
        await rollouter.end_save_pause()

    asyncio.run(run())
    assert rollouter.paused is False and rollouter._hard_paused is False
    assert rollouter._resume_event.is_set()


def test_monitor_resumes_quota_pause():
    rollouter = _make_pause_rollouter()
    rollouter._should_pause_generation = _async_const(False)
    rollouter.paused = True
    rollouter._hard_paused = False
    rollouter._resume_event.clear()
    asyncio.run(rollouter._maybe_auto_resume())
    assert rollouter.paused is False
    assert rollouter._resume_event.is_set()


def test_monitor_keeps_quota_pause_while_over_quota():
    rollouter = _make_pause_rollouter()
    rollouter._should_pause_generation = _async_const(True)
    rollouter.paused = True
    rollouter._hard_paused = False
    rollouter._resume_event.clear()
    asyncio.run(rollouter._maybe_auto_resume())
    assert rollouter.paused is True


def test_do_validate_hard_pause_flag_lifecycle():
    rollouter = _make_pause_rollouter()
    asyncio.run(rollouter.do_validate())
    assert rollouter._hard_paused is False and rollouter.paused is False


# ------------------------------------------------- trainer save accumulation


def _make_save_trainer(tmp_path, save_freq=20, version=20, pause=False):
    t = _make_trainer(first_sample_time=0.0)
    t.config = OmegaConf.create({"trainer": {"save_freq": save_freq, "default_local_dir": str(tmp_path)}})
    t.current_param_version = version
    t.last_ckpt_version = 0
    t.pause_generation_during_save = pause
    t.save_queue_state = False
    t.calls = []

    def _fake_save_checkpoint():
        # the real _save_checkpoint creates the global_step folder
        (tmp_path / f"global_step_{t.current_param_version}").mkdir(parents=True, exist_ok=True)
        t.calls.append("save_checkpoint")

    t._save_checkpoint = _fake_save_checkpoint
    t._save_timing_state = lambda folder, start: t.calls.append("timing_state")
    t._replay_checkpoint_state = lambda: {"stub": True}
    t.replay_buffer = SimpleNamespace(size=lambda: 0)

    class _Remote:
        def __init__(self, name):
            self.name = name

        def remote(self):
            async def run():
                t.calls.append(self.name)

            return run()

    t.rollouter = SimpleNamespace(begin_save_pause=_Remote("begin_pause"), end_save_pause=_Remote("end_pause"))
    return t


def test_replay_maybe_save_accumulates_save_time(tmp_path):
    trainer = _make_save_trainer(tmp_path)
    timing_raw = {}
    asyncio.run(trainer._replay_maybe_save(timing_raw))
    assert trainer.calls == ["save_checkpoint", "timing_state"]
    assert trainer.cumulative_save_time > 0
    assert trainer._step_save_time == trainer.cumulative_save_time
    assert trainer.last_ckpt_version == 20
    assert "save_checkpoint" in timing_raw
    assert (tmp_path / "global_step_20" / "replay_buffer.pt").exists()
    # second call at the same version is a no-op (already checkpointed)
    saved = trainer.cumulative_save_time
    asyncio.run(trainer._replay_maybe_save({}))
    assert trainer.cumulative_save_time == saved


def test_replay_maybe_save_disabled_leaves_time_zero(tmp_path):
    trainer = _make_save_trainer(tmp_path, save_freq=-1)
    asyncio.run(trainer._replay_maybe_save({}))
    assert trainer.calls == []
    assert trainer.cumulative_save_time == 0.0
    # force=True still respects save_freq=-1... but only off-cadence versions:
    trainer2 = _make_save_trainer(tmp_path, save_freq=20, version=13)
    asyncio.run(trainer2._replay_maybe_save({}))
    assert trainer2.calls == []  # off-cadence, no force
    asyncio.run(trainer2._replay_maybe_save({}, force=True))
    assert "save_checkpoint" in trainer2.calls  # force overrides the cadence


def test_replay_maybe_save_pause_brackets_the_save(tmp_path):
    trainer = _make_save_trainer(tmp_path, pause=True)
    asyncio.run(trainer._replay_maybe_save({}))
    assert trainer.calls == ["begin_pause", "save_checkpoint", "timing_state", "end_pause"]
