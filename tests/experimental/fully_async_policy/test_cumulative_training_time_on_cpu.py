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


def _async_const(value):
    async def fn(*args, **kwargs):
        return value

    return fn


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
    assert {k: v for k, v in state.items() if not isinstance(v, str)} == {
        "wall_time_since_first_sample": 50.0,
        "cumulative_validation_time": 5.0,
        "cumulative_save_time": 2.0,
        "cumulative_training_time": 35.0,  # virtual: 120 + (150-130) - 5 - 100
    }
    # plus the two absolute ISO timestamps (asserted in detail further down)
    assert set(state) == {
        "wall_time_since_first_sample",
        "cumulative_validation_time",
        "cumulative_save_time",
        "cumulative_training_time",
        "first_sample_time",
        "checkpoint_save_started",
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


def _make_pause_rollouter(first_sample_time=100.0, serialize_validation=True, validate_duration=0.02, over_quota=False):
    r = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
    r.lock = asyncio.Lock()
    r._resume_event = asyncio.Event()
    r._resume_event.set()
    r.paused = False
    r._hard_paused = False
    r.active_tasks = set()
    r.first_sample_time = first_sample_time
    r.cumulative_validation_time = 0.0
    r.cumulative_checkpoint_pause = 0.0
    r._save_pause_start = None
    r._validation_pause_start = None
    r.serialize_validation = serialize_validation
    r._validate = lambda: (time.sleep(validate_duration), {"val-core/acc": 1.0})[1]
    # _resume_generation re-checks the staleness quota before lifting a hard pause.
    r._should_pause_generation = _async_const(over_quota)
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


# ------------------------------------------------------- trainer save bracket


def _make_save_trainer(tmp_path, save_freq=20, version=20, pause=False, fail=False):
    """Trainer wired for _fit_save_checkpoint (the production save path)."""
    t = _make_trainer(first_sample_time=0.0)
    t.config = OmegaConf.create(
        {
            "trainer": {
                "save_freq": save_freq,
                "default_local_dir": str(tmp_path),
                "esi_redundant_time": 0,
            }
        }
    )
    t.current_param_version = version
    t.last_ckpt_version = 0
    t.pause_generation_during_save = pause
    t.timing_raw = {}
    t.calls = []

    def _fake_save_checkpoint():
        t.calls.append("save_checkpoint")
        if fail:
            raise RuntimeError("save exploded")
        folder = tmp_path / f"global_step_{t.current_param_version}"
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder)

    t._save_checkpoint = _fake_save_checkpoint
    t._save_timing_state = lambda folder, start: t.calls.append("timing_state")

    class _Remote:
        def __init__(self, name):
            self.name = name

        def remote(self):
            async def run():
                t.calls.append(self.name)

            return run()

    t.rollouter = SimpleNamespace(begin_save_pause=_Remote("begin_pause"), end_save_pause=_Remote("end_pause"))
    return t


def test_fit_save_checkpoint_accumulates_save_time(tmp_path):
    trainer = _make_save_trainer(tmp_path)
    asyncio.run(trainer._fit_save_checkpoint())
    assert trainer.calls == ["save_checkpoint", "timing_state"]
    assert trainer.cumulative_save_time > 0
    assert trainer._step_save_time == trainer.cumulative_save_time
    assert trainer.last_ckpt_version == 20
    assert "save_checkpoint" in trainer.timing_raw
    # a second call at the same version is a no-op (already checkpointed)
    saved = trainer.cumulative_save_time
    asyncio.run(trainer._fit_save_checkpoint())
    assert trainer.cumulative_save_time == saved


def test_fit_save_checkpoint_disabled_leaves_time_zero(tmp_path):
    trainer = _make_save_trainer(tmp_path, save_freq=-1)
    asyncio.run(trainer._fit_save_checkpoint())
    assert trainer.calls == []
    assert trainer.cumulative_save_time == 0.0
    # off-cadence version is skipped, force=True overrides the cadence
    trainer2 = _make_save_trainer(tmp_path, save_freq=20, version=13)
    asyncio.run(trainer2._fit_save_checkpoint())
    assert trainer2.calls == []
    asyncio.run(trainer2._fit_save_checkpoint(force=True))
    assert "save_checkpoint" in trainer2.calls


def test_fit_save_checkpoint_pause_brackets_the_save(tmp_path):
    trainer = _make_save_trainer(tmp_path, pause=True)
    asyncio.run(trainer._fit_save_checkpoint())
    assert trainer.calls == ["begin_pause", "save_checkpoint", "timing_state", "end_pause"]


def test_fit_save_checkpoint_resumes_and_accounts_on_failure(tmp_path):
    """A failing save must still release the stop-the-world pause (nothing else
    can: _maybe_auto_resume refuses hard pauses) and must still be accounted."""
    trainer = _make_save_trainer(tmp_path, pause=True, fail=True)
    try:
        asyncio.run(trainer._fit_save_checkpoint())
    except RuntimeError as e:
        assert "save exploded" in str(e)
    else:
        raise AssertionError("the save exception must propagate")
    assert trainer.calls == ["begin_pause", "save_checkpoint", "end_pause"]
    assert trainer.cumulative_save_time > 0, "the pause window elapsed; it must be accounted"
    assert trainer._step_save_time == trainer.cumulative_save_time


# ------------------------------------------------- pause / quota interaction


def test_save_pause_over_quota_pause_keeps_generation_paused():
    """A stop-the-world bracket opened while the staleness quota already forbids
    generation must not resume generation when it closes."""
    rollouter = _make_pause_rollouter(over_quota=True)

    async def run():
        rollouter.paused = True  # pre-existing soft (quota) pause
        rollouter._resume_event.clear()
        await rollouter.begin_save_pause()
        await rollouter.end_save_pause()

    asyncio.run(run())
    assert rollouter._hard_paused is False, "the hard pause must be released"
    assert rollouter.paused is True, "the staleness-quota pause must survive the bracket"
    assert not rollouter._resume_event.is_set()
    assert rollouter.cumulative_checkpoint_pause >= 0.0


def test_save_pause_without_quota_pressure_resumes():
    rollouter = _make_pause_rollouter(over_quota=False)

    async def run():
        await rollouter.begin_save_pause()
        await rollouter.end_save_pause()

    asyncio.run(run())
    assert rollouter.paused is False
    assert rollouter._hard_paused is False
    assert rollouter._resume_event.is_set()


def test_reset_staleness_does_not_break_a_hard_pause():
    """reset_staleness is a second resume path; it must not undo a hard pause."""
    rollouter = _make_pause_rollouter()
    rollouter.active_tasks = set()
    rollouter.staleness_samples = 0
    rollouter.step_start_time = time.time()
    rollouter.idle_start_time = rollouter.step_start_time
    rollouter.message_queue_client = SimpleNamespace(get_queue_size=_async_const(0))

    async def run():
        await rollouter._pause_generation_and_drain()
        await FullyAsyncRollouter.reset_staleness(rollouter)

    asyncio.run(run())
    assert rollouter.paused is True, "reset_staleness resumed a stop-the-world pause"
    assert not rollouter._resume_event.is_set()


def test_stamps_include_a_pause_still_in_progress():
    """Drain-based pauses complete in-flight generations, so samples are enqueued
    during the window; their stamps must include the elapsed part of it."""
    rollouter = _make_pause_rollouter()
    before = rollouter._live_validation_pause()
    assert before == 0.0
    rollouter._validation_pause_start = time.time() - 5.0
    live = rollouter._live_validation_pause()
    assert live >= 5.0, "an in-progress validation pause must be visible to the stamps"
    rollouter._save_pause_start = time.time() - 3.0
    assert rollouter._live_checkpoint_pause() >= 3.0
    # pre-anchor the accumulators stay untouched
    rollouter.first_sample_time = None
    assert rollouter._live_validation_pause() == 0.0
    assert rollouter._live_checkpoint_pause() == 0.0


# ------------------------------------------------- trainer validation accounting


def _make_validate_trainer(test_freq=1, version=1, validate_duration=0.02, use_trainer_do_validate=False):
    t = _make_trainer(first_sample_time=0.0)
    t.config = OmegaConf.create(
        {
            "trainer": {"test_freq": test_freq},
            "async_training": {"use_trainer_do_validate": use_trainer_do_validate},
        }
    )
    t.local_trigger_step = 1
    t.current_param_version = version
    t.logged = []
    t.logger = SimpleNamespace(log=lambda data, step: t.logged.append((data, step)))

    class _DoValidate:
        def remote(self):
            async def run():
                await asyncio.sleep(validate_duration)
                return {"val-core/acc": 1.0}

            return run()

    t.rollouter = SimpleNamespace(do_validate=_DoValidate())
    return t


def test_fit_validate_populates_the_step_stall():
    """_step_valid_time must be filled by production code, not only by tests."""
    trainer = _make_validate_trainer()
    asyncio.run(trainer._fit_validate())
    assert trainer._step_valid_time >= 0.02
    assert trainer.cumulative_validation_time == trainer._step_valid_time
    assert trainer.logged and trainer.logged[0][1] == 1


def test_fit_validate_cadence_table():
    """test_freq gates validation; -1 and 0 disable it without dividing by zero."""
    for test_freq, version, expected in [
        (-1, 5, False),
        (0, 5, False),
        (5, 0, False),  # version 0 never validates
        (5, 4, False),
        (5, 5, True),
        (5, 10, True),
        (1, 3, True),
    ]:
        trainer = _make_validate_trainer(test_freq=test_freq, version=version)
        asyncio.run(trainer._fit_validate())
        ran = trainer._step_valid_time > 0
        assert ran is expected, f"test_freq={test_freq} version={version}"
        if not expected:
            assert trainer.cumulative_validation_time == 0.0


def test_no_validation_means_virtual_clock_equals_wall_clock():
    trainer = _make_trainer(first_sample_time=0.0)
    _run_step(trainer, consumer_end=10.0, samples=[_sample(10.0)], step_end=16.0)
    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=16.0)
    assert step_data[TIMING_PREFIX + "cumulative_validation_time"] == 0.0
    assert step_data[TIMING_PREFIX + "cumulative_save_time"] == 0.0
    assert (
        step_data[TIMING_PREFIX + "cumulative_training_time"]
        == step_data[TIMING_PREFIX + "wall_time_since_first_sample"]
    )


def test_validation_and_save_in_the_same_step_subtract_additively():
    trainer = _make_trainer(first_sample_time=0.0, cumulative_validation_time=3.0, cumulative_save_time=4.0)
    end = _run_step(trainer, consumer_end=10.0, samples=[_sample(10.0)], step_end=30.0, valid_time=3.0, save_time=4.0)
    assert end == 10.0 + (30.0 - 10.0) - 3.0 - 4.0 == 23.0
    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=30.0)
    assert step_data[TIMING_PREFIX + "cumulative_training_time"] == 23.0
    assert step_data[TIMING_PREFIX + "wall_time_since_first_sample"] == 30.0


# ------------------------------------------------------------ invariants


def test_virtual_clock_is_monotonic_and_bounded():
    trainer = _make_trainer(first_sample_time=0.0)
    pause = 0.0
    free = 0.0
    for i in range(1, 12):
        ready = 5.0 * i
        valid = 2.0 if i % 3 == 0 else 0.0
        save = 1.0 if i % 4 == 0 else 0.0
        consumer_end = max(free, ready + pause)
        step_end = consumer_end + 4.0 + valid + save
        prev = trainer.virtual_free_time
        _run_step(
            trainer,
            consumer_end=consumer_end,
            samples=[_sample(ready + pause, pause)],
            step_end=step_end,
            valid_time=valid,
            save_time=save,
        )
        assert prev is None or trainer.virtual_free_time >= prev, "virtual clock went backwards"
        trainer.cumulative_validation_time += valid
        trainer.cumulative_save_time += save
        pause += valid + save
        free = step_end
        step_data = {}
        trainer._add_cumulative_time_metrics(step_data, now=step_end)
        ctt = step_data[TIMING_PREFIX + "cumulative_training_time"]
        wall = step_data[TIMING_PREFIX + "wall_time_since_first_sample"]
        assert 0.0 <= ctt <= wall, f"step {i}: {ctt} not within [0, {wall}]"


def _reference_run(ready_times, busy_times):
    """Schedule with neither validation nor checkpointing: each step starts at
    max(trainer free, batch ready) and runs for its busy time."""
    free = None
    ends = []
    for ready, busy in zip(ready_times, busy_times, strict=True):
        start = ready if free is None else max(free, ready)
        free = start + busy
        ends.append(free)
    return ends


def test_randomized_run_matches_the_no_validation_reference():
    """The strongest invariant: over a randomized schedule with validation and
    save windows injected, the virtual clock must reproduce the step-end times
    of an identical run that had neither. Covers rollout-bound, trainer-bound
    and balanced (alternating bottleneck) regimes."""
    import random

    for seed in range(12):
        rng = random.Random(seed)
        n = 10
        # a mixture of fast and slow producers exercises both bottlenecks
        gaps = [rng.choice([1.0, 3.0, 7.0, 11.0]) for _ in range(n)]
        busy = [rng.choice([2.0, 5.0, 9.0]) for _ in range(n)]
        ready = []
        acc = 0.0
        for g in gaps:
            acc += g
            ready.append(acc)
        reference_ends = _reference_run(ready, busy)

        trainer = _make_trainer(first_sample_time=0.0)
        pause_total = 0.0
        actual_free = None
        for i in range(n):
            valid = rng.choice([0.0, 0.0, 4.0])
            save = rng.choice([0.0, 0.0, 2.0])
            # the producer is frozen for the pause, so the sample lands later
            actual_ready = ready[i] + pause_total
            consumer_end = actual_ready if actual_free is None else max(actual_free, actual_ready)
            step_end = consumer_end + busy[i] + valid + save
            _run_step(
                trainer,
                consumer_end=consumer_end,
                samples=[_sample(actual_ready, pause_total)],
                step_end=step_end,
                valid_time=valid,
                save_time=save,
            )
            assert abs(trainer.virtual_free_time - reference_ends[i]) < 1e-9, (
                f"seed={seed} step={i}: virtual {trainer.virtual_free_time} != reference {reference_ends[i]}"
            )
            pause_total += valid + save
            actual_free = step_end


# ------------------------------------------------------------ anchor lifecycle


def test_trainer_latches_the_anchor_and_never_unlatches_it():
    trainer = _make_trainer(first_sample_time=None)
    values = [None, None, 123.5, None]

    class _GetFirstSampleTime:
        def remote(self):
            async def run():
                return values.pop(0)

            return run()

    trainer.rollouter = SimpleNamespace(get_first_sample_time=_GetFirstSampleTime())
    asyncio.run(trainer._latch_first_sample_time())
    assert trainer.rollouter_first_sample_time is None, "None must not latch"
    asyncio.run(trainer._latch_first_sample_time())
    asyncio.run(trainer._latch_first_sample_time())
    assert trainer.rollouter_first_sample_time == 123.5
    asyncio.run(trainer._latch_first_sample_time())
    assert trainer.rollouter_first_sample_time == 123.5, "a later None must not clobber the anchor"
    assert values == [None], "the RPC must stop once the anchor is latched"


def test_rollouter_sets_the_anchor_once():
    """_process_single_sample_streaming stamps the anchor on the first sample
    only, and stamps every enqueued sample with the live pause totals."""
    import numpy as np

    rollouter = _make_pause_rollouter(first_sample_time=None)
    rollouter.processed_sample_count = 0
    rollouter.total_generated_samples = 0
    rollouter.dropped_stale_samples = 0

    class _Batch:
        def __init__(self):
            self.non_tensor_batch = {}

        def __len__(self):
            return 1

    rollouter.async_rollout_manager = SimpleNamespace(generate_sequences_single=_async_const(_Batch()))
    rollouter.get_statistics = _async_const({})
    put = []

    class _MQ:
        async def put_sample(self, sample):
            put.append(sample)
            return True

    rollouter.message_queue_client = _MQ()

    def _make_sample(sid):
        return SimpleNamespace(full_batch=_Batch(), sample_id=sid, rollout_status=None)

    import verl.experimental.fully_async_policy.fully_async_rollouter as rollouter_module

    original_np = rollouter_module.np
    rollouter_module.np = np
    try:
        before = time.time()
        asyncio.run(FullyAsyncRollouter._process_single_sample_streaming(rollouter, _make_sample("a")))
        anchor = rollouter.first_sample_time
        assert before <= anchor <= time.time()
        time.sleep(0.01)
        asyncio.run(FullyAsyncRollouter._process_single_sample_streaming(rollouter, _make_sample("b")))
        assert rollouter.first_sample_time == anchor, "the anchor must never be overwritten"
    finally:
        rollouter_module.np = original_np
    assert len(put) == 2


def test_pre_anchor_windows_excluded_on_the_trainer_side(tmp_path):
    """val_before_train runs before the rollouter has processed any sample. The
    rollouter excludes pre-anchor pauses from its accumulators, so the trainer
    must too, or the audit totals disagree."""
    trainer = _make_validate_trainer(test_freq=1, version=1)
    trainer.rollouter_first_sample_time = None
    asyncio.run(trainer._fit_validate(val_before_train=True))
    assert trainer.cumulative_validation_time == 0.0, "pre-anchor validation must not be accumulated"
    assert trainer._step_valid_time >= 0.02, "the stall is still excluded from this step's busy time"

    saver = _make_save_trainer(tmp_path)
    saver.rollouter_first_sample_time = None
    asyncio.run(saver._fit_save_checkpoint())
    assert saver.calls == ["save_checkpoint", "timing_state"]
    assert saver.cumulative_save_time == 0.0, "pre-anchor save must not be accumulated"
    assert saver._step_save_time > 0


def test_first_step_save_is_accounted(tmp_path):
    """Regression (caught by the 2-iteration smoke run on 8xH100, 2026-08-21).

    The anchor is fetched from the rollouter, which returns None until it has processed its
    first sample. If the trainer latches it before generation, step 1 runs anchorless: the
    pre-anchor guard then drops that step's save from cumulative_save_time and writes an
    all-zero timing_state.json. Observed: global_step_1/timing_state.json was
    {0.0, 0.0, 0.0, 0.0} and global_step_2 reported cumulative_save_time=0.0 despite two
    completed saves.
    """
    trainer = _make_save_trainer(tmp_path)
    trainer.rollouter_first_sample_time = None
    asyncio.run(trainer._fit_save_checkpoint())
    assert trainer.cumulative_save_time == 0.0, "pre-anchor saves are deliberately excluded"

    # ... which is why the anchor must be latched before the save runs.
    trainer2 = _make_save_trainer(tmp_path, save_freq=1, version=1)
    trainer2.rollouter_first_sample_time = 100.0
    asyncio.run(trainer2._fit_save_checkpoint())
    assert trainer2.cumulative_save_time > 0.0, "a step-1 save must be accounted once anchored"


def test_anchor_is_latched_after_generation_in_fit_step():
    """Pin the ordering the regression above depends on.

    `_latch_first_sample_time` must be called *after* `_fit_generate`, because only once a
    batch has been pulled is the rollouter guaranteed to have an anchor to report. This is a
    source-order assertion because the ordering, not any single function, is the invariant.
    """
    import inspect

    src = inspect.getsource(FullyAsyncTrainer.fit_step)
    gen = src.index("_fit_generate(")
    latch = src.index("_latch_first_sample_time(")
    assert latch > gen, "_latch_first_sample_time must run after _fit_generate"


# ------------------------------------------------- absolute timestamps in timing_state


def _read_state(tmp_path, name="ts"):
    with open(tmp_path / name / "timing_state.json") as f:
        return json.load(f)


def test_timing_state_records_anchor_and_save_instant(tmp_path):
    """The four totals are all relative to the anchor; without it written down a checkpoint
    cannot be placed against anything else."""
    from datetime import datetime

    anchor, save_start = 1_755_787_412.412, 1_755_788_489.001
    trainer = _make_trainer(first_sample_time=anchor)
    trainer.virtual_free_time = 900.0
    folder = tmp_path / "ts"
    folder.mkdir()
    trainer._save_timing_state(str(folder), save_start=save_start)

    state = _read_state(tmp_path)
    # round-trip: the strings must parse back to exactly the instants they were built from
    assert abs(datetime.fromisoformat(state["first_sample_time"]).timestamp() - anchor) < 1e-3
    assert abs(datetime.fromisoformat(state["checkpoint_save_started"]).timestamp() - save_start) < 1e-3
    # ISO-8601 with an explicit offset, so it is unambiguous without knowing the host's tz
    assert state["first_sample_time"][:4].isdigit() and "T" in state["first_sample_time"]
    assert state["first_sample_time"][-6] in "+-", "timestamp must carry a UTC offset"


def test_checkpoint_save_started_is_save_start_not_now(tmp_path):
    """The whole file's semantics rest on snapshotting at save_start; the timestamp must
    follow that, not wall-clock at write time."""
    from datetime import datetime

    save_start = time.time() - 3600.0  # an hour ago
    trainer = _make_trainer(first_sample_time=save_start - 10.0)
    folder = tmp_path / "ts"
    folder.mkdir()
    trainer._save_timing_state(str(folder), save_start=save_start)

    written = datetime.fromisoformat(_read_state(tmp_path)["checkpoint_save_started"]).timestamp()
    assert abs(written - save_start) < 1e-3
    assert time.time() - written > 3000.0, "wrote 'now' instead of save_start"


def test_timing_state_without_anchor_records_null_first_sample(tmp_path):
    """Regression-adjacent: a checkpoint written before the anchor is latched has four zero
    totals, which is indistinguishable from a lost anchor unless the file says so."""
    trainer = _make_trainer(first_sample_time=None)
    folder = tmp_path / "ts"
    folder.mkdir()
    trainer._save_timing_state(str(folder), save_start=1_755_788_489.001)

    state = _read_state(tmp_path)
    assert state["first_sample_time"] is None, "a missing anchor must be explicit, not omitted"
    assert state["checkpoint_save_started"] is not None, "the save instant is always known"
    assert state["wall_time_since_first_sample"] == 0.0
    assert state["cumulative_training_time"] == 0.0


def test_restore_ignores_the_absolute_timestamps(tmp_path):
    """A resumed run establishes its own anchor; the two timestamps describe the run that
    wrote them and must not perturb the restored offsets."""
    anchor = 1_755_787_412.412
    saver = _make_trainer(first_sample_time=anchor, cumulative_validation_time=5.0, cumulative_save_time=2.0)
    saver.virtual_free_time = 900.0
    folder = tmp_path / "ts"
    folder.mkdir()
    saver._save_timing_state(str(folder), save_start=anchor + 50.0)

    resumed = _make_trainer()
    resumed._restore_timing_state(str(folder))
    assert abs(resumed.timing_wall_offset - 50.0) < 1e-6
    assert resumed.timing_validation_offset == 5.0
    assert resumed.timing_save_offset == 2.0
    # and no attribute was invented from the new keys
    assert not hasattr(resumed, "first_sample_time")


def test_cumulative_metrics_aggregate_as_last_not_mean():
    """The virtual clock is cumulative, so it must survive multi-step param versions.

    Every metric the trainer emits is buffered by ``MetricsAggregator`` and drained once per
    parameter sync, so with ``trigger_parameter_sync_step > 1`` (the recipe default is 4) the
    aggregation rule decides what actually gets logged. These four keys are especially easy to
    get wrong: ``_get_aggregation_type``'s name heuristic does substring matching, and the
    ``fully_async/timing/`` prefix contains ``"min"`` (ti-MIN-g), so absent an exact-name rule
    they aggregate as *min* -- the version's first step, a whole param version stale and flat
    across it. The ``fully_async/count/*`` counters are listed as ``"last"`` for the same
    reason.
    """
    from verl.experimental.fully_async_policy.detach_utils import MetricsAggregator

    agg = MetricsAggregator(total_gpus=1)
    trainer = _make_trainer(first_sample_time=1000.0, cumulative_validation_time=7.0, cumulative_save_time=3.0)
    trainer.virtual_free_time = 1040.0
    step_data = {}
    trainer._add_cumulative_time_metrics(step_data, now=1100.0)

    assert step_data, "the anchor is set, so metrics must be emitted"
    for key in step_data:
        assert agg._get_aggregation_type(key) == "last", f"{key} falls through to the name heuristic"

    # And end-to-end: two steps of one param version must report the second step's value.
    for value in (10.0, 20.0):
        agg.add_step_metrics(metrics={"fully_async/timing/cumulative_training_time": value}, sample_count=1)
    out = agg.get_aggregated_metrics()
    assert out["fully_async/timing/cumulative_training_time"] == 20.0, "must be last, not the 15.0 mean"
