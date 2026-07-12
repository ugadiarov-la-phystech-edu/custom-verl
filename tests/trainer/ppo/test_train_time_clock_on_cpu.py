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
"""
Tests for the training-time clock (training/train_time_s metric) on RayPPOTrainer.

The clock accumulates wall time spent training only: validation and checkpoint
saving are excluded, so validation scores can be plotted against pure training time.
"""

from unittest.mock import patch

import verl.trainer.ppo.ray_trainer as ray_trainer_module
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class _FakeClock:
    """Deterministic stand-in for time.time()."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_trainer() -> RayPPOTrainer:
    # The clock helpers only touch train_time_s/_train_clock_mark (class-level
    # defaults), so the trainer needs no config or worker setup.
    return RayPPOTrainer.__new__(RayPPOTrainer)


def test_class_defaults():
    trainer = _make_trainer()
    assert trainer.train_time_s == 0.0
    assert trainer._train_clock_mark is None


def test_noop_before_start():
    # fully_async_main calls _fit_validate(True) before fit() starts the clock;
    # advance/skip must be no-ops until then.
    trainer = _make_trainer()
    assert trainer._train_clock_advance() == 0.0
    trainer._train_clock_skip()
    assert trainer._train_clock_mark is None
    assert trainer.train_time_s == 0.0


def test_advance_accumulates_training_time():
    trainer = _make_trainer()
    clock = _FakeClock()
    with patch.object(ray_trainer_module, "time", clock):
        trainer._train_clock_start()
        clock.advance(30.0)
        assert trainer._train_clock_advance() == 30.0
        clock.advance(45.0)
        assert trainer._train_clock_advance() == 75.0


def test_advance_subtracts_paused_time():
    # Sync/separation trainers pass timing_raw["testing"] + timing_raw["save_checkpoint"]
    # as paused_s: a step with 60s wall time and 25s validation banks 35s.
    trainer = _make_trainer()
    clock = _FakeClock()
    with patch.object(ray_trainer_module, "time", clock):
        trainer._train_clock_start()
        clock.advance(60.0)
        assert trainer._train_clock_advance(paused_s=25.0) == 35.0


def test_skip_discards_pause():
    # The fully-async trainer brackets validation/checkpointing with advance() + skip():
    # time between the two must not be banked.
    trainer = _make_trainer()
    clock = _FakeClock()
    with patch.object(ray_trainer_module, "time", clock):
        trainer._train_clock_start()
        clock.advance(20.0)
        trainer._train_clock_advance()  # bank training time up to the pause
        clock.advance(300.0)  # validation
        trainer._train_clock_skip()
        clock.advance(10.0)
        assert trainer._train_clock_advance() == 30.0


def test_checkpoint_snapshot_includes_in_progress_step():
    # _save_checkpoint snapshots train_time_s plus the time since the last mark, so a
    # mid-step save persists the current step's training time without stopping the clock.
    trainer = _make_trainer()
    clock = _FakeClock()
    with patch.object(ray_trainer_module, "time", clock):
        trainer._train_clock_start()
        clock.advance(100.0)
        trainer._train_clock_advance()
        clock.advance(40.0)  # in-progress step at the moment _save_checkpoint runs
        snapshot = trainer.train_time_s + (clock.time() - trainer._train_clock_mark)
        assert snapshot == 140.0
        # the live clock itself is unaffected by the snapshot
        assert trainer._train_clock_advance() == 140.0


def test_subclasses_inherit_clock():
    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
    from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer
    from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer

    for cls in (SeparateRayPPOTrainer, OneStepOffRayTrainer, FullyAsyncTrainer):
        assert cls.train_time_s == 0.0
        assert cls._train_clock_mark is None
        for helper in ("_train_clock_start", "_train_clock_advance", "_train_clock_skip"):
            assert hasattr(cls, helper)
