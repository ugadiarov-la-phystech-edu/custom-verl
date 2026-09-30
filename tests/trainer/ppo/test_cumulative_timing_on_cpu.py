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
"""fully_async/timing/* cumulative metrics on the v1 trainer.

The v1 ``fit`` loop times ``save_checkpoint`` inside ``step`` and ``testing`` outside it, so
cumulative training time is the sum of ``step - save_checkpoint``. These tests pin both the helper's
arithmetic and the loop layout it depends on.
"""

import ast
import contextlib
import inspect
import math
import textwrap
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.metric_utils import CUMULATIVE_TIMING_PREFIX, compute_cumulative_timing_metrics

WALL = f"{CUMULATIVE_TIMING_PREFIX}/wall_time_since_first_sample"
VAL = f"{CUMULATIVE_TIMING_PREFIX}/cumulative_validation_time"
SAVE = f"{CUMULATIVE_TIMING_PREFIX}/cumulative_save_time"
TRAIN = f"{CUMULATIVE_TIMING_PREFIX}/cumulative_training_time"


# --------------------------------------------------------------------------- helper arithmetic


class TestHelper:
    def test_returns_exactly_the_four_keys(self):
        assert set(compute_cumulative_timing_metrics({}, {"step": 1.0})) == {WALL, VAL, SAVE, TRAIN}

    def test_keys_match_the_fully_async_trainer_names(self):
        assert CUMULATIVE_TIMING_PREFIX == "fully_async/timing"

    def test_plain_step(self):
        out = compute_cumulative_timing_metrics({}, {"step": 10.0})
        assert out == {WALL: 10.0, VAL: 0.0, SAVE: 0.0, TRAIN: 10.0}

    def test_save_is_inside_step_and_subtracted(self):
        out = compute_cumulative_timing_metrics({}, {"step": 10.0, "save_checkpoint": 3.0})
        assert out[TRAIN] == 7.0
        assert out[SAVE] == 3.0
        # save is part of step, so it must not be added to wall a second time
        assert out[WALL] == 10.0

    def test_testing_is_outside_step_and_added_to_wall(self):
        out = compute_cumulative_timing_metrics({}, {"step": 10.0, "testing": 5.0})
        assert out[WALL] == 15.0
        assert out[VAL] == 5.0
        assert out[TRAIN] == 10.0

    def test_all_three_timers(self):
        out = compute_cumulative_timing_metrics({}, {"step": 10.0, "save_checkpoint": 3.0, "testing": 5.0})
        assert out == {WALL: 15.0, VAL: 5.0, SAVE: 3.0, TRAIN: 7.0}

    @pytest.mark.parametrize("timing_raw", [{}, {"gen": 4.0, "update_actor": 2.0}])
    def test_missing_timers_count_as_zero(self, timing_raw):
        out = compute_cumulative_timing_metrics({}, timing_raw)
        assert out == {WALL: 0.0, VAL: 0.0, SAVE: 0.0, TRAIN: 0.0}

    def test_sub_timers_are_ignored(self):
        # gen / update_actor are nested inside step; counting them would double-count
        out = compute_cumulative_timing_metrics({}, {"step": 10.0, "gen": 6.0, "update_actor": 3.0})
        assert out[TRAIN] == 10.0 and out[WALL] == 10.0

    def test_training_never_goes_negative_on_timer_jitter(self):
        out = compute_cumulative_timing_metrics({}, {"step": 2.999, "save_checkpoint": 3.0})
        assert out[TRAIN] == 0.0
        assert out[SAVE] == 3.0

    def test_accepts_ints_and_returns_floats(self):
        out = compute_cumulative_timing_metrics({}, {"step": 4, "save_checkpoint": 1, "testing": 2})
        assert all(isinstance(v, float) for v in out.values())
        assert out == {WALL: 6.0, VAL: 2.0, SAVE: 1.0, TRAIN: 3.0}

    def test_accumulates_across_steps(self):
        cumulative = {}
        steps = [
            {"step": 10.0},
            {"step": 12.0, "save_checkpoint": 2.0},
            {"step": 9.0, "testing": 4.0},
            {"step": 11.0, "save_checkpoint": 1.5, "testing": 6.0},
        ]
        for timing_raw in steps:
            out = compute_cumulative_timing_metrics(cumulative, timing_raw)
        assert out == {WALL: 52.0, VAL: 10.0, SAVE: 3.5, TRAIN: 38.5}

    def test_monotone_non_decreasing(self):
        cumulative, prev = {}, None
        for i in range(20):
            timing_raw = {"step": 1.0 + i % 3, "save_checkpoint": 0.5 * (i % 2), "testing": float(i % 4 == 0)}
            out = compute_cumulative_timing_metrics(cumulative, timing_raw)
            if prev is not None:
                assert all(out[k] >= prev[k] for k in out)
            prev = out

    def test_identity_training_equals_wall_minus_validation_minus_save(self):
        cumulative = {}
        for i in range(50):
            timing_raw = {
                "step": 3.0 + (i % 7) * 0.37,
                "save_checkpoint": (i % 5 == 0) * 0.9,
                "testing": (i % 3 == 0) * 1.3,
            }
            out = compute_cumulative_timing_metrics(cumulative, timing_raw)
            assert math.isclose(out[TRAIN], out[WALL] - out[VAL] - out[SAVE], abs_tol=1e-9)

    def test_updates_the_running_dict_in_place(self):
        cumulative = {}
        compute_cumulative_timing_metrics(cumulative, {"step": 2.0, "testing": 1.0})
        assert cumulative == {"wall": 3.0, "validation": 1.0, "save": 0.0, "training": 2.0}

    def test_independent_dicts_do_not_share_state(self):
        a, b = {}, {}
        compute_cumulative_timing_metrics(a, {"step": 5.0})
        out_b = compute_cumulative_timing_metrics(b, {"step": 1.0})
        assert out_b[WALL] == 1.0
        assert a["wall"] == 5.0

    def test_does_not_mutate_timing_raw(self):
        timing_raw = {"step": 5.0, "save_checkpoint": 1.0}
        compute_cumulative_timing_metrics({}, timing_raw)
        assert timing_raw == {"step": 5.0, "save_checkpoint": 1.0}


# --------------------------------------------------------------------------- v1 loop layout tripwire


def _fit_ast():
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    source = textwrap.dedent(inspect.getsource(PPOTrainer.fit))
    return ast.parse(source).body[0]


def _timer_name(with_node):
    for item in with_node.items:
        call = item.context_expr
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "marked_timer":
            arg = call.args[0]
            if isinstance(arg, ast.Constant):
                return arg.value
    return None


def _timers_inside(node):
    return {_timer_name(n) for n in ast.walk(node) if isinstance(n, ast.With)} - {None}


class TestFitLayout:
    """The helper's semantics are only right while the loop keeps this timer nesting."""

    def _step_with(self):
        steps = [n for n in ast.walk(_fit_ast()) if isinstance(n, ast.With) and _timer_name(n) == "step"]
        assert len(steps) == 1, "expected exactly one marked_timer('step') in fit()"
        return steps[0]

    def test_save_checkpoint_is_timed_inside_step(self):
        assert "save_checkpoint" in _timers_inside(self._step_with())

    def test_testing_is_timed_outside_step(self):
        assert "testing" not in _timers_inside(self._step_with())
        assert "testing" in _timers_inside(_fit_ast())

    def test_fit_resets_the_running_totals_and_calls_the_helper(self):
        source = inspect.getsource(__import__("verl.trainer.ppo.v1.trainer_base", fromlist=["x"]).PPOTrainer.fit)
        assert "self._cumulative_timing" in source
        assert "compute_cumulative_timing_metrics(self._cumulative_timing, self.timing_raw)" in source


# --------------------------------------------------------------------------- fit() integration


DURATIONS = {"step": 10.0, "save_checkpoint": 3.0, "testing": 5.0}


@contextlib.contextmanager
def _fake_marked_timer(name, timing_raw, *args, **kwargs):
    yield
    timing_raw[name] = timing_raw.get(name, 0.0) + DURATIONS.get(name, 0.0)


class _Recorder:
    def __init__(self, *args, **kwargs):
        self.logged = []

    def log(self, data, step, *args, **kwargs):
        self.logged.append((step, dict(data)))


def _make_trainer(monkeypatch, total_steps, save_freq, test_freq):
    import verl.trainer.ppo.v1.trainer_base as tb

    recorder = _Recorder()
    monkeypatch.setattr(tb, "Tracking", lambda *a, **k: recorder)
    monkeypatch.setattr(tb, "ValidationGenerationsLogger", lambda *a, **k: None)
    monkeypatch.setattr(tb, "DapoFilteredRewardTableLogger", lambda *a, **k: _Recorder())
    monkeypatch.setattr(tb, "SkipManager", SimpleNamespace(init=lambda *a: None, set_step=lambda *a: None))
    monkeypatch.setattr(tb, "tq", SimpleNamespace(kv_clear=lambda **k: None))
    monkeypatch.setattr(tb, "marked_timer", _fake_marked_timer)
    monkeypatch.setattr(tb, "pprint", lambda *a, **k: None)

    from verl.trainer.ppo.v1 import get_trainer_cls

    # The concrete class main_ppo runs for trainer.v1.trainer_mode=sync (the baseline scripts).
    trainer = object.__new__(get_trainer_cls("sync"))
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "p",
                "experiment_name": "e",
                "logger": ["console"],
                "val_before_train": False,
                "total_epochs": 1,
                "save_freq": save_freq,
                "test_freq": test_freq,
                "rollout_data_dir": None,
            },
            "global_profiler": {"steps": None},
        }
    )
    trainer.global_steps = 0
    trainer.steps_per_epoch = total_steps
    trainer.total_training_steps = total_steps

    batch = SimpleNamespace(keys=[], partition_id="p")
    stubs = {
        "on_train_begin": lambda: None,
        "on_train_end": lambda: None,
        "on_step_begin": lambda: None,
        "on_step_end": lambda: None,
        "on_validate_begin": lambda: None,
        "on_validate_end": lambda: None,
        "_reissue_inflight_prompts": lambda: None,
        "_start_profiling": lambda: None,
        "_stop_profiling": lambda: None,
        "_shutdown_dump_executor": lambda: None,
        "_consume_sync_metrics": lambda: {},
        "_save_checkpoint": lambda: None,
        "_validate": lambda: {"val/score": 1.0},
        "_compute_metrics": lambda *a, **k: None,
        "step": lambda metrics, timing_raw: batch,
    }
    for name, fn in stubs.items():
        setattr(trainer, name, fn)
    return trainer, recorder


class TestFitIntegration:
    def test_metrics_reach_the_logger_every_step(self, monkeypatch):
        trainer, recorder = _make_trainer(monkeypatch, total_steps=3, save_freq=0, test_freq=0)
        trainer.fit(agent_loop_manager=None)
        steps = [(s, m) for s, m in recorder.logged if TRAIN in m]
        assert [s for s, _ in steps] == [1, 2, 3]
        assert [m[TRAIN] for _, m in steps] == [10.0, 20.0, 30.0]
        assert all(m[VAL] == 0.0 and m[SAVE] == 0.0 for _, m in steps)

    def test_save_and_validation_are_excluded_from_training_time(self, monkeypatch):
        # save and validate every 2nd step, plus the forced ones on the last step (4)
        trainer, recorder = _make_trainer(monkeypatch, total_steps=4, save_freq=2, test_freq=2)
        trainer.fit(agent_loop_manager=None)
        last = [m for _, m in recorder.logged if TRAIN in m][-1]
        # 4 steps x 10s; saves on steps 2 and 4 (3s each, inside step); validation on 2 and 4 (5s each)
        assert last[SAVE] == 6.0
        assert last[VAL] == 10.0
        assert last[TRAIN] == 40.0 - 6.0
        assert last[WALL] == 40.0 + 10.0
        assert math.isclose(last[TRAIN], last[WALL] - last[VAL] - last[SAVE])

    def test_initial_validation_is_not_counted(self, monkeypatch):
        trainer, recorder = _make_trainer(monkeypatch, total_steps=1, save_freq=0, test_freq=0)
        trainer.config.trainer.val_before_train = True
        trainer.fit(agent_loop_manager=None)
        step_metrics = [m for _, m in recorder.logged if TRAIN in m]
        assert len(step_metrics) == 1
        assert step_metrics[0][VAL] == 0.0 and step_metrics[0][WALL] == 10.0

    def test_running_totals_restart_on_each_fit_call(self, monkeypatch):
        trainer, recorder = _make_trainer(monkeypatch, total_steps=1, save_freq=0, test_freq=0)
        trainer.fit(agent_loop_manager=None)
        trainer.global_steps = 0
        trainer.fit(agent_loop_manager=None)
        values = [m[TRAIN] for _, m in recorder.logged if TRAIN in m]
        assert values == [10.0, 10.0]
