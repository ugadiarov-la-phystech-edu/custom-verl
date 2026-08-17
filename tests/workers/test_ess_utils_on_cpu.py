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

"""CPU tests for ESS-guided LR scaling (verl/workers/utils/ess.py):
the pure math (base resolution, trigger deadband, Kish ESS, per-sequence IS
sums) and the pre-optimizer-step hook contract (LR scale/restore around the
step, structured staleness/ess entry, reserved _ess/* keys popped)."""

import math

import pytest
import torch
import torch.nn as nn

from verl.utils import tensordict_utils as tu
from verl.workers.config import ESSScalingConfig
from verl.workers.utils.ess import (
    ESS_SUM_KEYS,
    compute_ess_lr_scale,
    compute_seq_is_sums,
    ess_ratio_from_sums,
    make_ess_optimizer_step_hook,
    resolve_ess_base,
)

NOMINAL_LR = 0.1


class TestResolveEssBase:
    def test_config_value_wins_over_override(self):
        assert resolve_ess_base(0.016, 0.033) == 0.016

    def test_none_config_falls_back_to_override(self):
        assert resolve_ess_base(None, 0.033) == 0.033

    def test_unresolved_returns_none(self):
        assert resolve_ess_base(None, None) is None

    def test_explicit_zero_config_wins(self):
        # falsy-vs-None surface: an explicit 0.0 must NOT fall through to the override
        assert resolve_ess_base(0.0, 0.5) == 0.0


class TestComputeEssLrScale:
    def test_legacy_attenuation(self):
        assert compute_ess_lr_scale(0.25, 0.5) == pytest.approx(0.5)
        assert compute_ess_lr_scale(0.5, 0.5) == 1.0

    def test_trigger_full_lr_at_or_above_threshold(self):
        assert compute_ess_lr_scale(0.4, 0.5, 0.5) == 1.0  # ratio 0.8 >= 0.5
        assert compute_ess_lr_scale(0.25, 0.5, 0.5) == 1.0  # exactly at threshold

    def test_trigger_discontinuity(self):
        eps = 1e-9
        assert compute_ess_lr_scale(0.25, 0.5, 0.5) == 1.0
        assert compute_ess_lr_scale(0.25 - eps, 0.5, 0.5) == pytest.approx(0.5, abs=1e-6)

    def test_zero_base_never_divides_by_zero(self):
        assert compute_ess_lr_scale(0.5, 0.0) == 1.0

    def test_tiny_base_does_not_explode(self):
        # the 1e-8 floor keeps the ratio finite; a tiny base just gives full lr
        assert compute_ess_lr_scale(0.5, 1e-12) == 1.0

    def test_zero_ess_and_zero_base_gives_zero(self):
        assert compute_ess_lr_scale(0.0, 0.0) == 0.0

    def test_explicit_none_trigger_is_legacy(self):
        for ess, base in [(0.25, 0.5), (0.5, 0.5), (0.75, 0.5)]:
            assert compute_ess_lr_scale(ess, base, None) == compute_ess_lr_scale(ess, base)

    def test_trigger_one_matches_legacy(self):
        for ess in (0.1, 0.25, 0.5, 0.75):
            assert compute_ess_lr_scale(ess, 0.5, 1.0) == pytest.approx(compute_ess_lr_scale(ess, 0.5))

    def test_trigger_above_one_matches_legacy(self):
        # ratios in (1, trigger) hit the min(1, ratio)=1 branch anyway
        for ess in (0.1, 0.4, 0.5, 0.75):
            assert compute_ess_lr_scale(ess, 0.5, 2.0) == pytest.approx(compute_ess_lr_scale(ess, 0.5))


class TestEssRatioFromSums:
    def test_equal_weights_give_ratio_one(self):
        ess, ratio = ess_ratio_from_sums(4.0, 4.0, 4.0)  # four weights of 1.0
        assert ess == pytest.approx(4.0, rel=1e-6)
        assert ratio == pytest.approx(1.0, rel=1e-6)

    def test_dominant_weight_gives_ratio_one_over_n(self):
        # weights [8, ~0, ~0, ~0]
        ess, ratio = ess_ratio_from_sums(8.0, 64.0, 4.0)
        assert ess == pytest.approx(1.0, rel=1e-6)
        assert ratio == pytest.approx(0.25, rel=1e-6)

    def test_empty_count(self):
        assert ess_ratio_from_sums(0.0, 0.0, 0.0) == (0.0, 0.0)


class TestComputeSeqIsSums:
    def _make(self, seq_is_targets, resp_len=6, dtype=torch.float32):
        n = len(seq_is_targets)
        log_prob = torch.zeros(n, resp_len, dtype=dtype)
        rollout = torch.zeros(n, resp_len, dtype=dtype)
        for i, w in enumerate(seq_is_targets):
            rollout[i, :] = -math.log(w) / resp_len
        mask = torch.ones(n, resp_len, dtype=torch.long)
        return log_prob, rollout, mask

    def test_sums_match_engineered_weights(self):
        weights = [2.0, 0.5, 1.0]
        sums = compute_seq_is_sums(*self._make(weights), threshold=None)
        assert sums["_ess/is_sum"] == pytest.approx(sum(weights), rel=1e-5)
        assert sums["_ess/is_sq_sum"] == pytest.approx(sum(w * w for w in weights), rel=1e-5)
        assert sums["_ess/count"] == 3.0

    def test_clipping_at_threshold(self):
        weights = [8.0, 0.5]
        sums = compute_seq_is_sums(*self._make(weights), threshold=2.0)
        assert sums["_ess/is_sum"] == pytest.approx(8.5, rel=1e-5)  # unclipped untouched
        assert sums["_ess/is_clipped_sum"] == pytest.approx(2.5, rel=1e-5)
        assert sums["_ess/is_clipped_sq_sum"] == pytest.approx(4.25, rel=1e-5)

    def test_mask_excludes_tokens(self):
        log_prob, rollout, mask = self._make([2.0])
        mask[0, 3:] = 0  # only half the tokens counted -> w = 2^(3/6) = sqrt(2)
        sums = compute_seq_is_sums(log_prob, rollout, mask, threshold=None)
        assert sums["_ess/is_sum"] == pytest.approx(math.sqrt(2.0), rel=1e-5)

    def test_bf16_inputs_long_sequence_stay_accurate(self):
        # fp32 accumulation must survive bf16 inputs over thousands of tokens
        log_prob, rollout, mask = self._make([8.0], resp_len=4096)
        sums = compute_seq_is_sums(log_prob.to(torch.bfloat16), rollout.to(torch.bfloat16), mask, threshold=None)
        assert sums["_ess/is_sum"] == pytest.approx(8.0, rel=0.05)


class _StubEngine:
    def __init__(self, lrs=(NOMINAL_LR,)):
        params = [nn.Parameter(torch.zeros(1)) for _ in lrs]
        self.optimizer = torch.optim.SGD([{"params": [p], "lr": lr} for p, lr in zip(params, lrs, strict=True)])

    def get_optimizer_param_groups(self):
        return self.optimizer.param_groups

    def get_data_parallel_group(self):
        return None


def _make_outputs(seq_is_lists):
    """seq_is_lists: list of per-micro-batch weight lists."""
    metrics = {}
    for weights in seq_is_lists:
        sums = {
            "_ess/is_sum": sum(weights),
            "_ess/is_sq_sum": sum(w * w for w in weights),
            "_ess/is_clipped_sum": sum(min(w, 2.0) for w in weights),
            "_ess/is_clipped_sq_sum": sum(min(w, 2.0) ** 2 for w in weights),
            "_ess/count": float(len(weights)),
        }
        for key, val in sums.items():
            metrics.setdefault(key, []).append(val)
    return {"metrics": metrics}


def _make_data(**non_tensor):
    return tu.get_tensordict(tensor_dict={}, non_tensor_dict=non_tensor)


class TestEssOptimizerStepHook:
    def test_scales_and_restores_lr(self):
        hook = make_ess_optimizer_step_hook(ESSScalingConfig(enable=True, base_ess_ratio=1.0))
        engine = _StubEngine()
        outputs = _make_outputs([[8.0, 1e-8], [1e-8, 1e-8]])  # ratio 0.25 across 4 seqs
        with hook(engine, _make_data(minibatch_idx=0), outputs):
            assert engine.optimizer.param_groups[0]["lr"] == pytest.approx(NOMINAL_LR * 0.25**0.5)
        assert engine.optimizer.param_groups[0]["lr"] == pytest.approx(NOMINAL_LR)

    def test_restores_lr_on_exception(self):
        hook = make_ess_optimizer_step_hook(ESSScalingConfig(enable=True, base_ess_ratio=1.0))
        engine = _StubEngine()
        with pytest.raises(RuntimeError):
            with hook(engine, _make_data(), _make_outputs([[8.0, 1e-8, 1e-8, 1e-8]])):
                raise RuntimeError("simulated non-finite grad handling")
        assert engine.optimizer.param_groups[0]["lr"] == pytest.approx(NOMINAL_LR)

    def test_multiple_param_groups(self):
        hook = make_ess_optimizer_step_hook(ESSScalingConfig(enable=True, base_ess_ratio=1.0))
        engine = _StubEngine(lrs=(0.1, 0.05))
        with hook(engine, _make_data(), _make_outputs([[8.0, 1e-8, 1e-8, 1e-8]])):
            assert engine.optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 0.5)
            assert engine.optimizer.param_groups[1]["lr"] == pytest.approx(0.05 * 0.5)
        assert engine.optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
        assert engine.optimizer.param_groups[1]["lr"] == pytest.approx(0.05)

    def test_trigger_no_scaling_at_or_above_threshold(self):
        hook = make_ess_optimizer_step_hook(ESSScalingConfig(enable=True, base_ess_ratio=0.25, trigger_ratio=0.5))
        engine = _StubEngine()
        with hook(engine, _make_data(), _make_outputs([[8.0, 1e-8, 1e-8, 1e-8]])):  # ratio/base = 1
            assert engine.optimizer.param_groups[0]["lr"] == pytest.approx(NOMINAL_LR)

    def test_unresolved_base_is_noop_but_entry_emitted(self):
        hook = make_ess_optimizer_step_hook(ESSScalingConfig(enable=True, base_ess_ratio=None))
        engine = _StubEngine()
        outputs = _make_outputs([[8.0, 1e-8, 1e-8, 1e-8]])
        with hook(engine, _make_data(), outputs):
            assert engine.optimizer.param_groups[0]["lr"] == pytest.approx(NOMINAL_LR)
        (entry,) = outputs["metrics"]["staleness/ess"]
        assert entry["base_ess_ratio"] is None
        assert entry["ess_scaled_lr"] == pytest.approx(NOMINAL_LR)

    def test_base_override_from_tensordict(self):
        hook = make_ess_optimizer_step_hook(ESSScalingConfig(enable=True, base_ess_ratio=None))
        engine = _StubEngine()
        outputs = _make_outputs([[8.0, 1e-8, 1e-8, 1e-8]])
        with hook(engine, _make_data(ess_base_override=1.0), outputs):
            assert engine.optimizer.param_groups[0]["lr"] == pytest.approx(NOMINAL_LR * 0.5)
        (entry,) = outputs["metrics"]["staleness/ess"]
        assert entry["base_ess_ratio"] == 1.0

    def test_entry_contract_and_reserved_keys_popped(self):
        hook = make_ess_optimizer_step_hook(ESSScalingConfig(enable=True, base_ess_ratio=1.0))
        outputs = _make_outputs([[1.0, 1.0], [1.0, 1.0]])
        with hook(_StubEngine(), _make_data(minibatch_idx=7), outputs):
            pass
        metrics = outputs["metrics"]
        for key in ESS_SUM_KEYS:
            assert key not in metrics
        (entry,) = metrics["staleness/ess"]
        assert set(entry.keys()) == {
            "minibatch_idx",
            "minibatch_ess",
            "minibatch_ess_clipped",
            "minibatch_ess_ratio",
            "minibatch_ess_ratio_clipped",
            "ess_scaled_lr",
            "base_ess_ratio",
        }
        assert entry["minibatch_idx"] == 7
        assert entry["minibatch_ess_ratio"] == pytest.approx(1.0, rel=1e-6)

    def test_micro_batch_sums_accumulate(self):
        # two micro-batches of two equal weights == one micro-batch of four
        hook = make_ess_optimizer_step_hook(ESSScalingConfig(enable=True, base_ess_ratio=1.0))
        split = _make_outputs([[1.0, 1.0], [1.0, 1.0]])
        whole = _make_outputs([[1.0, 1.0, 1.0, 1.0]])
        for outputs in (split, whole):
            with hook(_StubEngine(), _make_data(), outputs):
                pass
        (e1,) = split["metrics"]["staleness/ess"]
        (e2,) = whole["metrics"]["staleness/ess"]
        assert e1["minibatch_ess"] == pytest.approx(e2["minibatch_ess"], rel=1e-9)

    def test_missing_sums_is_inert(self):
        hook = make_ess_optimizer_step_hook(ESSScalingConfig(enable=True, base_ess_ratio=1.0))
        engine = _StubEngine()
        outputs = {"metrics": {}}
        with hook(engine, _make_data(), outputs):
            assert engine.optimizer.param_groups[0]["lr"] == pytest.approx(NOMINAL_LR)
        assert "staleness/ess" not in outputs["metrics"]


class TestTrainBatchHookOrdering:
    """BaseEngine.train_batch must run the hook's context around optimizer_step
    and keep the default path byte-identical when no hook is installed."""

    class _MiniEngine(__import__("verl.workers.engine.base", fromlist=["BaseEngine"]).BaseEngine):
        def __init__(self):
            self.stub = _StubEngine()
            self.stepped_lrs = []

        def optimizer_zero_grad(self):
            pass

        def forward_backward_batch(self, data, loss_function, forward_only=False):
            return {"metrics": dict(data_metrics), "loss": [0.0]}

        def optimizer_step(self):
            self.stepped_lrs.append(float(self.stub.optimizer.param_groups[0]["lr"]))
            return 0.5

        def is_mp_src_rank_with_outputs(self):
            return True

        def get_optimizer_param_groups(self):
            return self.stub.get_optimizer_param_groups()

        def get_data_parallel_group(self):
            return None

    def test_hook_context_wraps_optimizer_step(self):
        global data_metrics
        data_metrics = _make_outputs([[8.0, 1e-8, 1e-8, 1e-8]])["metrics"]  # ratio 0.25
        engine = self._MiniEngine()
        hook = make_ess_optimizer_step_hook(ESSScalingConfig(enable=True, base_ess_ratio=1.0))
        data = _make_data(minibatch_idx=0)
        outputs = engine.train_batch(data, loss_function=None, pre_optimizer_step_hook=hook)
        # LR was scaled DURING the step and restored afterwards
        assert engine.stepped_lrs == [pytest.approx(NOMINAL_LR * 0.5)]
        assert engine.stub.optimizer.param_groups[0]["lr"] == pytest.approx(NOMINAL_LR)
        # grad_norm added, structured entry present, reserved keys popped
        assert outputs["metrics"]["grad_norm"] == 0.5
        assert len(outputs["metrics"]["staleness/ess"]) == 1
        for key in ESS_SUM_KEYS:
            assert key not in outputs["metrics"]

    def test_no_hook_default_path(self):
        global data_metrics
        data_metrics = {}
        engine = self._MiniEngine()
        outputs = engine.train_batch(_make_data(), loss_function=None)
        assert engine.stepped_lrs == [pytest.approx(NOMINAL_LR)]
        assert outputs["metrics"]["grad_norm"] == 0.5
