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

"""ESS-guided learning-rate scaling (VCPO, arXiv:2602.17616).

Sequence-level importance-sampling ratios against the behavior (rollout)
policy predict gradient-estimate quality: their effective sample size
ESS = (sum w)^2 / (sum w^2) collapses when a mini-batch is dominated by a few
stale sequences. The brake scales the optimizer LR by
``rule(min(1, ess_ratio / base_ess_ratio))`` for that step only.

The pieces:

- ``compute_seq_is_sums`` runs inside the loss function per micro-batch and
  emits plain-float partial sums under reserved ``_ess/*`` metric keys.
- ``make_ess_optimizer_step_hook`` builds the engine-level pre-optimizer-step
  hook: it pops those sums, all-reduces them over the data-parallel group,
  computes the LR multiplier, scales/restores the optimizer LRs around
  ``optimizer_step()``, and emits one structured ``staleness/ess`` entry per
  mini-batch (the driver-side auto-base calibration consumes these).
"""

from __future__ import annotations

import math
from contextlib import AbstractContextManager

import torch

from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name, get_torch_device

__all__ = [
    "ESS_SUM_KEYS",
    "compute_ess_lr_scale",
    "compute_seq_is_sums",
    "ess_ratio_from_sums",
    "make_ess_optimizer_step_hook",
    "resolve_ess_base",
]

# Reserved per-micro-batch metric keys; consumed (popped) by the optimizer-step
# hook before metrics reach the DP allgather, so they never leak to the driver.
ESS_SUM_KEYS = (
    "_ess/is_sum",
    "_ess/is_sq_sum",
    "_ess/is_clipped_sum",
    "_ess/is_clipped_sq_sum",
    "_ess/count",
)


def resolve_ess_base(config_base, override):
    """Resolve the ESS-scaling reference ratio.

    An explicit config value wins; base_ess_ratio=None (auto-calibration)
    resolves to the driver-provided override — the first update's measured
    on-policy ESS ratio, passed back via meta_info["ess_base_override"].
    Returns None while neither is available (scaling is then a no-op).
    """
    return config_base if config_base is not None else override


def compute_ess_lr_scale(ess_ratio: float, base_ess_ratio: float, trigger_ratio: float | None = None) -> float:
    """LR multiplier of the ESS brake (before the sqrt/linear rule).

    Legacy (trigger_ratio=None): min(1, ess_ratio / base) — attenuate whenever
    the measured ESS falls below the reference. With ess_scaling.trigger_ratio
    set, scaling engages only when ess_ratio / base < trigger_ratio; at or
    above the threshold the mini-batch runs at full nominal lr (the multiplier
    jumps discontinuously at the threshold).
    """
    ratio = float(ess_ratio) / max(float(base_ess_ratio), 1e-8)
    if trigger_ratio is not None and ratio >= float(trigger_ratio):
        return 1.0
    return min(1.0, ratio)


def compute_seq_is_sums(
    log_prob: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    threshold: float | None,
) -> dict[str, float]:
    """Per-micro-batch partial sums of sequence-level IS ratios.

    fp32 throughout: a low-precision running sum over thousands of tokens
    loses the per-token increments once the partial sum grows, distorting
    exactly what the brake watches. The anchor is the update's own forward
    (``log_prob.detach()``); ``rollout_log_probs`` is the behavior policy.
    """
    with torch.no_grad():
        mask = response_mask.to(torch.float32)
        log_ratio = log_prob.detach().float() - rollout_log_probs.float()
        seq_is = torch.exp((log_ratio * mask).sum(-1))
        if threshold is not None and float(threshold) > 0:
            seq_is_clipped = torch.clamp(seq_is, max=float(threshold))
        else:
            seq_is_clipped = seq_is
        return {
            "_ess/is_sum": float(seq_is.sum().item()),
            "_ess/is_sq_sum": float((seq_is * seq_is).sum().item()),
            "_ess/is_clipped_sum": float(seq_is_clipped.sum().item()),
            "_ess/is_clipped_sq_sum": float((seq_is_clipped * seq_is_clipped).sum().item()),
            "_ess/count": float(seq_is.shape[0]),
        }


def ess_ratio_from_sums(is_sum: float, is_sq_sum: float, count: float, eps: float = 1e-8) -> tuple[float, float]:
    """Kish effective sample size and its ratio to the sequence count."""
    if count <= 0:
        return 0.0, 0.0
    ess = (is_sum**2) / (is_sq_sum + eps)
    return ess, ess / count


class _EssScaledOptimizerStep(AbstractContextManager):
    """Scales optimizer LRs on entry and restores them on exit (finally
    semantics), emitting the structured ``staleness/ess`` entry.

    The entry is built ONLY from the DP-all-reduced sums plus static config so
    it is identical on every rank — required because worker outputs are
    concatenated as non-tensor data across ranks downstream.
    """

    def __init__(self, engine, data, outputs, ess_config):
        self.engine = engine
        self.ess_config = ess_config
        self.param_groups = None
        self.base_lrs = None

        metrics = outputs.get("metrics", {}) if isinstance(outputs, dict) else outputs["metrics"]
        sums = self._pop_sums(metrics)
        # The all-reduce is a collective: a rank whose micro-batches emitted no
        # sums must still participate (contributing zeros), and the
        # active/inactive decision must be made on the GLOBAL count so that no
        # rank skips the collective while its peers enter it.
        sums = self._all_reduce(sums if sums is not None else [0.0] * len(ESS_SUM_KEYS))
        is_sum, is_sq_sum, clipped_sum, clipped_sq_sum, count = sums
        self.active = count > 0
        if not self.active:
            self.lr_mult = 1.0
            return
        ess, ess_ratio = ess_ratio_from_sums(is_sum, is_sq_sum, count)
        ess_clipped, ess_ratio_clipped = ess_ratio_from_sums(clipped_sum, clipped_sq_sum, count)

        ess_ratio_for_scaling = ess_ratio_clipped if self.ess_config.use_clipped else ess_ratio
        ess_base = resolve_ess_base(self.ess_config.base_ess_ratio, tu.get(data, key="ess_base_override", default=None))

        self.lr_mult = 1.0
        if ess_base is not None:
            lr_scale = compute_ess_lr_scale(ess_ratio_for_scaling, float(ess_base), self.ess_config.trigger_ratio)
            scaling_rule = self.ess_config.scaling_rule
            if scaling_rule == "sqrt":
                self.lr_mult = math.sqrt(lr_scale)
            elif scaling_rule == "linear":
                self.lr_mult = lr_scale
            else:
                raise NotImplementedError(f"{scaling_rule} not implemented for ESS scaling")

        self.param_groups = self.engine.get_optimizer_param_groups()
        self.base_lrs = [float(pg["lr"]) for pg in self.param_groups]
        scaled_lr = self.base_lrs[0] * self.lr_mult if self.base_lrs else None

        entry = {
            "minibatch_idx": int(tu.get(data, key="minibatch_idx", default=0)),
            "minibatch_ess": ess,
            "minibatch_ess_clipped": ess_clipped,
            "minibatch_ess_ratio": ess_ratio,
            "minibatch_ess_ratio_clipped": ess_ratio_clipped,
            "ess_scaled_lr": scaled_lr,
            # The reference actually used for scaling this step (config value or
            # driver override); None while unresolved (scaling is a no-op then).
            "base_ess_ratio": float(ess_base) if ess_base is not None else None,
        }
        metrics["staleness/ess"] = [entry]

    @staticmethod
    def _pop_sums(metrics) -> list[float] | None:
        if ESS_SUM_KEYS[0] not in metrics:
            return None
        sums = []
        for key in ESS_SUM_KEYS:
            # one float per micro-batch (append_to_dict lists); summing is
            # associative so this matches a whole-mini-batch computation
            values = metrics.pop(key)
            if not isinstance(values, list):
                values = [values]
            sums.append(float(sum(float(v) for v in values)))
        return sums

    def _all_reduce(self, sums: list[float]) -> list[float]:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return sums
        group = self.engine.get_data_parallel_group()
        if group is None:
            return sums
        tensor = torch.tensor(sums, dtype=torch.float64)
        if get_torch_device().is_available():
            tensor = tensor.to(get_device_name())
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=group)
        return tensor.tolist()

    def __enter__(self):
        if self.active and self.param_groups is not None and self.lr_mult != 1.0:
            for pg, base_lr in zip(self.param_groups, self.base_lrs, strict=True):
                pg["lr"] = base_lr * self.lr_mult
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.active and self.param_groups is not None:
            for pg, base_lr in zip(self.param_groups, self.base_lrs, strict=True):
                pg["lr"] = base_lr
        return False


def make_ess_optimizer_step_hook(ess_config):
    """Build the pre-optimizer-step hook installed on the actor's engine.

    The returned callable is invoked by ``BaseEngine.train_batch`` as
    ``hook(engine, data, outputs)`` and must return a context manager wrapping
    ``optimizer_step()``.
    """

    def hook(engine, data, outputs):
        return _EssScaledOptimizerStep(engine, data, outputs, ess_config)

    return hook
