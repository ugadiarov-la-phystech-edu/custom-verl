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
"""Min-ESS learning-rate brake for engine workers (``actor.ess_scaling``).

``ppo_loss`` emits every micro-batch's per-sequence log importance weights under
``ESS_SEQ_LOG_IS_KEY``. After the backward pass and before ``optimizer_step`` the engine calls the hook
built here; it pops those values, computes the mini-batch's global Kish ESS over the engine's ESS
reduction group, and scales every optimizer parameter group's lr by ``compute_min_ess_lr_scale`` for
that one step (restored afterwards, so the LR scheduler keeps its nominal schedule).

The collectives run on every rank of the group whether or not that rank braked or saw any sequences,
so no rank can skip them while its peers enter: the decision is made on the global result.
"""

from contextlib import AbstractContextManager

from verl.workers.utils.ess import compute_global_ess_from_log_weights, compute_min_ess_lr_scale
from verl.workers.utils.losses import ESS_SEQ_LOG_IS_KEY

__all__ = ["EssBrakeStep", "make_ess_optimizer_step_hook"]


def _rollout_is_threshold(actor_config) -> float | None:
    """The rollout-correction IS threshold the clipped ESS is measured against (None: no clipping)."""
    corr = actor_config.policy_loss.get("rollout_correction", None)
    raw = corr.get("rollout_is_threshold", None) if corr is not None else None
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        # string thresholds (e.g. "0.5_5.0" bounds) have no single upper cap
        return None


class EssBrakeStep(AbstractContextManager):
    """Context wrapped around one ``optimizer_step``: scales every param group's lr on entry and
    restores it on exit (also on exceptions). Adds the ESS metrics to ``outputs["metrics"]``."""

    def __init__(self, engine, outputs, actor_config):
        ess_config = actor_config.ess_scaling
        metrics = outputs.get("metrics") if isinstance(outputs, dict) else None
        per_micro_batch = metrics.pop(ESS_SEQ_LOG_IS_KEY, []) if metrics is not None else []
        seq_log_is = [float(v) for mb in per_micro_batch for v in (mb if isinstance(mb, list) else [mb])]

        ess, ess_ratio, ess_clipped, ess_ratio_clipped, count = compute_global_ess_from_log_weights(
            seq_log_is,
            rollout_is_threshold=_rollout_is_threshold(actor_config),
            group=engine.get_ess_reduction_group(),
        )
        measured = ess_clipped if ess_config.use_clipped else ess
        self.lr_mult = compute_min_ess_lr_scale(measured, ess_config.min_ess, ess_config.lr_scale, count=count)

        self.param_groups = engine.get_optimizer_param_groups()
        self.base_lrs = [float(pg["lr"]) for pg in self.param_groups]

        if metrics is not None:
            # one-element lists: the same shape as the per-micro-batch loss metrics, so the worker's
            # DP gather + flatten treats them alike (the values are identical on every rank)
            metrics["staleness/ess"] = [ess]
            metrics["staleness/ess_ratio"] = [ess_ratio]
            metrics["staleness/ess_clipped"] = [ess_clipped]
            metrics["staleness/ess_ratio_clipped"] = [ess_ratio_clipped]
            metrics["staleness/ess_num_seqs"] = [float(count)]
            metrics["actor/ess_lr_mult"] = [self.lr_mult]
            if self.base_lrs:
                metrics["actor/ess_scaled_lr"] = [self.base_lrs[0] * self.lr_mult]

    def __enter__(self):
        if self.lr_mult != 1.0:
            for pg, base_lr in zip(self.param_groups, self.base_lrs, strict=True):
                pg["lr"] = base_lr * self.lr_mult
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for pg, base_lr in zip(self.param_groups, self.base_lrs, strict=True):
            pg["lr"] = base_lr
        return False


def make_ess_optimizer_step_hook(actor_config):
    """Build the ``pre_optimizer_step_hook`` installed on the actor's engine (see BaseEngine.train_batch)."""

    def hook(engine, data, outputs):
        return EssBrakeStep(engine, outputs, actor_config)

    return hook
