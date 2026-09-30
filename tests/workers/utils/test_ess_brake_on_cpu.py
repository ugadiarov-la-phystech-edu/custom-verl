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
"""The ESS LR brake around one optimizer step (verl/workers/utils/ess_brake.py) and its wiring into
BaseEngine.train_batch: lr scaled on every param group only for that step, restored afterwards (also on
errors), failing closed on a broken measurement, and every rank of the reduction group agreeing."""

import math
import os
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.workers.config import ActorConfig, ESSScalingConfig, PolicyLossConfig
from verl.workers.engine.base import BaseEngine
from verl.workers.utils.ess import ess_from_log_weights
from verl.workers.utils.ess_brake import EssBrakeStep, _rollout_is_threshold, make_ess_optimizer_step_hook
from verl.workers.utils.losses import ESS_SEQ_LOG_IS_KEY

BASE_LRS = (1e-6, 3e-6)
DEGENERATE = [0.0, -50.0, -60.0]  # one sequence carries all the weight: ESS = 1
HEALTHY = [0.0, 0.0, 0.0, 0.0]  # uniform weights: ESS = 4


def _actor_config(threshold=2.0, rollout_correction="default", **ess):
    ess = {"enable": True, **ess}
    if rollout_correction == "default":
        rollout_correction = {"rollout_is": "token", "rollout_is_threshold": threshold, "loss_type": "reinforce"}
    return ActorConfig(
        strategy="megatron",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        ess_scaling=ESSScalingConfig(**ess),
        policy_loss=PolicyLossConfig(loss_mode="bypass_mode", rollout_correction=rollout_correction),
    )


class _FakeEngine:
    """Only what EssBrakeStep uses: the optimizer param groups and the reduction group."""

    def __init__(self, lrs=BASE_LRS, group=None):
        params = [torch.nn.Parameter(torch.zeros(1)) for _ in lrs]
        groups = [{"params": [p], "lr": lr} for p, lr in zip(params, lrs, strict=True)]
        self.optimizer = torch.optim.SGD(groups) if groups else SimpleNamespace(param_groups=[])
        self.group = group
        self.group_calls = 0

    def get_optimizer_param_groups(self):
        return self.optimizer.param_groups

    def get_ess_reduction_group(self):
        self.group_calls += 1
        return self.group


def _lrs(engine):
    return tuple(pg["lr"] for pg in engine.optimizer.param_groups)


def _outputs(*micro_batches):
    # the aggregated shape postprocess_batch_func produces (append_to_dict extends list values)
    return {"metrics": {ESS_SEQ_LOG_IS_KEY: [v for mb in micro_batches for v in mb], "actor/pg_loss": [0.1]}}


class TestEssBrakeStep:
    def test_degenerate_batch_brakes_every_group_for_one_step(self):
        engine = _FakeEngine()
        step = EssBrakeStep(engine, _outputs(DEGENERATE), _actor_config())
        assert step.lr_mult == 0.5
        assert _lrs(engine) == BASE_LRS  # nothing changes before the step
        with step:
            assert _lrs(engine) == pytest.approx(tuple(0.5 * lr for lr in BASE_LRS))
        assert _lrs(engine) == BASE_LRS

    def test_healthy_batch_steps_at_full_lr(self):
        engine = _FakeEngine()
        step = EssBrakeStep(engine, _outputs(HEALTHY), _actor_config())
        assert step.lr_mult == 1.0
        with step:
            assert _lrs(engine) == BASE_LRS
        assert _lrs(engine) == BASE_LRS

    @pytest.mark.parametrize("lr_scale", [0.25, 0.5, 1.0])
    def test_lr_scale_is_the_multiplier(self, lr_scale):
        engine = _FakeEngine()
        with EssBrakeStep(engine, _outputs(DEGENERATE), _actor_config(lr_scale=lr_scale)):
            assert _lrs(engine) == pytest.approx(tuple(lr_scale * lr for lr in BASE_LRS))

    def test_threshold_is_inclusive(self):
        # two equal sequences: ESS == 2 exactly
        engine = _FakeEngine()
        assert EssBrakeStep(engine, _outputs([0.0, 0.0]), _actor_config(min_ess=2.0)).lr_mult == 0.5
        assert EssBrakeStep(engine, _outputs([0.0, 0.0, 0.0]), _actor_config(min_ess=2.0)).lr_mult == 1.0

    def test_lr_restored_when_the_step_raises(self):
        engine = _FakeEngine()
        with pytest.raises(RuntimeError, match="boom"):
            with EssBrakeStep(engine, _outputs(DEGENERATE), _actor_config()):
                raise RuntimeError("boom")
        assert _lrs(engine) == BASE_LRS

    def test_restores_values_the_step_overwrote(self):
        # Megatron's optimizer or a scheduler may write lr inside the step; the brake still restores the
        # base values it recorded, so the next step starts from the nominal schedule
        engine = _FakeEngine()
        with EssBrakeStep(engine, _outputs(DEGENERATE), _actor_config()):
            for pg in engine.optimizer.param_groups:
                pg["lr"] = 123.0
        assert _lrs(engine) == BASE_LRS

    def test_sums_split_across_micro_batches_are_pooled(self):
        engine = _FakeEngine()
        pooled = EssBrakeStep(engine, _outputs([0.0, -50.0, -60.0, 1.0]), _actor_config(min_ess=1.9))
        split = EssBrakeStep(engine, _outputs([0.0, -50.0], [-60.0], [1.0]), _actor_config(min_ess=1.9))
        assert pooled.lr_mult == split.lr_mult

    def test_nested_micro_batch_lists_are_flattened(self):
        engine = _FakeEngine()
        outputs = {"metrics": {ESS_SEQ_LOG_IS_KEY: [[0.0, 0.0], [0.0], 0.0]}}
        EssBrakeStep(engine, outputs, _actor_config())
        assert outputs["metrics"]["staleness/ess_num_seqs"] == [4.0]
        assert outputs["metrics"]["staleness/ess"] == [pytest.approx(4.0)]

    def test_metrics(self):
        engine = _FakeEngine()
        outputs = _outputs([0.0, -1.0, 3.0])
        EssBrakeStep(engine, outputs, _actor_config(min_ess=3.0))
        m = outputs["metrics"]
        assert ESS_SEQ_LOG_IS_KEY not in m  # popped: never reaches the DP metric gather
        ess, ratio, ess_c, ratio_c, _ = ess_from_log_weights([0.0, -1.0, 3.0], rollout_is_threshold=2.0)
        assert m["staleness/ess"] == [pytest.approx(ess)]
        assert m["staleness/ess_ratio"] == [pytest.approx(ratio)]
        assert m["staleness/ess_clipped"] == [pytest.approx(ess_c)]
        assert m["staleness/ess_ratio_clipped"] == [pytest.approx(ratio_c)]
        assert m["staleness/ess_num_seqs"] == [3.0]
        assert m["actor/ess_lr_mult"] == [0.5]
        assert m["actor/ess_scaled_lr"] == [pytest.approx(0.5 * BASE_LRS[0])]
        assert m["actor/pg_loss"] == [0.1]  # other metrics untouched

    def test_no_sequences_means_no_scaling(self):
        engine = _FakeEngine()
        outputs = {"metrics": {}}
        step = EssBrakeStep(engine, outputs, _actor_config(min_ess=64.0))
        assert step.lr_mult == 1.0
        assert outputs["metrics"]["staleness/ess_num_seqs"] == [0.0]
        assert engine.group_calls == 1  # the collective is still entered

    def test_outputs_without_metrics_still_measure(self):
        # a rank without outputs must still join the reduction (a skipped collective would hang its peers)
        engine = _FakeEngine()
        step = EssBrakeStep(engine, {}, _actor_config())
        assert engine.group_calls == 1
        assert step.lr_mult == 1.0

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_broken_measurement_fails_closed(self, bad):
        engine = _FakeEngine()
        step = EssBrakeStep(engine, _outputs([0.0, bad, 0.0]), _actor_config(min_ess=1.0))
        assert step.lr_mult == 0.5

    def test_use_clipped_brakes_on_the_clipped_ess(self):
        # weights e^5 and 1: unclipped ESS ~1.013 (brakes at min_ess 1.1); clipped at 2 -> weights 2, 1, ESS 1.8
        engine = _FakeEngine()
        assert EssBrakeStep(engine, _outputs([5.0, 0.0]), _actor_config(use_clipped=False)).lr_mult == 0.5
        assert EssBrakeStep(engine, _outputs([5.0, 0.0]), _actor_config(use_clipped=True)).lr_mult == 1.0

    def test_empty_optimizer_param_groups(self):
        engine = _FakeEngine(lrs=())
        outputs = _outputs(DEGENERATE)
        with EssBrakeStep(engine, outputs, _actor_config()):
            pass
        assert "actor/ess_scaled_lr" not in outputs["metrics"]

    def test_hook_factory(self):
        engine = _FakeEngine()
        hook = make_ess_optimizer_step_hook(_actor_config())
        step = hook(engine, None, _outputs(DEGENERATE))
        assert isinstance(step, EssBrakeStep)
        assert step.lr_mult == 0.5


class TestRolloutIsThreshold:
    @pytest.mark.parametrize(
        "corr, expected",
        [
            ({"rollout_is_threshold": 2.0}, 2.0),
            ({"rollout_is_threshold": "3.5"}, 3.5),
            ({"rollout_is_threshold": "0.5_5.0"}, None),  # lower_upper bounds: no single cap
            ({"rollout_is_threshold": None}, None),
            ({}, None),
            (None, None),
        ],
    )
    def test_parse(self, corr, expected):
        assert _rollout_is_threshold(_actor_config(rollout_correction=corr)) == expected


class _StubEngine(BaseEngine):
    """BaseEngine.train_batch with the model work replaced by a recorder."""

    def __init__(self):
        self.events = []
        self.optimizer = _FakeEngine().optimizer

    def optimizer_zero_grad(self):
        self.events.append("zero_grad")

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        self.events.append("forward_backward")
        return _outputs(DEGENERATE)

    def optimizer_step(self):
        self.events.append(("step", _lrs(self)))
        return 1.5

    def is_mp_src_rank_with_outputs(self):
        return True

    def get_data_parallel_group(self):
        return None


def _batch():
    return TensorDict({"input_ids": torch.zeros(2, 3, dtype=torch.long)}, batch_size=[2])


class TestTrainBatchHook:
    def test_without_hook(self):
        engine = _StubEngine()
        out = engine.train_batch(data=_batch(), loss_function=None)
        assert engine.events == ["zero_grad", "forward_backward", ("step", BASE_LRS)]
        assert out["metrics"]["grad_norm"] == 1.5
        assert ESS_SEQ_LOG_IS_KEY in out["metrics"]  # untouched without the brake

    def test_hook_runs_after_backward_and_wraps_the_step(self):
        engine = _StubEngine()
        seen = {}

        def hook(eng, data, outputs):
            seen["called_after"] = list(eng.events)
            seen["data"] = data
            return EssBrakeStep(eng, outputs, _actor_config())

        batch = _batch()
        out = engine.train_batch(data=batch, loss_function=None, pre_optimizer_step_hook=hook)
        assert seen["called_after"] == ["zero_grad", "forward_backward"]
        assert seen["data"] is batch
        assert engine.events[-1] == ("step", pytest.approx(tuple(0.5 * lr for lr in BASE_LRS)))
        assert _lrs(engine) == BASE_LRS  # restored after the step
        assert out["metrics"]["grad_norm"] == 1.5
        assert out["metrics"]["actor/ess_lr_mult"] == [0.5]

    def test_real_ess_hook(self):
        engine = _StubEngine()
        engine.train_batch(_batch(), None, pre_optimizer_step_hook=make_ess_optimizer_step_hook(_actor_config()))
        assert engine.events[-1] == ("step", pytest.approx(tuple(0.5 * lr for lr in BASE_LRS)))

    def test_default_reduction_group_is_dp(self):
        engine = _StubEngine()
        assert engine.get_ess_reduction_group() is None
        assert engine.get_optimizer_param_groups() is engine.optimizer.param_groups


def _two_rank_brake_worker(rank, world_size, init_file, per_rank_logs, min_ess, out_dir):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        engine = _FakeEngine(group=dist.group.WORLD)
        outputs = _outputs(per_rank_logs[rank]) if per_rank_logs[rank] is not None else {}
        step = EssBrakeStep(engine, outputs, _actor_config(min_ess=min_ess))
        with step:
            lrs = _lrs(engine)
        metrics = outputs.get("metrics", {})
        result = (step.lr_mult, lrs, metrics.get("staleness/ess"), metrics.get("staleness/ess_num_seqs"))
        with open(os.path.join(out_dir, f"rank{rank}.txt"), "w") as f:
            f.write(repr(result))
    finally:
        dist.destroy_process_group()


class TestTwoRanks:
    """Real 2-process gloo run: the brake decision is made on the global ESS, so every rank scales its lr the
    same way, including a rank that saw no sequences (it still enters the collectives)."""

    @pytest.mark.parametrize(
        "per_rank_logs, min_ess, expected_mult",
        [
            ([[0.0, 0.0], [0.0, 0.0]], 3.0, 1.0),  # global ESS 4 > 3 although each rank alone has 2
            ([[0.0, -50.0], [-60.0, -70.0]], 1.1, 0.5),  # rank 0 dominates globally
            ([[0.0, 0.0, 0.0], []], 2.5, 1.0),  # rank 1 has no sequences but joins
            ([[0.0, -40.0], None], 1.1, 0.5),  # rank 1 has no metrics at all
        ],
    )
    def test_ranks_agree(self, tmp_path, per_rank_logs, min_ess, expected_mult):
        import torch.multiprocessing as mp

        mp.spawn(
            _two_rank_brake_worker,
            args=(2, str(tmp_path / "init"), per_rank_logs, min_ess, str(tmp_path)),
            nprocs=2,
            join=True,
        )
        results = [eval((tmp_path / f"rank{r}.txt").read_text()) for r in range(2)]  # noqa: S307 - our own repr
        pooled = [x for logs in per_rank_logs if logs for x in logs]
        ref_ess = ess_from_log_weights(pooled)[0]
        for rank, (mult, lrs, ess, count) in enumerate(results):
            assert mult == expected_mult
            assert lrs == pytest.approx(tuple(expected_mult * lr for lr in BASE_LRS))
            if per_rank_logs[rank] is not None:
                assert ess == [pytest.approx(ref_ess)]
                assert count == [float(len(pooled))]
