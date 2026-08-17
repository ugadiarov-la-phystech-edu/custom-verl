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

"""CPU integration tests for verl/workers/utils/losses.py::ppo_loss with
loss_mode=seq_adv_post_scale + ess_scaling — the full replay-mode loss path:
token-IS REINFORCE gradient, ESS partial sums, guards, and the pearson
diagnostic. Run with: pytest tests/workers/test_losses_ess_integration_on_cpu.py
"""

import math

import pytest
import torch

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.utils import tensordict_utils as tu
from verl.utils.metric import Metric
from verl.workers.config import ActorConfig, ESSScalingConfig, PolicyLossConfig
from verl.workers.utils.ess import make_ess_optimizer_step_hook
from verl.workers.utils.losses import ppo_loss

PROMPT_LEN = 2


def _make_config(
    loss_mode="seq_adv_post_scale",
    ess_enable=True,
    base_ess_ratio=None,
    trigger_ratio=None,
    rollout_is_threshold=2.0,
    log_probs_pearson_corr=False,
    entropy_coeff=0,
    use_kl_loss=False,
):
    return ActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        loss_agg_mode="seq-mean-token-mean",
        entropy_coeff=entropy_coeff,
        use_kl_loss=use_kl_loss,
        policy_loss=PolicyLossConfig(
            loss_mode=loss_mode,
            rollout_correction=RolloutCorrectionConfig(
                rollout_is="token",
                rollout_is_threshold=rollout_is_threshold,
                log_probs_pearson_corr=log_probs_pearson_corr,
            ),
        ),
        ess_scaling=ESSScalingConfig(enable=ess_enable, base_ess_ratio=base_ess_ratio, trigger_ratio=trigger_ratio),
    )


def _make_inputs(
    resp_log_probs,
    rollout_log_probs,
    mask_rows,
    adv_scalars,
    dp_size=1,
    global_batch_size=None,
    include_rollout=True,
    extra_tensors=None,
):
    """Build (model_output, data) for ppo_loss in the padded-tensor layout.

    ``resp_log_probs``: per-row lists of the response-token log-probs of the
    live policy (right-padded rows per ``mask_rows``). The flat model-output
    tensor is laid out so ``no_padding_2_padding`` recovers exactly these
    values; it is a leaf with requires_grad so gradients can be inspected via
    ``model_output["log_probs"].grad`` (use ``_resp_grads`` to slice them).
    """
    n = len(mask_rows)
    resp_len = len(mask_rows[0])
    mask = torch.tensor(mask_rows, dtype=torch.long)
    lens = mask.sum(-1).tolist()

    flat_len = sum(PROMPT_LEN + length for length in lens)
    flat = torch.zeros(flat_len, dtype=torch.float32)
    offset = 0
    for i, length in enumerate(lens):
        offset += PROMPT_LEN + length
        # response log-probs live at [offset - L - 1, offset - 1): left-shifted by one
        for t in range(length):
            flat[offset - length - 1 + t] = float(resp_log_probs[i][t])
    flat.requires_grad_(True)

    rollout = torch.zeros(n, resp_len, dtype=torch.float32)
    for i, length in enumerate(lens):
        for t in range(length):
            rollout[i, t] = float(rollout_log_probs[i][t])

    attention_mask = torch.cat([torch.ones(n, PROMPT_LEN, dtype=torch.long), mask], dim=1)
    adv = torch.tensor(adv_scalars, dtype=torch.float32).unsqueeze(-1) * mask.float()

    tensors = {
        "prompts": torch.zeros(n, PROMPT_LEN, dtype=torch.long),
        "responses": torch.zeros(n, resp_len, dtype=torch.long),
        "attention_mask": attention_mask,
        "response_mask": mask,
        "old_log_probs": rollout.clone(),  # alias semantics of the replay batch
        "advantages": adv,
    }
    if include_rollout:
        tensors["rollout_log_probs"] = rollout
    if extra_tensors:
        tensors.update(extra_tensors)
    data = tu.get_tensordict(
        tensor_dict=tensors,
        non_tensor_dict={
            "dp_size": dp_size,
            "batch_num_tokens": None,
            "global_batch_size": global_batch_size if global_batch_size is not None else n,
        },
    )
    model_output = {"log_probs": flat}
    return model_output, data


def _resp_grads(flat_grad, mask_rows):
    """Slice the per-response-token gradients back out of the flat leaf grad."""
    lens = [sum(row) for row in mask_rows]
    resp_len = len(mask_rows[0])
    out = torch.zeros(len(mask_rows), resp_len)
    offset = 0
    for i, length in enumerate(lens):
        offset += PROMPT_LEN + length
        for t in range(length):
            out[i, t] = flat_grad[offset - length - 1 + t]
    return out


def _metric_value(v):
    return float(v.aggregate()) if isinstance(v, Metric) else float(v)


class TestSeqAdvPostScaleGradient:
    def test_gradient_matches_hand_computed_reinforce(self):
        # grad wrt log_prob[i, t] must be -A_i * w_it / (L_i * global_batch_size)
        # with w_it = min(exp(logp - rollout_logp), threshold)
        mask_rows = [[1, 1, 1], [1, 1, 0]]
        resp_lp = [[-0.5, -1.0, -0.2], [-0.3, -0.9, 0.0]]
        roll_lp = [[-0.5, -2.0, -0.2], [-0.3, -0.1, 0.0]]  # w: [1, e, 1], [1, e^-0.8, -]
        advs = [2.0, -1.0]
        config = _make_config(ess_enable=False)
        model_output, data = _make_inputs(resp_lp, roll_lp, mask_rows, advs, global_batch_size=2)

        loss, _ = ppo_loss(config, model_output, data)
        loss.backward()
        grads = _resp_grads(model_output["log_probs"].grad, mask_rows)

        expected = torch.zeros_like(grads)
        for i, row in enumerate(mask_rows):
            length = sum(row)
            for t in range(length):
                w = min(math.exp(resp_lp[i][t] - roll_lp[i][t]), 2.0)
                expected[i, t] = -advs[i] * w / (length * 2)
        torch.testing.assert_close(grads, expected, rtol=1e-5, atol=1e-7)

    def test_micro_batch_split_gradient_invariance(self):
        # summing the losses of two half micro-batches (same global_batch_size)
        # must reproduce the whole-batch gradients — the dynamic-bsz claim
        mask_rows = [[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]]
        torch.manual_seed(7)
        resp_lp = (torch.randn(4, 3) * 0.5).tolist()
        roll_lp = (torch.randn(4, 3) * 0.5).tolist()
        advs = [1.5, -0.5, 2.0, -2.0]
        config = _make_config(ess_enable=False)

        whole_out, whole_data = _make_inputs(resp_lp, roll_lp, mask_rows, advs, global_batch_size=4)
        loss, _ = ppo_loss(config, whole_out, whole_data)
        loss.backward()
        whole_grads = _resp_grads(whole_out["log_probs"].grad, mask_rows)

        split_grads = []
        for sl in (slice(0, 2), slice(2, 4)):
            out, data = _make_inputs(resp_lp[sl], roll_lp[sl], mask_rows[sl], advs[sl], global_batch_size=4)
            loss, _ = ppo_loss(config, out, data)
            loss.backward()
            split_grads.append(_resp_grads(out["log_probs"].grad, mask_rows[sl]))
        torch.testing.assert_close(torch.cat(split_grads), whole_grads, rtol=1e-5, atol=1e-7)

    def test_provided_rollout_is_weights_pass_through(self):
        # zero provided weights must kill the gradient (no recomputation)
        mask_rows = [[1, 1, 1]]
        config = _make_config(ess_enable=False)
        model_output, data = _make_inputs(
            [[-0.5, -1.0, -0.2]],
            [[-0.4, -0.9, -0.1]],
            mask_rows,
            [1.0],
            extra_tensors={"rollout_is_weights": torch.zeros(1, 3)},
        )
        loss, _ = ppo_loss(config, model_output, data)
        loss.backward()
        assert torch.all(_resp_grads(model_output["log_probs"].grad, mask_rows) == 0)


class TestGuardsAndDiagnostics:
    def test_missing_rollout_log_probs_raises(self):
        config = _make_config()
        model_output, data = _make_inputs([[-0.5, -1.0]], [[-0.5, -1.0]], [[1, 1]], [1.0], include_rollout=False)
        with pytest.raises(ValueError, match="rollout_log_probs"):
            ppo_loss(config, model_output, data)

    def test_guard_rejects_entropy_in_loss(self):
        config = _make_config(entropy_coeff=0.01)
        model_output, data = _make_inputs([[-0.5, -1.0]], [[-0.5, -1.0]], [[1, 1]], [1.0])
        with pytest.raises(NotImplementedError, match="seq_adv_post_scale"):
            ppo_loss(config, model_output, data)

    def test_guard_rejects_kl_loss(self):
        config = _make_config(use_kl_loss=True)
        model_output, data = _make_inputs([[-0.5, -1.0]], [[-0.5, -1.0]], [[1, 1]], [1.0])
        with pytest.raises(NotImplementedError, match="seq_adv_post_scale"):
            ppo_loss(config, model_output, data)

    def test_entropy_logged_with_zero_coeff(self):
        # calculate_entropy=True + entropy_coeff=0: entropy is logged (source
        # parity: actor/entropy) but must not change the loss value
        mask_rows = [[1, 1, 1]]
        resp_lp, roll_lp, advs = [[-0.5, -1.0, -0.2]], [[-0.4, -0.9, -0.1]], [1.0]
        config = _make_config(ess_enable=False)
        out_plain, data_plain = _make_inputs(resp_lp, roll_lp, mask_rows, advs)
        loss_plain, _ = ppo_loss(config, out_plain, data_plain)

        out, data = _make_inputs(resp_lp, roll_lp, mask_rows, advs)
        out["entropy"] = torch.full_like(out["log_probs"].detach(), 1.3)
        loss, metrics = ppo_loss(config, out, data)
        assert "actor/entropy" in metrics and "actor/entropy_loss" in metrics
        assert _metric_value(metrics["actor/entropy"]) == pytest.approx(1.3, rel=1e-6)
        assert float(loss) == pytest.approx(float(loss_plain), rel=1e-6)

    def test_pearson_metric_emitted_when_enabled(self):
        mask_rows = [[1, 1, 1], [1, 1, 1]]
        torch.manual_seed(3)
        resp_lp = (torch.randn(2, 3) * 0.7).tolist()
        # rollout = live + const  =>  exp() perfectly correlated  =>  corr == 1
        roll_lp = [[v - 0.2 for v in row] for row in resp_lp]
        config = _make_config(log_probs_pearson_corr=True)
        model_output, data = _make_inputs(resp_lp, roll_lp, mask_rows, [1.0, -1.0])
        _, metrics = ppo_loss(config, model_output, data)
        assert metrics["training/rollout_actor_probs_pearson_corr"] == pytest.approx(1.0, abs=1e-4)

    def test_pearson_metric_absent_when_disabled(self):
        config = _make_config(log_probs_pearson_corr=False)
        model_output, data = _make_inputs([[-0.5, -1.0]], [[-0.4, -0.9]], [[1, 1]], [1.0])
        _, metrics = ppo_loss(config, model_output, data)
        assert "training/rollout_actor_probs_pearson_corr" not in metrics


class TestEssIntegration:
    def test_ess_sums_emitted_and_consumed_by_hook(self):
        # engineered sequence weights: w_seq = exp(sum(logp - rollout))
        mask_rows = [[1, 1], [1, 1]]
        resp_lp = [[0.0, 0.0], [0.0, 0.0]]
        roll_lp = [[-math.log(8.0) / 2] * 2, [math.log(8.0) / 2] * 2]  # w = [8, 1/8]
        config = _make_config(ess_enable=True, base_ess_ratio=1.0)
        model_output, data = _make_inputs(resp_lp, roll_lp, mask_rows, [1.0, 1.0])
        _, metrics = ppo_loss(config, model_output, data)

        assert metrics["_ess/is_sum"] == pytest.approx(8.0 + 0.125, rel=1e-4)
        assert metrics["_ess/is_clipped_sum"] == pytest.approx(2.0 + 0.125, rel=1e-4)
        assert metrics["_ess/count"] == 2.0

        # the optimizer-step hook consumes the sums and brakes the LR
        import torch.nn as nn

        params = [nn.Parameter(torch.zeros(1))]
        optimizer = torch.optim.SGD([{"params": params, "lr": 0.1}])
        engine = type(
            "E",
            (),
            {
                "get_optimizer_param_groups": lambda self: optimizer.param_groups,
                "get_data_parallel_group": lambda self: None,
            },
        )()
        outputs = {"metrics": {k: v for k, v in metrics.items() if k.startswith("_ess/")}}
        hook = make_ess_optimizer_step_hook(config.ess_scaling)
        expected_ratio = (8.125**2) / (64.0 + 0.125**2) / 2.0
        with hook(engine, tu.get_tensordict(tensor_dict={}, non_tensor_dict={}), outputs):
            assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * math.sqrt(expected_ratio), rel=1e-4)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
        (entry,) = outputs["metrics"]["staleness/ess"]
        assert entry["minibatch_ess_ratio"] == pytest.approx(expected_ratio, rel=1e-4)

    def test_no_sums_when_ess_disabled(self):
        config = _make_config(ess_enable=False)
        model_output, data = _make_inputs([[-0.5, -1.0]], [[-0.4, -0.9]], [[1, 1]], [1.0])
        _, metrics = ppo_loss(config, model_output, data)
        assert not any(k.startswith("_ess/") for k in metrics)
