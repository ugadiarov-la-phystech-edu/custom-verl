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
"""ppo_loss for the replay-buffer ESS recipe.

1. The ESS inputs: per-sequence log-IS sums (and the Pearson diagnostic) emitted per micro-batch, and a
   clear error when rollout_log_probs are missing.
2. Objective equivalence with the custom_vcpo source. There, ``skip_recompute_old_log_prob`` trained each
   trajectory as its own micro-batch with ``old = log_pi.detach()`` (PPO ratio == 1), token-TIS weights
   ``min(exp(log_pi - log_mu), C)``, advantage 1 and a loss multiplier ``adv_i / len(local mini-batch)``,
   then averaged gradients over DP. The engine path runs ``loss_mode=bypass_mode`` + ``loss_type=reinforce``
   + ``rollout_is=token`` with ``seq-mean-token-mean`` over arbitrary micro-batches and DP ranks; the
   gradient must be the same.
"""

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.utils import tensordict_utils as tu
from verl.utils.debug.metrics import rollout_actor_probs_pearson_corr
from verl.workers.config import ActorConfig, ESSScalingConfig, PolicyLossConfig
from verl.workers.utils.losses import ESS_SEQ_LOG_IS_KEY, ppo_loss

PROMPT_LEN = 3
RESP_LEN = 6
THRESHOLD = 2.0
LENGTHS = [6, 2, 4, 1, 5, 3, 6, 2]


def _rollout_correction(**extra):
    return {
        "rollout_is": "token",
        "rollout_is_threshold": THRESHOLD,
        "loss_type": "reinforce",
        "bypass_mode": True,
        **extra,
    }


def _engine_config(ess=False, pearson=False, loss_mode="bypass_mode", loss_agg_mode="seq-mean-token-mean"):
    return ActorConfig(
        strategy="megatron",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        loss_agg_mode=loss_agg_mode,
        clip_ratio_c=3.0,
        ess_scaling=ESSScalingConfig(enable=ess),
        policy_loss=PolicyLossConfig(
            loss_mode=loss_mode, rollout_correction=_rollout_correction(log_probs_pearson_corr=pearson)
        ),
    )


def _make_episode(lengths=LENGTHS, seed=0):
    """Per-sequence response log-probs of the current policy, rollout log-probs, and per-sequence advantages.

    Rollout log-probs are offset so that some token IS ratios exceed the TIS cap (truncation engages)."""
    g = torch.Generator().manual_seed(seed)
    bsz = len(lengths)
    mask = torch.zeros(bsz, RESP_LEN)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1.0
    policy = -torch.rand(bsz, RESP_LEN, generator=g, dtype=torch.float64) * 3
    rollout = policy - torch.randn(bsz, RESP_LEN, generator=g, dtype=torch.float64) * 0.8
    adv = torch.randn(bsz, generator=g, dtype=torch.float64)
    assert ((policy - rollout).exp() * mask > THRESHOLD).any(), "the TIS cap must engage in this fixture"
    return policy, rollout, adv, mask


def _batch(idx, policy_leaf, rollout, adv, mask, *, dp_size, global_bsz, with_rollout=True):
    """(model_output, data) for the sequences ``idx``, in the padded layout ppo_loss accepts, with the flat
    (total_nnz,) log-prob output of a remove-padding forward that no_padding_2_padding slices."""
    idx = list(idx)
    m = mask[idx]
    lens = m.sum(-1).long()
    attn = torch.cat([torch.ones(len(idx), PROMPT_LEN, dtype=torch.long), m.long()], dim=1)
    pieces = []
    for row, n in zip(idx, lens.tolist(), strict=True):
        # model output at position p predicts token p+1: the response log-probs sit at [P-1, P-1+n)
        pieces += [torch.zeros(PROMPT_LEN - 1, dtype=policy_leaf.dtype), policy_leaf[row, :n], policy_leaf.new_zeros(1)]
    flat = torch.cat(pieces)
    fields = {
        "prompts": torch.ones(len(idx), PROMPT_LEN, dtype=torch.long),
        "responses": torch.ones(len(idx), RESP_LEN, dtype=torch.long),
        "attention_mask": attn,
        "response_mask": m,
        # bypass mode: the trainer sets old_log_probs = rollout_log_probs
        "old_log_probs": rollout[idx],
        "advantages": adv[idx, None] * m,
    }
    if with_rollout:
        fields["rollout_log_probs"] = rollout[idx]
    data = TensorDict(fields, batch_size=[len(idx)])
    tu.assign_non_tensor(data, dp_size=dp_size, batch_num_tokens=int(mask.sum()), global_batch_size=global_bsz)
    return {"log_probs": flat}, data


def _engine_grad(policy, rollout, adv, mask, *, dp_size, micro_batches_per_rank):
    """Gradient the engine path accumulates: per rank, sum of micro-batch ppo_loss backward; DDP then means
    over ranks."""
    bsz = mask.shape[0]
    leaf = policy.clone().requires_grad_(True)
    per_rank = bsz // dp_size
    rank_grads = []
    for r in range(dp_size):
        leaf.grad = None
        rows = list(range(r * per_rank, (r + 1) * per_rank))
        step = per_rank // micro_batches_per_rank
        for s in range(0, per_rank, step):
            model_output, data = _batch(rows[s : s + step], leaf, rollout, adv, mask, dp_size=dp_size, global_bsz=bsz)
            loss, _ = ppo_loss(_engine_config(), model_output, data)
            loss.backward()
        rank_grads.append(leaf.grad.clone())
    return torch.stack(rank_grads).mean(0)


def _source_grad(policy, rollout, adv, mask, *, dp_size):
    """custom_vcpo's per-trajectory skip-recompute update, written out from megatron_actor.py."""
    bsz = mask.shape[0]
    leaf = policy.clone().requires_grad_(True)
    per_rank = bsz // dp_size
    cfg = ActorConfig(
        strategy="megatron",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio=0.2,
        clip_ratio_c=3.0,
        loss_agg_mode="seq-mean-token-mean",
    )
    rank_grads = []
    for r in range(dp_size):
        leaf.grad = None
        for i in range(r * per_rank, (r + 1) * per_rank):
            n = int(mask[i].sum())
            log_prob = leaf[i : i + 1, :n]
            old = log_prob.detach().clone()
            tis = (old - rollout[i : i + 1, :n]).exp().clamp(max=THRESHOLD)
            pg_loss, _ = compute_policy_loss_vanilla(
                old_log_prob=old,
                log_prob=log_prob,
                advantages=torch.ones_like(log_prob),
                response_mask=torch.ones_like(log_prob),
                loss_agg_mode="seq-mean-token-mean",
                config=cfg,
                rollout_is_weights=tis,
            )
            (pg_loss * adv[i] / per_rank).backward()  # loss_multiplier = adv_i / len(minibatch)
        rank_grads.append(leaf.grad.clone())
    return torch.stack(rank_grads).mean(0)


class TestObjectiveEquivalence:
    @pytest.mark.parametrize("dp_size", [1, 2, 4])
    @pytest.mark.parametrize("micro_batches_per_rank", [1, 2])
    def test_engine_gradient_matches_source(self, dp_size, micro_batches_per_rank):
        policy, rollout, adv, mask = _make_episode()
        if len(LENGTHS) // dp_size < micro_batches_per_rank:
            pytest.skip("fewer sequences per rank than micro-batches")
        engine = _engine_grad(
            policy, rollout, adv, mask, dp_size=dp_size, micro_batches_per_rank=micro_batches_per_rank
        )
        source = _source_grad(policy, rollout, adv, mask, dp_size=dp_size)
        assert engine.abs().sum() > 0
        # rtol covers the 1e-8 epsilon in the seq-mean-token-mean denominator (relative 1e-8 on a 1-token sequence)
        torch.testing.assert_close(engine, source, rtol=1e-7, atol=1e-12)

    def test_micro_batch_split_invariance(self):
        policy, rollout, adv, mask = _make_episode(seed=3)
        grads = [_engine_grad(policy, rollout, adv, mask, dp_size=1, micro_batches_per_rank=k) for k in (1, 2, 4, 8)]
        for g in grads[1:]:
            torch.testing.assert_close(g, grads[0], rtol=1e-9, atol=1e-12)

    def test_padding_tokens_get_no_gradient(self):
        policy, rollout, adv, mask = _make_episode()
        grad = _engine_grad(policy, rollout, adv, mask, dp_size=2, micro_batches_per_rank=2)
        assert (grad[mask == 0] == 0).all()

    def test_differs_without_tis(self):
        # guards the fixture: with the cap engaged, dropping TIS changes the gradient, so the equality above
        # really exercises the token-TIS weights
        policy, rollout, adv, mask = _make_episode()
        source = _source_grad(policy, rollout, adv, mask, dp_size=1)
        naive = _source_grad(policy, policy, adv, mask, dp_size=1)  # rollout == policy: all weights 1
        assert not torch.allclose(source, naive)

    def test_token_mean_is_not_equivalent(self):
        # the per-trajectory multiplier adv_i/len(mb) with a per-sequence token-mean is seq-mean-token-mean;
        # a global token-mean weights long sequences more
        policy, rollout, adv, mask = _make_episode()
        source = _source_grad(policy, rollout, adv, mask, dp_size=1)
        leaf = policy.clone().requires_grad_(True)
        model_output, data = _batch(range(len(LENGTHS)), leaf, rollout, adv, mask, dp_size=1, global_bsz=len(LENGTHS))
        loss, _ = ppo_loss(_engine_config(loss_agg_mode="token-mean"), model_output, data)
        loss.backward()
        assert not torch.allclose(leaf.grad, source)


class TestEssInputs:
    def _call(self, config, rows=(0, 1, 2), with_rollout=True, policy_offset=0.0):
        policy, rollout, adv, mask = _make_episode()
        leaf = (policy + policy_offset).requires_grad_(True)
        model_output, data = _batch(
            rows, leaf, rollout, adv, mask, dp_size=1, global_bsz=len(LENGTHS), with_rollout=with_rollout
        )
        _, metrics = ppo_loss(config, model_output, data)
        return metrics, leaf.detach(), rollout, mask

    def test_emits_per_sequence_log_is_sums(self):
        rows = (0, 1, 3)
        metrics, policy, rollout, mask = self._call(_engine_config(ess=True), rows=rows)
        expected = [float(((policy[i] - rollout[i]) * mask[i]).sum()) for i in rows]
        assert metrics[ESS_SEQ_LOG_IS_KEY] == pytest.approx(expected, rel=1e-12)
        assert isinstance(metrics[ESS_SEQ_LOG_IS_KEY], list)

    def test_sums_ignore_padding(self):
        # a huge gap on padding positions must not leak into the sums
        policy, rollout, adv, mask = _make_episode()
        rollout = torch.where(mask.bool(), rollout, torch.full_like(rollout, -1e6))
        leaf = policy.clone().requires_grad_(True)
        model_output, data = _batch((1, 3), leaf, rollout, adv, mask, dp_size=1, global_bsz=len(LENGTHS))
        _, metrics = ppo_loss(_engine_config(ess=True), model_output, data)
        assert max(abs(v) for v in metrics[ESS_SEQ_LOG_IS_KEY]) < 100

    def test_disabled_emits_nothing_and_needs_no_rollout_log_probs(self):
        metrics, *_ = self._call(_engine_config(ess=False), with_rollout=False)
        assert ESS_SEQ_LOG_IS_KEY not in metrics
        assert "training/rollout_actor_probs_pearson_corr" not in metrics

    @pytest.mark.parametrize("ess, pearson", [(True, False), (False, True), (True, True)])
    def test_missing_rollout_log_probs_is_an_error(self, ess, pearson):
        with pytest.raises(ValueError, match="rollout_log_probs"):
            self._call(_engine_config(ess=ess, pearson=pearson), with_rollout=False)

    def test_pearson_metric(self):
        metrics, policy, rollout, mask = self._call(_engine_config(pearson=True), rows=(0, 2, 4))
        rows = [0, 2, 4]
        expected = rollout_actor_probs_pearson_corr(policy[rows], rollout[rows], mask[rows].bool())
        assert metrics["training/rollout_actor_probs_pearson_corr"] == pytest.approx(expected)
        assert ESS_SEQ_LOG_IS_KEY not in metrics

    def test_sums_do_not_depend_on_loss_mode(self):
        a, *_ = self._call(_engine_config(ess=True))
        b, *_ = self._call(_engine_config(ess=True, loss_mode="vanilla"))
        assert a[ESS_SEQ_LOG_IS_KEY] == pytest.approx(b[ESS_SEQ_LOG_IS_KEY])

    def test_loss_value_unchanged_by_ess(self):
        policy, rollout, adv, mask = _make_episode()
        losses = []
        for ess in (False, True):
            leaf = policy.clone().requires_grad_(True)
            model_output, data = _batch(range(4), leaf, rollout, adv, mask, dp_size=1, global_bsz=len(LENGTHS))
            loss, _ = ppo_loss(_engine_config(ess=ess), model_output, data)
            losses.append(loss.item())
        assert losses[0] == losses[1]
