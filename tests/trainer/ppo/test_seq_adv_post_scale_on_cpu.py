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

"""CPU tests for the seq_adv_post_scale policy loss (VCPO per-traj semantics):
unit-advantage clip-branch selection, per-sequence advantage post-scaling,
self-anchored PPO ratio, and token-level IS weighting."""

import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_policy_loss_seq_adv_post_scale,
    compute_policy_loss_vanilla,
    get_policy_loss_fn,
)
from verl.workers.config import ActorConfig

RESP_LEN = 6


def _make_config(**overrides) -> ActorConfig:
    kwargs = dict(
        strategy="fsdp2",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        loss_agg_mode="seq-mean-token-mean",
    )
    kwargs.update(overrides)
    return ActorConfig(**kwargs)


def _rows(n=2, seed=0, spread=0.5):
    torch.manual_seed(seed)
    old_lp = torch.randn(n, RESP_LEN) * 0.3
    log_prob = (old_lp + torch.randn(n, RESP_LEN) * spread).requires_grad_(True)
    mask = torch.ones(n, RESP_LEN)
    return old_lp, log_prob, mask


def _adv(scalars):
    return torch.tensor(scalars).unsqueeze(-1).expand(len(scalars), RESP_LEN).clone()


class TestSeqAdvPostScaleLoss:
    def test_registered_in_policy_loss_registry(self):
        assert get_policy_loss_fn("seq_adv_post_scale") is compute_policy_loss_seq_adv_post_scale

    def test_zero_advantages_give_zero_loss_and_grad(self):
        config = _make_config()
        old_lp, log_prob, mask = _rows()
        loss, _ = compute_policy_loss_seq_adv_post_scale(
            old_log_prob=old_lp,
            log_prob=log_prob,
            advantages=_adv([0.0, 0.0]),
            response_mask=mask,
            loss_agg_mode="seq-mean-token-mean",
            config=config,
        )
        assert float(loss) == 0.0
        loss.backward()
        assert torch.all(log_prob.grad == 0)

    def test_self_anchored_ratio_starts_at_negative_mean_advantage(self):
        # ratio == 1 everywhere at update start, no clipping active, no IS
        # weights -> loss value is exactly -mean(adv_scalars)
        config = _make_config()
        old_lp, log_prob, mask = _rows()
        advs = [1.5, -0.5]
        loss, metrics = compute_policy_loss_seq_adv_post_scale(
            old_log_prob=old_lp,
            log_prob=log_prob,
            advantages=_adv(advs),
            response_mask=mask,
            loss_agg_mode="seq-mean-token-mean",
            config=config,
        )
        assert float(loss) == pytest.approx(-sum(advs) / len(advs), rel=1e-6)
        assert metrics["actor/ppo_kl"] == pytest.approx(0.0, abs=1e-8)
        assert metrics["actor/seq_adv_scalar_mean"] == pytest.approx(sum(advs) / len(advs), rel=1e-6)

    def test_scaling_is_linear_in_the_advantage(self):
        # loss(adv = -2) == -2 * loss(adv = +1): clip branch chosen as if A=+1,
        # sign applied afterwards
        config = _make_config()
        old_lp, log_prob, mask = _rows(spread=0.8)

        def loss_for(advs, rollout_is_weights=None):
            loss, _ = compute_policy_loss_seq_adv_post_scale(
                old_log_prob=old_lp,
                log_prob=log_prob,
                advantages=_adv(advs),
                response_mask=mask,
                loss_agg_mode="seq-mean-token-mean",
                config=config,
                rollout_is_weights=rollout_is_weights,
            )
            return loss

        base = loss_for([1.0, 1.0])
        torch.testing.assert_close(loss_for([-2.0, -2.0]), -2.0 * base, rtol=1e-6, atol=1e-8)

    def test_differs_from_vanilla_for_negative_advantages(self):
        # vanilla with in-loss negative advantages picks the other clip branch;
        # feed both losses the SAME self-anchored ratio inputs to isolate the
        # branch-selection difference... with ratio==1 clipping is inert, so
        # use vanilla with a genuinely different old_log_prob anchor instead.
        config = _make_config()
        old_lp, log_prob, mask = _rows(spread=0.8)
        advs = _adv([-2.0, -2.0])
        vanilla_loss, _ = compute_policy_loss_vanilla(
            old_log_prob=old_lp,
            log_prob=log_prob,
            advantages=advs,
            response_mask=mask,
            loss_agg_mode="seq-mean-token-mean",
            config=config,
        )
        parity_loss, _ = compute_policy_loss_seq_adv_post_scale(
            old_log_prob=old_lp,
            log_prob=log_prob,
            advantages=advs,
            response_mask=mask,
            loss_agg_mode="seq-mean-token-mean",
            config=config,
        )
        assert not torch.allclose(vanilla_loss, parity_loss, rtol=1e-3)

    def test_token_is_weights_scale_the_gradient(self):
        config = _make_config()
        _, log_prob, mask = _rows()
        weights = torch.full((2, RESP_LEN), 0.5)
        loss_w, _ = compute_policy_loss_seq_adv_post_scale(
            old_log_prob=None,
            log_prob=log_prob,
            advantages=_adv([1.0, 1.0]),
            response_mask=mask,
            loss_agg_mode="seq-mean-token-mean",
            config=config,
            rollout_is_weights=weights,
        )
        loss_unw, _ = compute_policy_loss_seq_adv_post_scale(
            old_log_prob=None,
            log_prob=log_prob,
            advantages=_adv([1.0, 1.0]),
            response_mask=mask,
            loss_agg_mode="seq-mean-token-mean",
            config=config,
        )
        torch.testing.assert_close(loss_w, 0.5 * loss_unw, rtol=1e-6, atol=1e-9)

    def test_equivalence_with_per_row_unit_advantage_reference(self):
        """Reference semantics from the fork: per sequence, vanilla loss with
        unit advantages on that row alone (seq-mean-token-mean over one row =
        token mean), scaled by the row's advantage, averaged over rows."""
        config = _make_config()
        old_lp, log_prob, mask = _rows(n=3, seed=1, spread=0.8)
        advs = [1.5, -0.5, 0.7]
        weights = torch.rand(3, RESP_LEN) + 0.5

        per_row = []
        unit = torch.ones(1, RESP_LEN)
        for i in range(3):
            row_loss, _ = compute_policy_loss_vanilla(
                old_log_prob=log_prob.detach()[i : i + 1],  # self-anchored
                log_prob=log_prob[i : i + 1],
                advantages=unit,
                response_mask=mask[i : i + 1],
                loss_agg_mode="seq-mean-token-mean",
                config=config,
                rollout_is_weights=weights[i : i + 1],
            )
            per_row.append(advs[i] * row_loss)
        expected = torch.stack(per_row).mean()

        actual, _ = compute_policy_loss_seq_adv_post_scale(
            old_log_prob=old_lp,
            log_prob=log_prob,
            advantages=_adv(advs),
            response_mask=mask,
            loss_agg_mode="seq-mean-token-mean",
            config=config,
            rollout_is_weights=weights,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)

    def test_global_batch_normalization_across_micro_batches(self):
        # with global_batch_size set, micro-batch losses sum to the whole-batch loss
        config = _make_config()
        old_lp, log_prob, mask = _rows(n=4, seed=2)
        advs = [1.0, -1.0, 0.5, 2.0]

        whole, _ = compute_policy_loss_seq_adv_post_scale(
            old_log_prob=old_lp,
            log_prob=log_prob,
            advantages=_adv(advs),
            response_mask=mask,
            loss_agg_mode="seq-mean-token-mean",
            config=config,
        )

        config.global_batch_info["global_batch_size"] = 4
        micro_sum = torch.tensor(0.0)
        for i in range(0, 4, 2):
            part, _ = compute_policy_loss_seq_adv_post_scale(
                old_log_prob=old_lp[i : i + 2],
                log_prob=log_prob[i : i + 2],
                advantages=_adv(advs)[i : i + 2],
                response_mask=mask[i : i + 2],
                loss_agg_mode="seq-mean-token-mean",
                config=config,
            )
            micro_sum = micro_sum + part
        torch.testing.assert_close(micro_sum, whole, rtol=1e-6, atol=1e-8)
