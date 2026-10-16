# Copyright 2026 custom-verl authors
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
"""ppo_loss with the replay arm's DECOUPLED objective (the sync PPO recipe).

The trainer recomputes pi_old once per update and adds the token TIS weights w = min(pi_old / pi_rollout, C); every
optimizer mini-batch then runs policy_loss.loss_mode=vanilla, i.e. PPO-clip on r = pi_theta / pi_old with
clip_ratio_low / clip_ratio_high and the dual clip c, times w.

1. First mini-batch of an update (pi_theta == pi_old): the gradient equals token-TIS REINFORCE with the same w, so the
   decoupled objective with ONE mini-batch per update cannot differ from the bypass arm.
2. Later mini-batches (pi_theta != pi_old): clip-higher is asymmetric (A > 0 loses its gradient above 1 + 0.28, A < 0
   below 1 - 0.2) and the dual clip c = 10 caps A < 0 tokens with r > c.
3. The ESS brake inputs and the Pearson diagnostic (pi_theta vs pi_rollout) are still emitted.
"""

import math

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig, ESSScalingConfig, PolicyLossConfig
from verl.workers.utils.losses import ESS_SEQ_LOG_IS_KEY, ppo_loss

PROMPT_LEN = 3
RESP_LEN = 6
CAP = 8.0


def _config(ess=True, pearson=True, low=0.2, high=0.28, c=10.0):
    return ActorConfig(
        strategy="megatron",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        loss_agg_mode="token-mean",
        clip_ratio=low,
        clip_ratio_low=low,
        clip_ratio_high=high,
        clip_ratio_c=c,
        ess_scaling=ESSScalingConfig(enable=ess),
        policy_loss=PolicyLossConfig(
            loss_mode="vanilla",
            rollout_correction={
                "rollout_is": "token",
                "rollout_is_threshold": CAP,
                "bypass_mode": False,
                "loss_type": "reinforce",
                "log_probs_pearson_corr": pearson,
            },
        ),
    )


def _call(config, policy_leaf, old, rollout, adv, mask):
    """ppo_loss on one padded micro-batch = the whole (global) mini-batch, dp=1."""
    bsz = mask.shape[0]
    lens = mask.sum(-1).long()
    attn = torch.cat([torch.ones(bsz, PROMPT_LEN, dtype=torch.long), mask.long()], dim=1)
    pieces = []
    for row, n in enumerate(lens.tolist()):
        # model output at position p predicts token p+1: the response log-probs sit at [P-1, P-1+n)
        pieces += [torch.zeros(PROMPT_LEN - 1, dtype=policy_leaf.dtype), policy_leaf[row, :n], policy_leaf.new_zeros(1)]
    weights = (old - rollout).exp().clamp(max=CAP) * mask
    data = TensorDict(
        {
            "prompts": torch.ones(bsz, PROMPT_LEN, dtype=torch.long),
            "responses": torch.ones(bsz, RESP_LEN, dtype=torch.long),
            "attention_mask": attn,
            "response_mask": mask,
            "old_log_probs": old,
            "rollout_log_probs": rollout,
            "rollout_is_weights": weights,
            "advantages": adv[:, None] * mask,
        },
        batch_size=[bsz],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=int(mask.sum()), global_batch_size=bsz)
    return ppo_loss(config, {"log_probs": torch.cat(pieces)}, data), weights


def _episode(seed=0):
    g = torch.Generator().manual_seed(seed)
    lengths = [6, 2, 4, 1, 5, 3]
    mask = torch.zeros(len(lengths), RESP_LEN, dtype=torch.float64)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1.0
    policy = -torch.rand(len(lengths), RESP_LEN, generator=g, dtype=torch.float64) * 3
    rollout = policy - torch.randn(len(lengths), RESP_LEN, generator=g, dtype=torch.float64) * 1.0
    rollout[0, 0] = policy[0, 0] - 3.0  # one token far above the TIS cap (ratio e^3 ~ 20)
    adv = torch.randn(len(lengths), generator=g, dtype=torch.float64)
    return policy, rollout, adv, mask


def test_first_minibatch_equals_token_tis_reinforce():
    policy, rollout, adv, mask = _episode()
    assert ((policy - rollout).exp() * mask > CAP).any(), "the TIS cap must engage in this fixture"
    leaf = policy.clone().requires_grad_(True)
    (loss, _metrics), weights = _call(_config(), leaf, policy.clone(), rollout, adv, mask)  # pi_old == pi_theta
    loss.backward()
    ref_leaf = policy.clone().requires_grad_(True)
    ref = -(adv[:, None] * weights * ref_leaf * mask).sum() / mask.sum()  # token-mean REINFORCE x TIS
    ref.backward()
    assert torch.allclose(leaf.grad, ref_leaf.grad, atol=1e-10)
    assert torch.allclose(leaf.grad[mask == 0], torch.zeros_like(leaf.grad[mask == 0]))


def _one_token_grad(ratio, advantage, **cfg):
    """d loss / d log pi_theta for a single token with r = pi_theta / pi_old = ratio and w = 1."""
    old = torch.tensor([[-1.0]], dtype=torch.float64)
    mask = torch.ones(1, 1, dtype=torch.float64)
    global RESP_LEN
    saved, RESP_LEN = RESP_LEN, 1
    try:
        leaf = (old + math.log(ratio)).clone().requires_grad_(True)
        (loss, metrics), _ = _call(_config(**cfg), leaf, old, old.clone(), torch.tensor([advantage]), mask)
        loss.backward()
    finally:
        RESP_LEN = saved
    return leaf.grad.item(), metrics


@pytest.mark.parametrize(
    "ratio, advantage, clipped",
    [
        (1.25, 1.0, False),  # inside 1 + 0.28
        (1.30, 1.0, True),  # clip-higher bound
        (0.70, 1.0, False),  # A > 0 is never clipped from below
        (0.85, -1.0, False),  # inside 1 - 0.2
        (0.75, -1.0, True),  # lower bound
        (1.50, -1.0, False),  # A < 0 above 1 + eps: -A * r is the max, still trained ...
        (9.00, -1.0, False),  # ... up to the dual-clip bound c = 10
        (12.0, -1.0, True),  # dual clip: constant -A * c, no gradient
    ],
)
def test_clip_higher_and_dual_clip(ratio, advantage, clipped):
    grad, _ = _one_token_grad(ratio, advantage)
    if clipped:
        assert grad == pytest.approx(0.0, abs=1e-12)
    else:
        assert grad == pytest.approx(-advantage * ratio, rel=1e-9)  # d(-A r)/d log pi = -A r


def test_symmetric_band_clips_where_clip_higher_does_not():
    grad_high, _ = _one_token_grad(1.25, 1.0, high=0.28)
    grad_sym, _ = _one_token_grad(1.25, 1.0, high=0.2)
    assert grad_high != 0.0 and grad_sym == pytest.approx(0.0, abs=1e-12)


def test_ess_and_pearson_inputs_are_emitted():
    policy, rollout, adv, mask = _episode(seed=1)
    old = policy - 0.05  # a later mini-batch: pi_theta moved away from pi_old
    leaf = policy.clone().requires_grad_(True)
    (_loss, metrics), _ = _call(_config(ess=True, pearson=True), leaf, old, rollout, adv, mask)
    expected = [float(((policy[i] - rollout[i]) * mask[i]).sum()) for i in range(mask.shape[0])]
    # the brake measures pi_theta / pi_rollout, not pi_old
    assert metrics[ESS_SEQ_LOG_IS_KEY] == pytest.approx(expected, rel=1e-12)
    assert "training/rollout_actor_probs_pearson_corr" in metrics
    assert "actor/pg_clipfrac" in metrics
