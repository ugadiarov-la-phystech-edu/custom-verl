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
"""Mixed bypass loss (rollout_correction.replay_loss_type): PPO-clip on fresh rows, REINFORCE + TIS on replayed rows.

Checked against hand-written references: the per-part objectives, the single joint aggregation (all loss_agg_modes,
micro-batching), clip-higher's asymmetry, the metrics, the rejection mask, the unchanged defaults, the
compute_policy_loss_vanilla refactor, and the config validation."""

import math

import pytest
import torch

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_policy_loss_bypass_mode,
    compute_policy_loss_vanilla,
)
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask
from verl.workers.config import ActorConfig, PolicyLossConfig

C = 8.0  # token-TIS cap of the 2+6 arm
LOW, HIGH, DUAL = 0.2, 0.28, 3.0
AGG_MODES = ["token-mean", "token-sum", "seq-mean-token-sum", "seq-mean-token-mean"]
LENGTHS = [6, 2, 5, 1, 4, 6, 3, 2]
FRESH = [1, 0, 1, 1, 0, 0, 1, 0]
RESP_LEN = 6


def _config(loss_type="ppo_clip", replay_loss_type="reinforce", loss_agg_mode="token-mean", **rc):
    cfg = ActorConfig(
        strategy="megatron",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        loss_agg_mode=loss_agg_mode,
        clip_ratio=LOW,
        clip_ratio_low=LOW,
        clip_ratio_high=HIGH,
        clip_ratio_c=DUAL,
        policy_loss=PolicyLossConfig(
            loss_mode="bypass_mode",
            rollout_correction={
                "rollout_is": "token",
                "rollout_is_threshold": C,
                "bypass_mode": True,
                "loss_type": loss_type,
                "replay_loss_type": replay_loss_type,
                **rc,
            },
        ),
    )
    return cfg


def _set_global(cfg, mask, dp_size=1):
    cfg.global_batch_info.update(
        dp_size=dp_size,
        batch_num_tokens=int(mask.sum()),
        global_batch_size=int((mask.sum(-1) > 0).sum()),
        loss_scale_factor=None,
    )
    return cfg


def _episode(seed=0, lengths=LENGTHS, spread=0.6):
    g = torch.Generator().manual_seed(seed)
    bsz = len(lengths)
    mask = torch.zeros(bsz, RESP_LEN, dtype=torch.float64)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1.0
    log_prob = -torch.rand(bsz, RESP_LEN, generator=g, dtype=torch.float64) * 3
    rollout = log_prob - torch.randn(bsz, RESP_LEN, generator=g, dtype=torch.float64) * spread
    adv = torch.randn(bsz, generator=g, dtype=torch.float64)[:, None] * mask
    return log_prob, rollout, adv, mask


def _ref_ppo(log_prob, rollout, adv):
    r = torch.exp(torch.clamp(log_prob - rollout, -20.0, 20.0))
    pg1 = -adv * r
    pg2 = -adv * torch.clamp(r, 1 - LOW, 1 + HIGH)
    c1 = torch.maximum(pg1, pg2)
    c2 = torch.minimum(-adv * DUAL, c1)
    return torch.where(adv < 0, c2, c1)


def _ref_reinforce(log_prob, rollout, adv):
    w = torch.exp(log_prob - rollout).detach().clamp(max=C)
    return -adv * log_prob * w


def _ref_mixed(log_prob, rollout, adv, fresh):
    return torch.where(fresh.bool()[:, None], _ref_ppo(log_prob, rollout, adv), _ref_reinforce(log_prob, rollout, adv))


def _loss(cfg, log_prob, rollout, adv, mask, is_fresh, agg="token-mean"):
    return compute_policy_loss_bypass_mode(
        old_log_prob=rollout,
        log_prob=log_prob,
        advantages=adv,
        response_mask=mask,
        loss_agg_mode=agg,
        config=cfg,
        is_fresh=is_fresh,
    )


def _loss_and_grad(cfg, log_prob, rollout, adv, mask, is_fresh, agg="token-mean"):
    leaf = log_prob.clone().requires_grad_(True)
    loss, metrics = _loss(cfg, leaf, rollout, adv, mask, is_fresh, agg)
    loss.backward()
    return loss.detach(), leaf.grad, metrics


FRESH_T = torch.tensor(FRESH, dtype=torch.float32)


class TestReducesToTheSingleLosses:
    @pytest.mark.parametrize("agg", AGG_MODES)
    def test_all_fresh_equals_ppo_clip(self, agg):
        lp, ro, adv, mask = _episode()
        mixed = _loss_and_grad(_set_global(_config(), mask), lp, ro, adv, mask, torch.ones(len(LENGTHS)), agg)
        plain = _loss_and_grad(_set_global(_config(replay_loss_type=None), mask), lp, ro, adv, mask, None, agg)
        torch.testing.assert_close(mixed[0], plain[0], rtol=0, atol=1e-12)
        torch.testing.assert_close(mixed[1], plain[1], rtol=0, atol=1e-12)
        assert mixed[2]["actor/pg_clipfrac"] == pytest.approx(plain[2]["actor/pg_clipfrac"])
        assert mixed[2]["actor/pg_clipfrac_lower"] == pytest.approx(plain[2]["actor/pg_clipfrac_lower"])

    @pytest.mark.parametrize("agg", AGG_MODES)
    def test_all_replayed_equals_reinforce_tis(self, agg):
        lp, ro, adv, mask = _episode()
        mixed = _loss_and_grad(_set_global(_config(), mask), lp, ro, adv, mask, torch.zeros(len(LENGTHS)), agg)
        cfg = _set_global(_config(loss_type="reinforce", replay_loss_type=None), mask)
        plain = _loss_and_grad(cfg, lp, ro, adv, mask, None, agg)
        torch.testing.assert_close(mixed[0], plain[0], rtol=0, atol=1e-12)
        torch.testing.assert_close(mixed[1], plain[1], rtol=0, atol=1e-12)
        assert mixed[2]["actor/pg_clipfrac"] == 0.0 and mixed[2]["actor/mixed/replay_token_frac"] == 1.0

    @pytest.mark.parametrize("loss_type", ["ppo_clip", "reinforce"])
    def test_default_ignores_the_tag(self, loss_type):
        # replay_loss_type=None (the default): is_fresh changes nothing, metrics keep their old keys
        lp, ro, adv, mask = _episode()
        cfg = _set_global(_config(loss_type=loss_type, replay_loss_type=None), mask)
        tagged = _loss_and_grad(cfg, lp, ro, adv, mask, FRESH_T)
        untagged = _loss_and_grad(cfg, lp, ro, adv, mask, None)
        torch.testing.assert_close(tagged[0], untagged[0], rtol=0, atol=0)
        torch.testing.assert_close(tagged[1], untagged[1], rtol=0, atol=0)
        assert tagged[2].keys() == untagged[2].keys()
        assert not any(k.startswith("actor/mixed/") for k in tagged[2])

    def test_no_tag_falls_back_to_loss_type(self):
        # replay_loss_type set but no is_fresh (e.g. a non-replay trainer): every row uses loss_type
        lp, ro, adv, mask = _episode()
        got = _loss_and_grad(_set_global(_config(), mask), lp, ro, adv, mask, None)
        want = _loss_and_grad(_set_global(_config(replay_loss_type=None), mask), lp, ro, adv, mask, None)
        torch.testing.assert_close(got[0], want[0], rtol=0, atol=0)
        torch.testing.assert_close(got[1], want[1], rtol=0, atol=0)

    def test_registered_entry_point_is_the_mixed_loss(self):
        assert core_algos.get_policy_loss_fn("bypass_mode") is compute_policy_loss_bypass_mode


class TestMixedObjective:
    @pytest.mark.parametrize("agg", AGG_MODES)
    def test_matches_reference(self, agg):
        lp, ro, adv, mask = _episode(seed=1)
        cfg = _set_global(_config(loss_agg_mode=agg), mask)
        loss, grad, _ = _loss_and_grad(cfg, lp, ro, adv, mask, FRESH_T, agg)
        leaf = lp.clone().requires_grad_(True)
        ref = agg_loss(_ref_mixed(leaf, ro, adv, FRESH_T), mask, agg, **cfg.global_batch_info)
        ref.backward()
        torch.testing.assert_close(loss, ref.detach(), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(grad, leaf.grad, rtol=1e-12, atol=1e-12)

    def test_is_one_joint_aggregation_not_two_means(self):
        # token-mean over all tokens: a 50/50 average of the two parts' own means would weight differently
        lp, ro, adv, mask = _episode(seed=2)
        cfg = _set_global(_config(), mask)
        loss, _ = _loss(cfg, lp, ro, adv, mask, FRESH_T)
        f = FRESH_T.bool()[:, None].to(mask.dtype) * mask
        r = mask - f
        joint = (_ref_mixed(lp, ro, adv, FRESH_T) * mask).sum() / mask.sum()
        separate = 0.5 * ((_ref_ppo(lp, ro, adv) * f).sum() / f.sum() + (_ref_reinforce(lp, ro, adv) * r).sum() / r.sum())
        torch.testing.assert_close(loss, joint, rtol=1e-12, atol=1e-12)
        assert not math.isclose(loss.item(), separate.item(), rel_tol=1e-6)

    @pytest.mark.parametrize("agg", AGG_MODES)
    @pytest.mark.parametrize("chunks", [[8], [4, 4], [3, 2, 3], [1] * 8])
    def test_micro_batch_and_dp_invariance(self, agg, chunks):
        lp, ro, adv, mask = _episode(seed=4)
        full_cfg = _set_global(_config(loss_agg_mode=agg), mask)
        full_loss, full_grad, _ = _loss_and_grad(full_cfg, lp, ro, adv, mask, FRESH_T, agg)
        leaf = lp.clone().requires_grad_(True)
        total = 0.0
        start = 0
        for n in chunks:
            rows = slice(start, start + n)
            start += n
            cfg = _config(loss_agg_mode=agg)
            cfg.global_batch_info.update(full_cfg.global_batch_info)  # global denominators, as ppo_loss sets them
            loss, _ = _loss(cfg, leaf[rows], ro[rows], adv[rows], mask[rows], FRESH_T[rows], agg)
            loss.backward()
            total = total + loss.detach()
        torch.testing.assert_close(total, full_loss, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(leaf.grad, full_grad, rtol=1e-12, atol=1e-12)

    def test_padding_gets_no_gradient(self):
        lp, ro, adv, mask = _episode(seed=5)
        _, grad, _ = _loss_and_grad(_set_global(_config(), mask), lp, ro, adv, mask, FRESH_T)
        assert (grad[mask == 0] == 0).all() and grad.abs().sum() > 0

    def test_swapped_roles(self):
        # replay_loss_type=ppo_clip with loss_type=reinforce: the parts swap
        lp, ro, adv, mask = _episode(seed=6)
        cfg = _set_global(_config(loss_type="reinforce", replay_loss_type="ppo_clip"), mask)
        loss, _ = _loss(cfg, lp, ro, adv, mask, FRESH_T)
        ref = agg_loss(_ref_mixed(lp, ro, adv, 1 - FRESH_T), mask, "token-mean", **cfg.global_batch_info)
        torch.testing.assert_close(loss, ref, rtol=1e-12, atol=1e-12)

    def test_same_type_on_both_parts_equals_plain(self):
        lp, ro, adv, mask = _episode(seed=7)
        both = _loss(_set_global(_config(replay_loss_type="ppo_clip"), mask), lp, ro, adv, mask, FRESH_T)[0]
        plain = _loss(_set_global(_config(replay_loss_type=None), mask), lp, ro, adv, mask, None)[0]
        torch.testing.assert_close(both, plain, rtol=0, atol=1e-12)


def _single_token(log_ratio, adv, fresh):
    """One row per (log_ratio, adv) pair, one token each; returns d loss / d log_prob per row."""
    n = len(log_ratio)
    lp = torch.full((n, 1), -1.0, dtype=torch.float64)
    ro = lp - torch.tensor(log_ratio, dtype=torch.float64)[:, None]
    a = torch.tensor(adv, dtype=torch.float64)[:, None]
    mask = torch.ones(n, 1, dtype=torch.float64)
    cfg = _set_global(_config(), mask)
    _, grad, metrics = _loss_and_grad(cfg, lp, ro, a, mask, torch.tensor(fresh, dtype=torch.float32))
    return grad[:, 0] * n, metrics  # undo the token-mean 1/n


class TestClipBehaviour:
    def test_clip_higher_is_asymmetric_on_fresh_tokens(self):
        r = [1.25, 1.3, 0.85, 0.75]
        grad, _ = _single_token([math.log(x) for x in r], [1.0, 1.0, -1.0, -1.0], [1, 1, 1, 1])
        # A>0: unclipped up to 1+0.28 (d(-A r)/dlog = -A r), clipped beyond; A<0: unclipped down to 1-0.2
        torch.testing.assert_close(grad, torch.tensor([-1.25, 0.0, 0.85, 0.0], dtype=torch.float64))

    def test_dual_clip_on_fresh_tokens(self):
        # A<0 and r > c=3: the loss is the constant -A*c, so no gradient
        grad, metrics = _single_token([math.log(4.0), math.log(2.0)], [-1.0, -1.0], [1, 1])
        torch.testing.assert_close(grad, torch.tensor([0.0, 2.0], dtype=torch.float64))
        assert metrics["actor/pg_clipfrac_lower"] == pytest.approx(0.5)

    def test_replayed_tokens_are_never_clipped_but_tis_capped(self):
        r = [1.3, 0.5, 10.0]
        grad, metrics = _single_token([math.log(x) for x in r], [1.0, 1.0, 1.0], [0, 0, 0])
        # REINFORCE + TIS: d(-A log pi * w)/dlog = -A w, w = min(r, C)
        torch.testing.assert_close(grad, torch.tensor([-1.3, -0.5, -C], dtype=torch.float64))
        assert metrics["actor/pg_clipfrac"] == 0.0


class TestMetrics:
    def test_clipfrac_counts_fresh_tokens_only(self):
        # fresh: [clipped, unclipped]; replayed: two tokens far outside the clip range
        grad, m = _single_token([math.log(1.5), math.log(1.1), math.log(5.0), math.log(5.0)], [1.0] * 4, [1, 1, 0, 0])
        assert m["actor/pg_clipfrac"] == pytest.approx(0.5)
        assert m["actor/mixed/fresh_token_frac"] == pytest.approx(0.5)
        assert m["actor/mixed/replay_token_frac"] == pytest.approx(0.5)

    def test_token_fractions_and_kl(self):
        lp, ro, adv, mask = _episode(seed=8)
        _, m = _loss(_set_global(_config(), mask), lp, ro, adv, mask, FRESH_T)
        fresh_tokens = (mask * FRESH_T.double()[:, None]).sum() / mask.sum()
        assert m["actor/mixed/fresh_token_frac"] == pytest.approx(fresh_tokens.item())
        assert m["actor/mixed/fresh_token_frac"] + m["actor/mixed/replay_token_frac"] == pytest.approx(1.0)
        kl = ((ro - lp) * mask).sum() / mask.sum()
        assert m["actor/ppo_kl"] == pytest.approx(kl.item())
        assert any(k.startswith("rollout_corr/") for k in m), "rollout-correction metrics must still be merged"

    def test_micro_batch_without_ppo_rows(self):
        lp, ro, adv, mask = _episode(seed=9)
        _, m = _loss(_set_global(_config(), mask), lp, ro, adv, mask, torch.zeros(len(LENGTHS)))
        assert m["actor/pg_clipfrac"] == 0.0 and m["actor/pg_clipfrac_lower"] == 0.0
        assert m["actor/mixed/fresh_token_frac"] == 0.0


class TestRejectionMask:
    def test_rejection_applies_to_both_parts(self):
        lp, ro, adv, mask = _episode(seed=10, spread=1.5)
        rs = {"rollout_rs": "token_k1", "rollout_rs_threshold": "0.5_2.0"}
        cfg = _set_global(_config(**rs), mask)
        loss, _ = _loss(cfg, lp, ro, adv, mask, FRESH_T)
        _, kept, _ = compute_rollout_correction_and_rejection_mask(
            old_log_prob=lp,
            rollout_log_prob=ro,
            response_mask=mask,
            rollout_is="token",
            rollout_is_threshold=C,
            rollout_rs="token_k1",
            rollout_rs_threshold="0.5_2.0",
        )
        kept = kept.to(mask.dtype)
        assert 0 < kept.sum() < mask.sum(), "the fixture must reject some tokens"
        ref = agg_loss(_ref_mixed(lp, ro, adv, FRESH_T), kept, "token-mean", **cfg.global_batch_info)
        torch.testing.assert_close(loss, ref, rtol=1e-12, atol=1e-12)


class TestValidation:
    def test_rejects_bad_types_and_shapes(self):
        lp, ro, adv, mask = _episode()
        with pytest.raises(ValueError, match="replay_loss_type"):
            _loss(_set_global(_config(replay_loss_type="vanilla"), mask), lp, ro, adv, mask, FRESH_T)
        with pytest.raises(ValueError, match="loss_type"):
            _loss(_set_global(_config(loss_type="vanilla"), mask), lp, ro, adv, mask, FRESH_T)
        with pytest.raises(ValueError, match="rows"):
            _loss(_set_global(_config(), mask), lp, ro, adv, mask, FRESH_T[:3])

    def test_config(self):
        assert RolloutCorrectionConfig().replay_loss_type is None
        for t in ("reinforce", "ppo_clip"):
            assert RolloutCorrectionConfig(bypass_mode=True, replay_loss_type=t).replay_loss_type == t
        with pytest.raises(ValueError, match="must be None"):
            RolloutCorrectionConfig(bypass_mode=True, replay_loss_type="vanilla")
        with pytest.raises(ValueError, match="bypass_mode"):
            RolloutCorrectionConfig(bypass_mode=False, replay_loss_type="reinforce")

    def test_yaml_default(self):
        from pathlib import Path

        import yaml

        root = Path(core_algos.__file__).resolve().parents[1] / "config"
        assert yaml.safe_load((root / "algorithm" / "rollout_correction.yaml").read_text())["replay_loss_type"] is None
        for gen in sorted(root.glob("_generated_*.yaml")):
            cfg = yaml.safe_load(gen.read_text())
            assert cfg["algorithm"]["rollout_correction"]["replay_loss_type"] is None, gen.name


def _old_vanilla(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config, rollout_is_weights):
    """compute_policy_loss_vanilla before the _ppo_clip_token_losses refactor, written out."""
    neg_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(neg_kl)
    ppo_kl = (-neg_kl * response_mask).sum() / response_mask.sum()
    pg1 = -advantages * ratio
    pg2 = -advantages * torch.clamp(ratio, 1 - config.clip_ratio_low, 1 + config.clip_ratio_high)
    c1 = torch.maximum(pg1, pg2)
    clipfrac = (torch.gt(pg2, pg1).float() * response_mask).sum() / response_mask.sum()
    pg3 = -advantages * config.clip_ratio_c
    c2 = torch.min(pg3, c1)
    clipfrac_lower = ((torch.gt(c1, pg3) * (advantages < 0).float()) * response_mask).sum() / response_mask.sum()
    pg = torch.where(advantages < 0, c2, c1)
    if rollout_is_weights is not None:
        pg = pg * rollout_is_weights
    loss = agg_loss(pg, response_mask, loss_agg_mode, **config.global_batch_info)
    return loss, {"actor/pg_clipfrac": clipfrac.item(), "actor/ppo_kl": ppo_kl.item(), "actor/pg_clipfrac_lower": clipfrac_lower.item()}


class TestVanillaRefactor:
    @pytest.mark.parametrize("agg", AGG_MODES)
    @pytest.mark.parametrize("with_is", [False, True])
    def test_unchanged(self, agg, with_is):
        lp, old, adv, mask = _episode(seed=11, spread=1.0)
        cfg = _set_global(_config(loss_agg_mode=agg), mask)
        w = torch.rand_like(lp) * 2 if with_is else None
        leaf_new = lp.clone().requires_grad_(True)
        new_loss, new_m = compute_policy_loss_vanilla(old, leaf_new, adv, mask, agg, cfg, w)
        new_loss.backward()
        leaf_old = lp.clone().requires_grad_(True)
        old_loss, old_m = _old_vanilla(old, leaf_old, adv, mask, agg, cfg, w)
        old_loss.backward()
        torch.testing.assert_close(new_loss, old_loss, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(leaf_new.grad, leaf_old.grad, rtol=1e-12, atol=1e-12)
        for k, v in old_m.items():
            assert new_m[k] == pytest.approx(v, rel=1e-6, abs=1e-12), k
        assert old_m["actor/pg_clipfrac"] > 0 and old_m["actor/pg_clipfrac_lower"] > 0, "fixture must clip both ways"
