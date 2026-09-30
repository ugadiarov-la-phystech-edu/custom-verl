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
"""actor.ess_scaling and rollout_correction.log_probs_pearson_corr: validation, yaml defaults, and what the
actor worker actually receives after Hydra composition (omega_conf_to_dataclass of the actor config)."""

import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException
from omegaconf import OmegaConf

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import ActorConfig, ESSScalingConfig

CONFIG_DIR = os.path.abspath("verl/trainer/config")

# the objective the replay-buffer ESS scripts select (bypass + REINFORCE + token TIS), with the two
# explicit policy_loss keys losses.py needs (it reads only actor.policy_loss)
REPLAY_ESS_OVERRIDES = [
    "algorithm.rollout_correction.bypass_mode=True",
    "algorithm.rollout_correction.loss_type=reinforce",
    "algorithm.rollout_correction.rollout_is=token",
    "algorithm.rollout_correction.rollout_is_threshold=2.0",
    "algorithm.rollout_correction.log_probs_pearson_corr=True",
    "actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode",
    "+actor_rollout_ref.actor.policy_loss.rollout_correction=${algorithm.rollout_correction}",
    "actor_rollout_ref.actor.ess_scaling.enable=True",
    "actor_rollout_ref.actor.ess_scaling.min_ess=1.07",
    "actor_rollout_ref.actor.ess_scaling.lr_scale=0.5",
]


def _compose(config_name, overrides=()):
    overrides = ["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1", *overrides]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


class TestESSScalingConfig:
    def test_defaults_are_off(self):
        cfg = ESSScalingConfig()
        assert (cfg.enable, cfg.min_ess, cfg.lr_scale, cfg.use_clipped) == (False, 1.1, 0.5, False)

    @pytest.mark.parametrize("min_ess", [1.0, 1.07, 1.1, 64.0])
    def test_valid_min_ess(self, min_ess):
        assert ESSScalingConfig(min_ess=min_ess).min_ess == min_ess

    @pytest.mark.parametrize("min_ess", [0.0, 0.5, 0.999])
    def test_min_ess_below_floor_rejected(self, min_ess):
        # the max-shifted ESS never goes below 1, so a threshold below 1 would never brake
        with pytest.raises(AssertionError, match="min_ess"):
            ESSScalingConfig(min_ess=min_ess)

    @pytest.mark.parametrize("lr_scale", [1e-6, 0.5, 1.0])
    def test_valid_lr_scale(self, lr_scale):
        assert ESSScalingConfig(lr_scale=lr_scale).lr_scale == lr_scale

    @pytest.mark.parametrize("lr_scale", [0.0, -0.5, 1.01, 2.0])
    def test_lr_scale_outside_unit_interval_rejected(self, lr_scale):
        with pytest.raises(AssertionError, match="lr_scale"):
            ESSScalingConfig(lr_scale=lr_scale)

    def test_frozen(self):
        cfg = ESSScalingConfig()
        with pytest.raises(Exception, match="frozen"):
            cfg.enable = True

    def test_actor_config_default(self):
        actor = ActorConfig(strategy="megatron", rollout_n=1, ppo_micro_batch_size_per_gpu=1)
        assert isinstance(actor.ess_scaling, ESSScalingConfig)
        assert actor.ess_scaling.enable is False


def test_pearson_flag_default_off():
    assert RolloutCorrectionConfig().log_probs_pearson_corr is False


@pytest.mark.parametrize("config_name", ["ppo_trainer", "ppo_megatron_trainer"])
def test_trainer_yaml_defaults(config_name):
    cfg = _compose(config_name)
    ess = cfg.actor_rollout_ref.actor.ess_scaling
    assert OmegaConf.to_container(ess) == {
        "_target_": "verl.workers.config.ESSScalingConfig",
        "enable": False,
        "min_ess": 1.1,
        "lr_scale": 0.5,
        "use_clipped": False,
    }
    assert cfg.algorithm.rollout_correction.log_probs_pearson_corr is False
    actor = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
    assert isinstance(actor.ess_scaling, ESSScalingConfig)
    assert actor.ess_scaling.enable is False


@pytest.mark.parametrize("config_name", ["ppo_trainer", "ppo_megatron_trainer"])
def test_invalid_yaml_values_fail_at_conversion(config_name):
    cfg = _compose(config_name, ["actor_rollout_ref.actor.ess_scaling.lr_scale=0"])
    # Hydra's instantiate wraps the dataclass assertion
    with pytest.raises(InstantiationException, match="lr_scale"):
        omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)


class TestWorkerSideActorConfig:
    """What ActorRolloutRefWorker.init_model builds from the composed actor config (omega_conf_to_dataclass)."""

    @pytest.fixture(scope="class")
    @staticmethod
    def actor():
        cfg = _compose("ppo_megatron_trainer", REPLAY_ESS_OVERRIDES)
        return omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)

    def test_ess_scaling_reaches_worker(self, actor):
        assert actor.ess_scaling == ESSScalingConfig(enable=True, min_ess=1.07, lr_scale=0.5, use_clipped=False)

    def test_loss_mode_is_bypass(self, actor):
        # losses.ppo_loss dispatches on policy_loss.loss_mode only
        assert actor.policy_loss.loss_mode == "bypass_mode"

    def test_rollout_correction_survives(self, actor):
        corr = actor.policy_loss.rollout_correction
        assert corr["loss_type"] == "reinforce"
        assert corr["rollout_is"] == "token"
        assert float(corr["rollout_is_threshold"]) == 2.0
        assert corr["bypass_mode"] is True
        assert corr["log_probs_pearson_corr"] is True

    def test_without_explicit_policy_loss_keys_the_worker_does_not_see_the_objective(self):
        # the footgun the scripts guard against: algorithm.rollout_correction alone never reaches the loss
        overrides = [o for o in REPLAY_ESS_OVERRIDES if not o.lstrip("+").startswith("actor_rollout_ref.actor.policy")]
        actor = omega_conf_to_dataclass(_compose("ppo_megatron_trainer", overrides).actor_rollout_ref.actor)
        assert actor.policy_loss.loss_mode != "bypass_mode"
        assert actor.policy_loss.rollout_correction.get("loss_type", "ppo_clip") != "reinforce"
