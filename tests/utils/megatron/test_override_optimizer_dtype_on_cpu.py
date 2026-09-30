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
"""override_optimizer_config dtype fields reach Megatron's OptimizerConfig as torch.dtype.

Hydra delivers ``+actor_rollout_ref.actor.optim.override_optimizer_config.main_params_dtype=bfloat16``
as the string "bfloat16". OptimizerConfig does not convert it, so without the fix the precision-aware
optimizer would get a string master-weight dtype and every ``== torch.bfloat16`` check would be False.
"""

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("megatron.core")

from verl.utils.megatron.optimizer import init_megatron_optim_config  # noqa: E402

BASE = {
    "optimizer": "adam",
    "lr": 1e-6,
    "min_lr": 0.0,
    "clip_grad": 1.0,
    "weight_decay": 0.1,
}


def _config(**override):
    return OmegaConf.create({**BASE, "override_optimizer_config": override})


def _precision_aware(**dtypes):
    return _config(
        optimizer_cpu_offload=True,
        optimizer_offload_fraction=1.0,
        use_torch_optimizer_for_cpu_offload=True,
        use_precision_aware_optimizer=True,
        **dtypes,
    )


@pytest.mark.parametrize(
    "value, expected",
    [
        ("bfloat16", torch.bfloat16),
        ("bf16", torch.bfloat16),
        ("float32", torch.float32),
        ("fp32", torch.float32),
    ],
)
def test_main_params_dtype_string_becomes_torch_dtype(value, expected):
    cfg = init_megatron_optim_config(_precision_aware(main_params_dtype=value))
    assert cfg.main_params_dtype is expected
    assert cfg.use_precision_aware_optimizer is True


def test_the_baseline_scripts_combination():
    # exactly what the examples/baselines scripts pass
    cfg = init_megatron_optim_config(_precision_aware(main_params_dtype="bfloat16"))
    assert cfg.main_params_dtype == torch.bfloat16
    assert cfg.optimizer_cpu_offload is True
    assert cfg.use_torch_optimizer_for_cpu_offload is True


@pytest.mark.parametrize("key", ["main_grads_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"])
def test_every_dtype_override_is_converted(key):
    cfg = init_megatron_optim_config(_precision_aware(**{key: "bfloat16"}))
    assert getattr(cfg, key) is torch.bfloat16


def test_torch_dtype_values_pass_through():
    # programmatic callers may already pass a torch.dtype (OmegaConf needs allow_objects for that)
    cfg_in = OmegaConf.create(
        {
            **BASE,
            "override_optimizer_config": {
                "optimizer_cpu_offload": True,
                "use_torch_optimizer_for_cpu_offload": True,
                "use_precision_aware_optimizer": True,
                "main_params_dtype": torch.bfloat16,
            },
        },
        flags={"allow_objects": True},
    )
    assert init_megatron_optim_config(cfg_in).main_params_dtype is torch.bfloat16


def test_unknown_dtype_string_is_an_error():
    with pytest.raises(RuntimeError, match="unexpected precision"):
        init_megatron_optim_config(_precision_aware(main_params_dtype="bfloat17"))


def test_non_dtype_overrides_are_untouched():
    cfg = init_megatron_optim_config(_config(optimizer_offload_fraction=0.5, overlap_cpu_optimizer_d2h_h2d=False))
    assert cfg.optimizer_offload_fraction == 0.5
    assert cfg.overlap_cpu_optimizer_d2h_h2d is False
