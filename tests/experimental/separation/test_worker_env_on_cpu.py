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
"""Tests for trainer.worker_env plumbing in SeparateRayPPOTrainer._init_worker_groups.

trainer.worker_env must reach the training-side RayWorkerGroup as plain string
env vars (delivered via Ray runtime_env), and must not be forwarded at all when
absent — rollout-side processes are spawned elsewhere and never see it.
"""

from unittest import mock

from omegaconf import OmegaConf

from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer


def _make_trainer(trainer_overrides=None):
    """Build a bare SeparateRayPPOTrainer with just the state _init_worker_groups reads."""
    trainer = SeparateRayPPOTrainer.__new__(SeparateRayPPOTrainer)
    trainer_cfg = {"ray_wait_register_center_timeout": None}
    if trainer_overrides:
        trainer_cfg.update(trainer_overrides)
    trainer.config = OmegaConf.create(
        {
            "trainer": trainer_cfg,
            "global_profiler": {"steps": None},
        }
    )
    trainer.device_name = "cpu"
    trainer.resource_pool_to_cls = {mock.MagicMock(): {"actor": mock.MagicMock()}}
    wg_dict = mock.MagicMock()
    wg_dict.spawn.return_value = {"actor": mock.MagicMock()}
    trainer.ray_worker_group_cls = mock.MagicMock(return_value=wg_dict)
    return trainer


@mock.patch("verl.experimental.separation.ray_trainer.create_colocated_worker_cls")
def test_worker_env_forwarded_as_str_dict(mock_create_cls):
    trainer = _make_trainer(
        {"worker_env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "SOME_INT": 1}}
    )

    trainer._init_worker_groups()

    _, kwargs = trainer.ray_worker_group_cls.call_args
    assert kwargs["worker_env"] == {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SOME_INT": "1",
    }
    # Ray runtime_env requires plain str values, not OmegaConf nodes
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in kwargs["worker_env"].items())


@mock.patch("verl.experimental.separation.ray_trainer.create_colocated_worker_cls")
def test_worker_env_absent_by_default(mock_create_cls):
    trainer = _make_trainer()

    trainer._init_worker_groups()

    _, kwargs = trainer.ray_worker_group_cls.call_args
    assert "worker_env" not in kwargs


@mock.patch("verl.experimental.separation.ray_trainer.create_colocated_worker_cls")
def test_worker_env_empty_dict_not_forwarded(mock_create_cls):
    trainer = _make_trainer({"worker_env": {}})

    trainer._init_worker_groups()

    _, kwargs = trainer.ray_worker_group_cls.call_args
    assert "worker_env" not in kwargs
