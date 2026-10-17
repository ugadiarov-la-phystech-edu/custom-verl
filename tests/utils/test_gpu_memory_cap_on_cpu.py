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
"""VERL_GPU_MEM_CAP_GB: the allocator-cap helper and its wiring into the actor worker."""

import logging

import pytest
from omegaconf import OmegaConf

import verl.utils.gpu_memory_cap as gpu_memory_cap
from verl.utils.gpu_memory_cap import (
    GPU_MEM_CAP_ENV,
    apply_gpu_memory_cap,
    apply_rollout_gpu_memory_cap,
    rollout_allocator_fraction,
)

GiB = 1024**3


class _FakeDevice:
    def __init__(self, total_gib=143.8, available=True):
        self.total = int(total_gib * GiB)
        self.available = available
        self.fractions = []
        self.fraction_devices = []

    def is_available(self):
        return self.available

    def current_device(self):
        return 0

    def get_device_properties(self, idx):
        assert idx == 0
        return type("Props", (), {"total_memory": self.total})()

    def set_per_process_memory_fraction(self, fraction, device=None):
        self.fractions.append(fraction)
        self.fraction_devices.append(device)


@pytest.fixture
def device(monkeypatch):
    dev = _FakeDevice()
    monkeypatch.setattr(gpu_memory_cap, "get_torch_device", lambda: dev)
    monkeypatch.delenv(GPU_MEM_CAP_ENV, raising=False)
    return dev


class TestUnset:
    def test_env_name(self):
        assert GPU_MEM_CAP_ENV == "VERL_GPU_MEM_CAP_GB"

    def test_absent_env_is_a_noop(self, device):
        assert apply_gpu_memory_cap() is None
        assert device.fractions == []

    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_or_blank_is_a_noop(self, device, monkeypatch, value):
        monkeypatch.setenv(GPU_MEM_CAP_ENV, value)
        assert apply_gpu_memory_cap() is None
        assert device.fractions == []


class TestApplied:
    def test_h100_on_h200_fraction(self, device, monkeypatch):
        monkeypatch.setenv(GPU_MEM_CAP_ENV, "80")
        fraction = apply_gpu_memory_cap()
        assert fraction == pytest.approx(80 / 143.8, rel=1e-6)
        assert device.fractions == [fraction]

    @pytest.mark.parametrize("total_gib", [94.0, 143.8, 192.0])
    def test_fraction_is_relative_to_the_device_size(self, monkeypatch, total_gib):
        dev = _FakeDevice(total_gib=total_gib)
        monkeypatch.setattr(gpu_memory_cap, "get_torch_device", lambda: dev)
        monkeypatch.setenv(GPU_MEM_CAP_ENV, "80")
        assert apply_gpu_memory_cap() == pytest.approx(80 / total_gib, rel=1e-6)

    @pytest.mark.parametrize("value, gib", [("108", 108.0), ("79.5", 79.5), (" 80 ", 80.0), ("1e2", 100.0)])
    def test_numeric_forms_are_accepted(self, device, monkeypatch, value, gib):
        monkeypatch.setenv(GPU_MEM_CAP_ENV, value)
        assert apply_gpu_memory_cap() == pytest.approx(gib / 143.8, rel=1e-6)

    def test_logs_what_it_applied(self, device, monkeypatch, caplog):
        monkeypatch.setenv(GPU_MEM_CAP_ENV, "80")
        with caplog.at_level(logging.WARNING, logger=gpu_memory_cap.logger.name):
            apply_gpu_memory_cap()
        assert "Capped the trainer allocator at 80.0 GiB" in caplog.text


class TestGuards:
    @pytest.mark.parametrize("value", ["143.8", "200"])
    def test_cap_at_or_above_the_device_is_skipped_with_a_warning(self, device, monkeypatch, caplog, value):
        monkeypatch.setenv(GPU_MEM_CAP_ENV, value)
        with caplog.at_level(logging.WARNING, logger=gpu_memory_cap.logger.name):
            assert apply_gpu_memory_cap() is None
        assert device.fractions == []
        assert "leaving the allocator uncapped" in caplog.text

    @pytest.mark.parametrize("value", ["0", "-1", "-0.5", "nan"])
    def test_non_positive_raises(self, device, monkeypatch, value):
        monkeypatch.setenv(GPU_MEM_CAP_ENV, value)
        with pytest.raises(ValueError, match="must be a positive number"):
            apply_gpu_memory_cap()
        assert device.fractions == []

    @pytest.mark.parametrize("value", ["80GB", "eighty", "80,5"])
    def test_garbage_raises_a_clear_error(self, device, monkeypatch, value):
        monkeypatch.setenv(GPU_MEM_CAP_ENV, value)
        with pytest.raises(ValueError, match=GPU_MEM_CAP_ENV):
            apply_gpu_memory_cap()

    def test_no_accelerator_is_a_noop(self, monkeypatch):
        dev = _FakeDevice(available=False)
        monkeypatch.setattr(gpu_memory_cap, "get_torch_device", lambda: dev)
        monkeypatch.setenv(GPU_MEM_CAP_ENV, "80")
        assert apply_gpu_memory_cap() is None
        assert dev.fractions == []


# --------------------------------------------------------------------------- standalone rollout cap

H200 = int(139.8 * GiB)


class TestRolloutFraction:
    def test_unset_cap_is_none(self, device):
        assert rollout_allocator_fraction(H200, 0.5) is None

    def test_same_cap_as_the_trainer(self, device, monkeypatch):
        monkeypatch.setenv(GPU_MEM_CAP_ENV, "78")
        fraction = rollout_allocator_fraction(H200, 0.5)
        assert fraction == pytest.approx(78 / 139.8, rel=1e-6)
        monkeypatch.setattr(gpu_memory_cap, "get_torch_device", lambda: _FakeDevice(total_gib=139.8))
        assert apply_gpu_memory_cap() == pytest.approx(fraction, rel=1e-9)

    @pytest.mark.parametrize("value, gib", [(" 80 ", 80.0), ("1e2", 100.0), ("79.5", 79.5)])
    def test_numeric_forms(self, device, monkeypatch, value, gib):
        monkeypatch.setenv(GPU_MEM_CAP_ENV, value)
        assert rollout_allocator_fraction(H200, 0.5) == pytest.approx(gib / 139.8, rel=1e-6)

    @pytest.mark.parametrize("value", ["0", "-1", "eighty", "nan"])
    def test_invalid_cap_raises(self, device, monkeypatch, value):
        monkeypatch.setenv(GPU_MEM_CAP_ENV, value)
        with pytest.raises(ValueError, match=GPU_MEM_CAP_ENV):
            rollout_allocator_fraction(H200, 0.5)

    @pytest.mark.parametrize("cap", ["139.8", "200"])
    def test_whole_device_or_more_is_uncapped(self, device, monkeypatch, caplog, cap):
        monkeypatch.setenv(GPU_MEM_CAP_ENV, cap)
        with caplog.at_level(logging.WARNING, logger=gpu_memory_cap.logger.name):
            assert rollout_allocator_fraction(H200, 0.5) is None
        assert "leaving the rollout allocator uncapped" in caplog.text

    @pytest.mark.parametrize("util", [0.6, 0.558, 0.9])
    def test_not_above_gpu_memory_utilization_is_uncapped(self, device, monkeypatch, caplog, util):
        monkeypatch.setenv(GPU_MEM_CAP_ENV, "78")  # fraction 0.5579
        with caplog.at_level(logging.WARNING, logger=gpu_memory_cap.logger.name):
            assert rollout_allocator_fraction(H200, util) is None
        assert "gpu_memory_utilization" in caplog.text

    def test_just_above_gpu_memory_utilization_is_applied(self, device, monkeypatch):
        monkeypatch.setenv(GPU_MEM_CAP_ENV, "78")
        assert rollout_allocator_fraction(H200, 0.557) == pytest.approx(78 / 139.8, rel=1e-6)


class TestApplyRolloutCap:
    def _device(self, monkeypatch, **kw):
        dev = _FakeDevice(total_gib=139.8, **kw)
        monkeypatch.setattr(gpu_memory_cap, "get_torch_device", lambda: dev)
        return dev

    def test_sets_the_fraction_once_on_the_current_device_and_logs(self, device, monkeypatch, caplog):
        dev = self._device(monkeypatch)
        monkeypatch.setenv(GPU_MEM_CAP_ENV, "78")
        with caplog.at_level(logging.WARNING, logger=gpu_memory_cap.logger.name):
            fraction = apply_rollout_gpu_memory_cap(0.5)
        assert fraction == pytest.approx(78 / 139.8, rel=1e-6)
        assert dev.fractions == [fraction]
        assert dev.fraction_devices == [0]
        assert "Capped the rollout allocator at 78.0 GiB of 139.8 GiB" in caplog.text

    def test_unset_sets_nothing(self, device, monkeypatch):
        dev = self._device(monkeypatch)
        assert apply_rollout_gpu_memory_cap(0.5) is None
        assert dev.fractions == []

    def test_no_accelerator_sets_nothing(self, device, monkeypatch):
        dev = self._device(monkeypatch, available=False)
        monkeypatch.setenv(GPU_MEM_CAP_ENV, "78")
        assert apply_rollout_gpu_memory_cap(0.5) is None
        assert dev.fractions == []

    def test_guarded_fraction_sets_nothing(self, device, monkeypatch):
        dev = self._device(monkeypatch)
        monkeypatch.setenv(GPU_MEM_CAP_ENV, "78")
        assert apply_rollout_gpu_memory_cap(0.7) is None
        assert dev.fractions == []


# --------------------------------------------------------------------------- worker wiring


class _StopInit(Exception):
    """Raised right after the cap call site to end the worker constructor early."""


@pytest.mark.parametrize(
    "role, expect_cap",
    [
        ("actor", True),
        ("actor_rollout", True),
        ("actor_rollout_ref", True),
        ("rollout", False),
        ("ref", False),
    ],
)
def test_actor_worker_caps_trainer_roles_only_and_before_setup(monkeypatch, role, expect_cap):
    import verl.workers.engine_workers as engine_workers

    calls = []
    monkeypatch.setattr(engine_workers, "apply_gpu_memory_cap", lambda: calls.append(role))
    monkeypatch.setattr(engine_workers.Worker, "__init__", lambda self: None)

    def stop(*args, **kwargs):
        raise _StopInit

    # The first thing the constructor does after the cap is build the profiler config;
    # stopping there proves the cap runs before any profiler / model setup.
    monkeypatch.setattr(engine_workers, "omega_conf_to_dataclass", stop)

    config = OmegaConf.create({"actor": {"profiler": {}}, "rollout": {"profiler": {}}, "ref": {"profiler": {}}})
    worker = object.__new__(engine_workers.ActorRolloutRefWorker)
    with pytest.raises(_StopInit):
        engine_workers.ActorRolloutRefWorker.__init__(worker, config, role=role)
    assert calls == ([role] if expect_cap else [])
