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
"""VERL_GPU_MEM_CAP_GB: cap PyTorch's allocator to emulate a smaller card.

* Trainer processes: ``apply_gpu_memory_cap`` caps at the cap, from
  ``ActorRolloutRefWorker.__init__`` (``verl/workers/engine_workers.py``) in actor-role processes, so it
  applies to every training engine (FSDP, Megatron, ...).
* Standalone rollout workers (vLLM on their own GPUs): ``apply_rollout_gpu_memory_cap`` caps at the same
  cap, and only if it stays above ``gpu_memory_utilization``, from the vLLM worker right after ``init_device``
  (``verl/workers/rollout/vllm_rollout/utils.py``). Without it, the caching allocator keeps every
  odd-sized sampler block ([running requests x vocab] fp32, a new size each step) and grows to
  ~100 GiB on a 140 GiB card while ~44 GiB is live; at the cap it frees cached blocks and retries.
* Hybrid / colocated rollout shares GPUs with the trainer and is never capped.
"""

import logging
import os

from verl.utils.device import get_torch_device

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

GPU_MEM_CAP_ENV = "VERL_GPU_MEM_CAP_GB"


def _read_cap_bytes() -> float | None:
    """``VERL_GPU_MEM_CAP_GB`` in bytes, or ``None`` when unset or blank.

    Raises:
        ValueError: if the variable is set to something that is not a positive number.
    """
    cap_gb = os.environ.get(GPU_MEM_CAP_ENV, "").strip()
    if not cap_gb:
        return None
    try:
        cap_bytes = float(cap_gb) * (1024**3)
    except ValueError:
        raise ValueError(f"{GPU_MEM_CAP_ENV} must be a positive number of GiB, got {cap_gb!r}") from None
    if not cap_bytes > 0:  # also rejects nan
        raise ValueError(f"{GPU_MEM_CAP_ENV} must be a positive number of GiB, got {cap_gb!r}")
    return cap_bytes


def apply_gpu_memory_cap() -> float | None:
    """Cap this process's PyTorch allocator at ``VERL_GPU_MEM_CAP_GB`` GiB.

    Used to emulate a smaller card, e.g. running an 80 GiB H100 recipe on a 143 GiB H200, so a run's
    memory envelope is representative. Returns the applied fraction, or ``None`` when the knob is
    unset or not applicable.

    It is an API call inside the process for two reasons:

    * torch has no env-var form: ``PYTORCH_CUDA_ALLOC_CONF`` rejects ``per_process_memory_fraction``.
    * It must not reach the rollout engines. vLLM budgets weights, activations and KV cache as a
      fraction of the device's total memory, so a hidden allocator ceiling would leave it planning
      for memory it cannot get. Cap the rollout side by lowering ``rollout.gpu_memory_utilization``
      to the same absolute budget instead (``fraction_H100 * 80 / device_GiB``, e.g. 0.5 on an
      80 GiB card is about 0.28 on a 143 GiB one). In verl's colocated hybrid engine the vLLM
      server is a separate Ray actor (``vLLMHttpServer``), so capping the actor worker bounds the
      trainer only.

    This bounds the caching allocator, not the hardware. The CUDA context, NCCL buffers and cuBLAS
    workspaces sit outside it (around 1-2 GiB), and fragmentation against a soft cap differs from a
    real wall. So fitting under the cap is evidence that the recipe fits the smaller card, not proof.

    Raises:
        ValueError: if the variable is set to something that is not a positive number.
    """
    cap_bytes = _read_cap_bytes()
    if cap_bytes is None:
        return None
    cap_gb = os.environ[GPU_MEM_CAP_ENV].strip()
    device = get_torch_device()
    if not device.is_available():
        return None
    total_bytes = device.get_device_properties(device.current_device()).total_memory
    if cap_bytes >= total_bytes:
        logger.warning(
            "%s=%s is at or above the device's %.1f GiB; leaving the allocator uncapped",
            GPU_MEM_CAP_ENV,
            cap_gb,
            total_bytes / 1024**3,
        )
        return None
    fraction = cap_bytes / total_bytes
    device.set_per_process_memory_fraction(fraction)
    logger.warning(
        "Capped the trainer allocator at %.1f GiB of %.1f GiB (fraction %.4f) via %s",
        cap_bytes / 1024**3,
        total_bytes / 1024**3,
        fraction,
        GPU_MEM_CAP_ENV,
    )
    return fraction


def rollout_allocator_fraction(total_bytes: int, gpu_memory_utilization: float) -> float | None:
    """The allocator fraction for a standalone rollout worker: ``cap / total``.

    ``None`` when the cap is unset, when the fraction reaches the whole device (nothing to cap), or when it is
    not above ``gpu_memory_utilization``: vLLM plans weights, activations and KV cache as
    ``gpu_memory_utilization * total``, so a lower ceiling would leave it planning for memory it cannot get.

    Raises:
        ValueError: on an invalid ``VERL_GPU_MEM_CAP_GB``.
    """
    cap_bytes = _read_cap_bytes()
    if cap_bytes is None:
        return None
    fraction = cap_bytes / total_bytes
    if fraction >= 1.0:
        logger.warning(
            "%s=%.1f GiB is at or above the device's %.1f GiB; leaving the rollout allocator uncapped",
            GPU_MEM_CAP_ENV,
            cap_bytes / 1024**3,
            total_bytes / 1024**3,
        )
        return None
    if fraction <= gpu_memory_utilization:
        logger.warning(
            "Rollout allocator cap fraction %.4f is not above gpu_memory_utilization=%s (vLLM's own budget); "
            "leaving the rollout allocator uncapped. Lower rollout.gpu_memory_utilization or raise %s",
            fraction,
            gpu_memory_utilization,
            GPU_MEM_CAP_ENV,
        )
        return None
    return fraction


def apply_rollout_gpu_memory_cap(gpu_memory_utilization: float) -> float | None:
    """Cap the current device's allocator for a standalone rollout worker (see ``rollout_allocator_fraction``).

    Call it after the worker has selected its device. Returns the applied fraction, or ``None``.
    """
    if _read_cap_bytes() is None:
        return None
    device = get_torch_device()
    if not device.is_available():
        return None
    index = device.current_device()
    total_bytes = device.get_device_properties(index).total_memory
    fraction = rollout_allocator_fraction(total_bytes, gpu_memory_utilization)
    if fraction is None:
        return None
    device.set_per_process_memory_fraction(fraction, index)
    logger.warning(
        "Capped the rollout allocator at %.1f GiB of %.1f GiB (fraction %.4f) via %s; gpu_memory_utilization=%s",
        fraction * total_bytes / 1024**3,
        total_bytes / 1024**3,
        fraction,
        GPU_MEM_CAP_ENV,
        gpu_memory_utilization,
    )
    return fraction
