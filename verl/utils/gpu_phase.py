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
"""Per-GPU phase monitoring via a Ray custom metric.

Publishes a single ``ray.util.metrics.Gauge`` named ``verl_gpu_phase`` whose value is an integer
**phase code** telling what the worker owning a given GPU is currently doing (idle / generating /
forward log-prob / training / weight-sync / sleeping). One time series per GPU, tagged so it lines
up with Ray's built-in ``ray_node_gpus_utilization{GpuIndex}`` (the ``gpu`` tag equals Ray's
``GpuIndex``), giving a per-GPU phase + utilization view in Grafana with no manual mapping.

This is the worker-side half of the monitoring design (the rollout ``gen``/``sleep`` phases are
emitted from the rollout replica; the trainer phases ride the ``register()`` decorator). It is a
no-op unless ``VERL_GPU_PHASE_MONITOR=1`` and ``ray.util.metrics`` is importable, so the hot
worker-dispatch path costs one cheap boolean check when disabled.
"""

import logging
import os

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ---------------------------------------------------------------------------
# Phase codes (gauge value). Keep in sync with the Grafana dashboard legend.
# ---------------------------------------------------------------------------
IDLE = 0
GEN = 1
LOGPROB_FWD = 2
REF_FWD = 3
VALUES_FWD = 4
UPDATE_ACTOR = 5
UPDATE_CRITIC = 6
WEIGHT_SYNC = 7
SLEEP = 8

PHASE_NAMES = {
    IDLE: "idle",
    GEN: "gen",
    LOGPROB_FWD: "logprob_fwd",
    REF_FWD: "ref_fwd",
    VALUES_FWD: "values_fwd",
    UPDATE_ACTOR: "update_actor",
    UPDATE_CRITIC: "update_critic",
    WEIGHT_SYNC: "weight_sync",
    SLEEP: "sleep",
}

# Registered worker-method name -> phase code. Methods absent here are not phases and are ignored,
# so the marker only fires for the handful of GPU-heavy entry points (shared across all trainers).
METHOD_PHASE = {
    "compute_log_prob": LOGPROB_FWD,
    "compute_ref_log_prob": REF_FWD,
    "compute_values": VALUES_FWD,
    "update_actor": UPDATE_ACTOR,
    "update_critic": UPDATE_CRITIC,
    "update_weights": WEIGHT_SYNC,
}

_ENV_FLAG = "VERL_GPU_PHASE_MONITOR"

# Lazily-initialized Gauge handle. ``False`` means "tried and unavailable" (distinct from ``None``
# = "not yet tried") so we only attempt the import/registration once.
_GAUGE = None


def enabled() -> bool:
    """True if per-GPU phase monitoring is switched on via the environment flag."""
    return os.environ.get(_ENV_FLAG, "0") not in ("0", "", "false", "False")


def _gauge():
    """Return the shared Gauge, creating it on first use; ``None`` if metrics are unavailable."""
    global _GAUGE
    if _GAUGE is None:
        try:
            from ray.util.metrics import Gauge

            _GAUGE = Gauge(
                "verl_gpu_phase",
                description="Current verl phase per GPU (see PHASE_NAMES for the code legend).",
                tag_keys=("rank", "gpu", "node", "role"),
            )
        except Exception as e:  # ray not initialized / metrics export disabled
            logger.warning("verl_gpu_phase metric unavailable, GPU phase monitoring disabled: %s", e)
            _GAUGE = False
    return _GAUGE or None


def phase_tags(worker) -> dict | None:
    """Build & cache the metric tags ``{rank, gpu, node, role}`` for a worker instance.

    ``gpu`` is the physical accelerator id Ray assigned to this worker, matching the ``GpuIndex``
    label on ``ray_node_gpus_utilization`` so the phase series aligns per-GPU. Cached on the worker
    to keep the dispatch hot-path cheap.
    """
    tags = getattr(worker, "_gpu_phase_tags", None)
    if tags is not None:
        return tags
    try:
        import ray

        from verl.utils.device import get_resource_name

        ctx = ray.get_runtime_context()
        accel_key = get_resource_name()  # Ray accelerator resource key, e.g. "GPU" / "NPU"
        ids = ctx.get_accelerator_ids().get(accel_key, [])
        gpu = str(ids[0]) if ids else os.environ.get("CUDA_VISIBLE_DEVICES", "")
        tags = {
            "rank": str(getattr(worker, "_rank", os.environ.get("RANK", "0"))),
            "gpu": gpu,
            "node": ctx.get_node_id(),
            "role": type(worker).__name__,
        }
    except Exception as e:
        logger.warning("could not build GPU phase tags: %s", e)
        tags = False
    try:
        worker._gpu_phase_tags = tags
    except Exception:
        pass
    return tags or None


def set_phase(worker, code: int) -> None:
    """Set the current phase code for the GPU owned by ``worker`` (no-op if disabled/unavailable)."""
    if not enabled():
        return
    g = _gauge()
    if g is None:
        return
    tags = phase_tags(worker)
    if tags is None:
        return
    try:
        g.set(code, tags=tags)
    except Exception as e:
        logger.warning("failed to set verl_gpu_phase: %s", e)


_NODE_ID = None


def _node_id() -> str:
    global _NODE_ID
    if _NODE_ID is None:
        try:
            import ray

            _NODE_ID = ray.get_runtime_context().get_node_id()
        except Exception:
            _NODE_ID = ""
    return _NODE_ID


def set_phase_gpus(gpu_ids, code: int, *, role: str, rank=None) -> None:
    """Set the phase code for a set of physical GPU ids (for non-``Worker`` emitters, e.g. the
    rollout server which may span several GPUs via tensor-parallelism).

    Emits one ``verl_gpu_phase`` series per gpu id so each lines up with the matching
    ``ray_node_gpus_utilization{GpuIndex}``. No-op if disabled/unavailable.
    """
    if not enabled():
        return
    g = _gauge()
    if g is None:
        return
    node = _node_id()
    for gid in gpu_ids:
        gid = str(gid).strip()
        if not gid:
            continue
        try:
            g.set(code, tags={"rank": str(rank) if rank is not None else gid, "gpu": gid, "node": node, "role": role})
        except Exception as e:
            logger.warning("failed to set verl_gpu_phase for gpu %s: %s", gid, e)
