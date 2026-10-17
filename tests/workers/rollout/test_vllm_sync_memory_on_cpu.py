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
"""Rollout-side memory release around a weight sync (the CUDA OOM in BucketedWeightSender._init_buffer).

A vLLM worker's caching allocator keeps every block it ever reserved (validation bursts, eager batches above the
captured CUDA-graph sizes, FP8 re-quantization temporaries); gpu_memory_utilization only sized the KV pool at
startup. Before a sync the rollout-side checkpoint worker allocates its transfer buffers on the same GPU, so:

- release_device_cache / vLLMColocateWorkerExtension.release_cached_memory return free blocks and report memory;
- vLLMHttpServer.release_kv_cache (called first in CheckpointEngineManager.update_weights) runs that on every
  worker instead of being a no-op, and the replicas pass the stats up;
- update_weights_from_ipc releases again after the post-load transforms;
- CheckpointEngineManager.update_weights folds the stats into its sync metrics as rollout_mem/*_gib.
No GPU: the device and the engines are fakes.
"""

import asyncio
from types import SimpleNamespace

import pytest

from verl.checkpoint_engine import base as ce_base
from verl.workers.rollout.vllm_rollout import utils as vutils
from verl.workers.rollout.vllm_rollout import vllm_async_server as vserver

GIB = 1 << 30


class _FakeDevice:
    """torch.cuda stand-in: a reservation that empty_cache shrinks to the allocated bytes."""

    def __init__(self, reserved_gib=90.0, allocated_gib=70.0, total_gib=139.8):
        self.calls = []
        self.reserved = reserved_gib * GIB
        self.allocated = allocated_gib * GIB
        self.peak = (allocated_gib + 5.0) * GIB
        self.total = total_gib * GIB
        self.others = 6.0 * GIB  # other processes on the GPU (e.g. the checkpoint worker)

    def synchronize(self):
        self.calls.append("synchronize")

    def memory_reserved(self):
        return self.reserved

    def memory_allocated(self):
        return self.allocated

    def max_memory_allocated(self):
        return self.peak

    def ipc_collect(self):
        self.calls.append("ipc_collect")

    def empty_cache(self):
        self.calls.append("empty_cache")
        self.reserved = self.allocated

    def mem_get_info(self):
        return self.total - self.reserved - self.others, self.total

    def reset_peak_memory_stats(self):
        self.calls.append("reset_peak")
        self.peak = self.allocated


@pytest.fixture
def device(monkeypatch):
    dev = _FakeDevice()
    monkeypatch.setattr(vutils, "get_torch_device", lambda: dev)
    return dev


# ------------------------------------------------------------------ worker side


def test_release_device_cache_with_stats(device):
    stats = vutils.release_device_cache(collect_stats=True)
    assert device.calls == ["synchronize", "ipc_collect", "empty_cache", "reset_peak"]
    assert stats["reserved_before"] == pytest.approx(90.0)
    assert stats["reserved_after"] == pytest.approx(70.0)  # the 20 GiB of free blocks went back to the driver
    assert stats["allocated"] == pytest.approx(70.0)
    assert stats["max_allocated"] == pytest.approx(75.0)  # peak since the previous call, then reset
    assert stats["device_used"] == pytest.approx(76.0)  # this process after the release + the other process
    assert device.peak == device.allocated


def test_release_device_cache_without_stats_only_releases(device):
    assert vutils.release_device_cache() == {}
    assert device.calls == ["synchronize", "ipc_collect", "empty_cache"]
    assert device.reserved == device.allocated


def test_release_without_ipc_collect(monkeypatch):
    class NoIpcDevice:  # e.g. a device module without CUDA IPC
        def __init__(self):
            self.calls = []

        def synchronize(self):
            self.calls.append("synchronize")

        def empty_cache(self):
            self.calls.append("empty_cache")

    dev = NoIpcDevice()
    monkeypatch.setattr(vutils, "get_torch_device", lambda: dev)
    assert vutils.release_device_cache() == {}
    assert dev.calls == ["synchronize", "empty_cache"]


def test_worker_extension_rpc_returns_the_stats(device):
    stats = vutils.vLLMColocateWorkerExtension.release_cached_memory(None)
    assert set(stats) == {"reserved_before", "reserved_after", "allocated", "max_allocated", "device_used"}
    assert "empty_cache" in device.calls


# ------------------------------------------------------------------ server / replica


class _Engine:
    def __init__(self, per_worker):
        self.per_worker = per_worker
        self.rpcs = []

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.rpcs.append(method)
        return list(self.per_worker)


@pytest.mark.parametrize("free_cache_engine", [True, False])
def test_server_release_kv_cache_runs_the_worker_rpc(free_cache_engine):
    engine = _Engine([{"reserved_before": 90.0}, {"reserved_before": 80.0}])
    server = SimpleNamespace(node_rank=0, engine=engine, config=SimpleNamespace(free_cache_engine=free_cache_engine))
    out = asyncio.run(vserver.vLLMHttpServer.release_kv_cache(server))
    assert engine.rpcs == ["release_cached_memory"]
    assert out == [{"reserved_before": 90.0}, {"reserved_before": 80.0}]


def test_server_release_kv_cache_is_a_no_op_off_rank_zero():
    engine = _Engine([{"x": 1.0}])
    server = SimpleNamespace(node_rank=1, engine=engine, config=SimpleNamespace(free_cache_engine=True))
    assert asyncio.run(vserver.vLLMHttpServer.release_kv_cache(server)) is None
    assert engine.rpcs == []


class _Remote:
    def __init__(self, events, name, result=None):
        self.events, self.name, self.result = events, name, result

    def remote(self, *a, **k):
        self.events.append(self.name)

        async def done():
            return self.result

        return done()


def test_vllm_replica_drains_then_returns_every_server_s_stats():
    events = []
    servers = [
        SimpleNamespace(
            wait_for_requests_to_drain=_Remote(events, "drain"),
            release_kv_cache=_Remote(events, f"release{i}", [{"reserved_before": 10.0 * (i + 1)}]),
        )
        for i in range(2)
    ]
    replica = SimpleNamespace(servers=servers)
    out = asyncio.run(vserver.vLLMReplica.release_kv_cache(replica))
    assert events[0] == "drain" and set(events[1:]) == {"release0", "release1"}
    assert out == [[{"reserved_before": 10.0}], [{"reserved_before": 20.0}]]


def test_generic_replica_returns_the_servers_results():
    from verl.workers.rollout.replica import RolloutReplica

    events = []
    servers = [SimpleNamespace(release_kv_cache=_Remote(events, "release", None)) for _ in range(3)]
    out = asyncio.run(RolloutReplica.release_kv_cache(SimpleNamespace(servers=servers)))
    assert out == [None, None, None] and events == ["release"] * 3


# ------------------------------------------------------------------ checkpoint manager


def test_summarize_rollout_memory_takes_the_max_over_workers():
    results = [
        [[{"reserved_before": 90.0, "reserved_after": 70.0, "device_used": 77.0}, {"reserved_before": 120.0}]],
        [None],  # e.g. a non-rank-0 server
        None,  # e.g. an sglang / trtllm replica
        [[{"reserved_before": 100.0, "reserved_after": 71.0, "note": "x"}]],
    ]
    assert ce_base.summarize_rollout_memory(results) == {
        "rollout_mem/reserved_before_gib": 120.0,
        "rollout_mem/reserved_after_gib": 71.0,
        "rollout_mem/device_used_gib": 77.0,
    }
    assert ce_base.summarize_rollout_memory(None) == {}
    assert ce_base.summarize_rollout_memory([[None]]) == {}


def test_release_kv_cache_replicas_returns_the_replicas_results():
    class Replica:
        def __init__(self, stats):
            self.stats = stats

        async def release_kv_cache(self):
            return self.stats

    mgr = ce_base.CheckpointEngineManager.__new__(ce_base.CheckpointEngineManager)
    mgr.replicas = [Replica([[{"reserved_before": 1.0}]]), Replica(None)]

    async def run():
        return await mgr.release_kv_cache_replicas()

    assert asyncio.run(run()) == [[[{"reserved_before": 1.0}]], None]


def test_update_weights_releases_first_and_reports_rollout_memory(monkeypatch):
    order = []

    class Replica:
        workers = ["w0"]

        async def abort_all_requests(self):
            order.append("abort")

        async def release_kv_cache(self):
            order.append("release")
            return [[{"reserved_before": 88.0, "reserved_after": 70.0}]]

        async def resume_kv_cache(self):
            order.append("resume_kv")

        async def resume_generation(self):
            order.append("resume_gen")

    class WG:
        world_size = 1

        def __init__(self, name):
            self.name = name

        def update_weights(self, **kw):
            order.append(f"{self.name}.update_weights")
            return [{"sync/changed_ratio": 0.5}] if self.name == "actor" else [None]

        def execute_checkpoint_engine(self, cmds):
            order.append(f"{self.name}.finalize")
            return []

    monkeypatch.setattr(ce_base, "RayWorkerGroup", lambda **kw: WG("rollout"))
    monkeypatch.setattr(ce_base, "RayClassWithInitArgs", lambda **kw: None)
    monkeypatch.setattr(ce_base.ray, "get", lambda refs: list(refs))
    mgr = ce_base.CheckpointEngineManager.__new__(ce_base.CheckpointEngineManager)
    mgr.backend = "nccl"
    mgr.replicas = [Replica()]
    mgr.actor_wg = WG("actor")
    mgr.build_process_group = lambda rollout: order.append("build_pg")

    async def run():
        return await mgr.update_weights(global_steps=3)

    metrics = asyncio.run(run())
    # the release happens after the abort and BEFORE any transfer buffer is built or weights are sent
    assert order.index("abort") < order.index("release") < order.index("build_pg") < order.index("actor.update_weights")
    assert metrics == {
        "sync/changed_ratio": 0.5,
        "rollout_mem/reserved_before_gib": 88.0,
        "rollout_mem/reserved_after_gib": 70.0,
    }


# ------------------------------------------------------------------ after the post-load transforms


def test_update_weights_from_ipc_releases_after_the_quantized_post_processing(monkeypatch):
    import verl.utils.vllm.vllm_quant_utils as quant_utils
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer as bwt

    order = []

    class Receiver:
        def __init__(self, **kw):
            pass

        def receive_weights(self, on_bucket_received):
            order.append("receive")

    monkeypatch.setattr(bwt, "BucketedWeightReceiver", Receiver)
    monkeypatch.setattr(vutils, "is_refit_quant_model", lambda cfg: True)
    monkeypatch.setattr(quant_utils, "prepare_quanted_weights_for_loading", lambda m: order.append("prepare") or "s")
    monkeypatch.setattr(
        quant_utils, "process_quanted_weights_after_loading", lambda m, s: order.append("requantize")
    )
    monkeypatch.setattr(vutils, "release_device_cache", lambda collect_stats=False: order.append("release") or {})
    worker = SimpleNamespace(
        device="cpu",
        _is_qat_model=False,
        _is_modelopt_qat=False,
        model_runner=SimpleNamespace(vllm_config=None),
        _iter_all_models=lambda: iter(["model"]),
        _get_zmq_handle=lambda: "ipc:///tmp/x",
    )
    vutils.vLLMColocateWorkerExtension.update_weights_from_ipc(worker)
    assert order == ["prepare", "receive", "requantize", "release"]
