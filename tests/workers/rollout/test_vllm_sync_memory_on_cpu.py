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
    stats = object.__new__(vutils.vLLMColocateWorkerExtension).release_cached_memory()
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


# --- rollout_mem/* reaching the fully-async trainer's logged metrics ---


def test_rollout_memory_metrics_keeps_only_rollout_mem_keys():
    metrics = {"rollout_mem/reserved_before_gib": 99.0, "timing/x": 1.0, "rollout_mem/allocated_gib": 40.0}
    assert ce_base.rollout_memory_metrics(metrics) == {
        "rollout_mem/reserved_before_gib": 99.0,
        "rollout_mem/allocated_gib": 40.0,
    }
    assert ce_base.rollout_memory_metrics(None) == {}
    assert ce_base.rollout_memory_metrics({}) == {}


def _trainer_class(fat):
    actor = fat.FullyAsyncTrainer
    return actor.__ray_metadata__.modified_class if hasattr(actor, "__ray_metadata__") else actor


def test_fully_async_trainer_logs_rollout_memory_with_the_cycle(monkeypatch):
    from verl.experimental.fully_async_policy import fully_async_trainer as fat

    logged = []
    fake = SimpleNamespace(
        metrics_aggregator=SimpleNamespace(
            get_aggregated_metrics=lambda rollout_resource_utilization=None: {"actor/entropy": 0.5},
            reset=lambda: None,
        ),
        _add_cumulative_time_metrics=lambda metrics, state: None,
        rollouter=SimpleNamespace(get_timing_state=SimpleNamespace(remote=lambda: "ref")),
        logger=SimpleNamespace(log=lambda data, step: logged.append((step, dict(data)))),
        current_param_version=7,
    )
    monkeypatch.setattr(fat.ray, "get", lambda ref: {})
    reset_timing = {"dynamic_resource/rollout_resource_utilization": 0.9, "rollout_mem/reserved_before_gib": 98.5}
    _trainer_class(fat)._fit_log_aggregated_training_metrics(fake, reset_timing)
    assert logged == [(7, {"actor/entropy": 0.5, "rollout_mem/reserved_before_gib": 98.5})]


def test_fit_update_weights_folds_the_sync_metrics_into_its_return():
    """Source-level guard: the update_weights() return must not be dropped again."""
    import inspect

    from verl.experimental.fully_async_policy import fully_async_trainer as fat

    src = inspect.getsource(_trainer_class(fat)._fit_update_weights)
    assert "sync_metrics = await self.checkpoint_manager.update_weights(" in src
    assert "rollout_memory_metrics(sync_metrics)" in src


# --- opt-in allocator snapshot (VERL_ROLLOUT_MEM_SNAPSHOT_DIR) ---


class _FakeCudaMemory:
    def __init__(self):
        self.calls = []

    def _record_memory_history(self, enabled="all", **kwargs):
        self.calls.append(("record", enabled, kwargs.get("stacks")))
        self.max_entries = kwargs.get("max_entries", self.__dict__.get("max_entries"))

    def _dump_snapshot(self, path):
        self.calls.append(("dump", path))
        with open(path, "wb") as f:
            f.write(b"snap")


@pytest.fixture
def cuda_memory(monkeypatch):
    mem = _FakeCudaMemory()
    monkeypatch.setattr(vutils.torch.cuda, "memory", mem)
    return mem


def test_snapshotter_is_off_without_the_env(monkeypatch, cuda_memory):
    monkeypatch.delenv(vutils.MEM_SNAPSHOT_DIR_ENV, raising=False)
    snap = vutils.RolloutMemorySnapshotter()
    assert not snap.enabled
    assert snap.on_release(100.0) is None
    assert cuda_memory.calls == []


def test_snapshotter_records_from_the_first_sync_and_dumps_later_ones(monkeypatch, cuda_memory, tmp_path):
    monkeypatch.setenv(vutils.MEM_SNAPSHOT_DIR_ENV, str(tmp_path / "snaps"))
    monkeypatch.setenv(vutils.MEM_SNAPSHOT_MIN_GIB_ENV, "60")
    monkeypatch.setenv(vutils.MEM_SNAPSHOT_MAX_ENV, "2")
    snap = vutils.RolloutMemorySnapshotter()
    assert snap.on_release(47.0) is None  # first call: start recording only
    assert cuda_memory.calls == [("record", "all", "python")]
    assert cuda_memory.max_entries == 1_000_000
    assert snap.on_release(50.0) is None  # below the threshold: no dump
    first = snap.on_release(99.0)
    assert first.endswith("_sync3.pickle") and (tmp_path / "snaps").is_dir()
    second = snap.on_release(80.0)
    assert second.endswith("_sync4.pickle")
    assert cuda_memory.calls[-1] == ("record", None, None)  # max reached: recording stopped
    assert not snap.enabled
    assert snap.on_release(120.0) is None
    assert [c[0] for c in cuda_memory.calls] == ["record", "dump", "dump", "record"]


def test_release_cached_memory_snapshots_before_releasing(monkeypatch, device, cuda_memory, tmp_path):
    monkeypatch.setenv(vutils.MEM_SNAPSHOT_DIR_ENV, str(tmp_path))
    order = []
    real_release = vutils.release_device_cache
    monkeypatch.setattr(
        vutils.RolloutMemorySnapshotter, "on_release", lambda self, gib: order.append(("snap", round(gib)))
    )
    monkeypatch.setattr(
        vutils, "release_device_cache", lambda collect_stats=False: order.append("release") or real_release(True)
    )
    ext = object.__new__(vutils.vLLMColocateWorkerExtension)
    stats = ext.release_cached_memory()
    assert order == [("snap", 90), "release"]
    assert stats["reserved_before"] == pytest.approx(90.0)
    ext.release_cached_memory()
    assert order[2:] == [("snap", round(device.reserved / GIB)), "release"]  # one snapshotter per worker


def test_ppo_runtime_env_forwards_the_snapshot_env(monkeypatch):
    from verl.trainer import constants_ppo

    monkeypatch.setenv("VERL_ROLLOUT_MEM_SNAPSHOT_DIR", "/tmp/snaps")
    monkeypatch.setenv("VERL_ROLLOUT_MEM_SNAPSHOT_MIN_GIB", "60")
    env = constants_ppo.get_ppo_ray_runtime_env()["env_vars"]
    assert env["VERL_ROLLOUT_MEM_SNAPSHOT_DIR"] == "/tmp/snaps"
    assert env["VERL_ROLLOUT_MEM_SNAPSHOT_MIN_GIB"] == "60"


def test_snapshotter_trace_length_is_configurable(monkeypatch, cuda_memory, tmp_path):
    monkeypatch.setenv(vutils.MEM_SNAPSHOT_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(vutils.MEM_SNAPSHOT_ENTRIES_ENV, "5000")
    vutils.RolloutMemorySnapshotter().on_release(10.0)
    assert cuda_memory.max_entries == 5000


# --- standalone rollout allocator cap (VERL_GPU_MEM_CAP_GB) ---


@pytest.mark.parametrize(
    "mode, expected",
    [("STANDALONE", "1"), ("HYBRID", None), ("COLOCATED", None)],
)
def test_allocator_cap_env_marks_standalone_servers_only(mode, expected):
    environ = {vserver.ALLOCATOR_CAP_ENV: "1"}  # stale mark from a previous server in this process
    vserver._allocator_cap_env(getattr(vserver.RolloutMode, mode), environ)
    assert environ.get(vserver.ALLOCATOR_CAP_ENV) == expected
    assert vserver.ALLOCATOR_CAP_ENV == vutils.ROLLOUT_ALLOCATOR_CAP_ENV


def test_server_init_marks_the_workers():
    import inspect

    src = inspect.getsource(vserver.vLLMHttpServer.__init__)
    assert "_allocator_cap_env(rollout_mode, os.environ)" in src


def _cap_config(util=0.5):
    return SimpleNamespace(cache_config=SimpleNamespace(gpu_memory_utilization=util))


def test_maybe_apply_rollout_memory_cap_follows_the_mark(monkeypatch):
    calls = []
    monkeypatch.setattr(vutils, "apply_rollout_gpu_memory_cap", lambda util: calls.append(util) or 0.586)
    monkeypatch.delenv(vutils.ROLLOUT_ALLOCATOR_CAP_ENV, raising=False)
    assert vutils.maybe_apply_rollout_memory_cap(_cap_config()) is None
    monkeypatch.setenv(vutils.ROLLOUT_ALLOCATOR_CAP_ENV, "0")
    assert vutils.maybe_apply_rollout_memory_cap(_cap_config()) is None
    assert calls == []
    monkeypatch.setenv(vutils.ROLLOUT_ALLOCATOR_CAP_ENV, "1")
    assert vutils.maybe_apply_rollout_memory_cap(_cap_config(0.45)) == 0.586
    assert calls == [0.45]


class _Worker:
    def __init__(self, order):
        self.order = order

    def init_device(self, *args, **kwargs):
        self.order.append(("init_device", args, kwargs))
        return "device-ready"


def test_wrapped_init_device_caps_right_after_selecting_the_gpu(monkeypatch):
    order = []
    monkeypatch.setenv(vutils.ROLLOUT_ALLOCATOR_CAP_ENV, "1")
    monkeypatch.setattr(vutils, "apply_rollout_gpu_memory_cap", lambda util: order.append(("cap", util)) or 0.586)
    worker = _Worker(order)
    assert vutils.wrap_init_device_with_memory_cap(worker, _cap_config(0.5)) is True
    assert order == []  # nothing until vLLM calls init_device
    assert worker.init_device(1, k=2) == "device-ready"
    assert order == [("init_device", (1,), {"k": 2}), ("cap", 0.5)]


@pytest.mark.parametrize("mark, config", [(None, _cap_config()), ("0", _cap_config()), ("1", None)])
def test_init_device_is_untouched_without_the_mark_or_config(monkeypatch, mark, config):
    if mark is None:
        monkeypatch.delenv(vutils.ROLLOUT_ALLOCATOR_CAP_ENV, raising=False)
    else:
        monkeypatch.setenv(vutils.ROLLOUT_ALLOCATOR_CAP_ENV, mark)
    worker = _Worker([])
    assert vutils.wrap_init_device_with_memory_cap(worker, config) is False
    assert "init_device" not in vars(worker)


def test_worker_extension_new_wraps_init_device():
    import inspect

    src = inspect.getsource(vutils.vLLMColocateWorkerExtension.__new__)
    assert "wrap_init_device_with_memory_cap(instance, vllm_config)" in src


def test_ppo_runtime_env_forwards_the_cap_env(monkeypatch):
    from verl.trainer import constants_ppo

    monkeypatch.delenv("VERL_GPU_MEM_CAP_GB", raising=False)
    env = constants_ppo.get_ppo_ray_runtime_env()["env_vars"]
    assert "VERL_GPU_MEM_CAP_GB" not in env
    monkeypatch.setenv("VERL_GPU_MEM_CAP_GB", "78")
    env = constants_ppo.get_ppo_ray_runtime_env()["env_vars"]
    assert env["VERL_GPU_MEM_CAP_GB"] == "78"
