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
"""CPU tests for the online vLLM metrics monitor (verl/workers/rollout/vllm_metrics.py) and its sync-trainer wiring.

The replay rollouter's wiring is tested in tests/experimental/fully_async_policy/test_rollouter_replay_on_cpu.py.
The v1 trainer's wiring (trainer.use_v1=true, which the sync baselines use) is tested in
tests/trainer/ppo/v1/test_vllm_metrics_wiring_on_cpu.py; TestSyncTrainerWiring below covers the v0 RayPPOTrainer.
"""

import importlib.util
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from omegaconf import OmegaConf

from verl.workers.rollout import vllm_metrics
from verl.workers.rollout.vllm_metrics import (
    INTERVAL_ENV,
    METRIC_FAMILIES,
    VllmMetricsMonitor,
    VllmMetricsWindow,
    maybe_create_vllm_metrics_monitor,
    parse_vllm_metrics,
)

# A /metrics payload in vLLM 0.24's exposition format: two data-parallel engines, counters with _total and _created,
# histograms with buckets, and families the monitor does not log.
PAYLOAD = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen3-8B"} 40.0
vllm:num_requests_running{engine="1",model_name="Qwen/Qwen3-8B"} 20.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="Qwen/Qwen3-8B"} 100.0
vllm:num_requests_waiting{engine="1",model_name="Qwen/Qwen3-8B"} 0.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen/Qwen3-8B"} 0.9
vllm:kv_cache_usage_perc{engine="1",model_name="Qwen/Qwen3-8B"} 0.5
# HELP vllm:num_preemptions_total Cumulative number of preemption from the engine.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="Qwen/Qwen3-8B"} 3.0
vllm:num_preemptions_total{engine="1",model_name="Qwen/Qwen3-8B"} 1.0
# HELP vllm:num_preemptions_created Cumulative number of preemption from the engine.
# TYPE vllm:num_preemptions_created gauge
vllm:num_preemptions_created{engine="0",model_name="Qwen/Qwen3-8B"} 1.7600000e+09
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="Qwen/Qwen3-8B"} 5000.0
vllm:generation_tokens_total{engine="1",model_name="Qwen/Qwen3-8B"} 1000.0
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{engine="0",model_name="Qwen/Qwen3-8B"} 12.5
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1",model_name="Qwen/Qwen3-8B"} 2.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="Qwen/Qwen3-8B"} 10.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="Qwen/Qwen3-8B"} 10.0
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",cache_dtype="fp8_e4m3",engine="0"} 1.0
# HELP http_requests_total Total number of requests by method, status and handler.
# TYPE http_requests_total counter
http_requests_total{handler="/v1/completions",method="POST",status="2xx"} 7.0
"""


def snap(running=0, waiting=0, kv=0.0, preempt=0, prompt=0, gen=0, success=0, pq=0, ph=0, ttft=(0.0, 0)):
    """One server's parsed payload."""
    return {
        "vllm:num_requests_running": float(running),
        "vllm:num_requests_waiting": float(waiting),
        "vllm:kv_cache_usage_perc": kv,
        "vllm:num_preemptions": float(preempt),
        "vllm:prompt_tokens": float(prompt),
        "vllm:generation_tokens": float(gen),
        "vllm:request_success": float(success),
        "vllm:prefix_cache_queries": float(pq),
        "vllm:prefix_cache_hits": float(ph),
        "vllm:time_to_first_token_seconds_sum": ttft[0],
        "vllm:time_to_first_token_seconds_count": float(ttft[1]),
    }


HEALTH_KEYS = {"vllm/samples", "vllm/scrape_errors"}


# ------------------------------------------------------------------ parser


class TestParse:
    def test_values_are_summed_over_engines_and_kv_usage_is_averaged(self):
        m = parse_vllm_metrics(PAYLOAD)
        assert m == {
            "vllm:num_requests_running": 60.0,
            "vllm:num_requests_waiting": 100.0,
            "vllm:kv_cache_usage_perc": pytest.approx(0.7),
            "vllm:num_preemptions": 4.0,
            "vllm:generation_tokens": 6000.0,
            "vllm:time_to_first_token_seconds_sum": 12.5,
            "vllm:time_to_first_token_seconds_count": 10.0,
        }

    def test_created_buckets_and_unlogged_families_are_ignored(self):
        m = parse_vllm_metrics(PAYLOAD)
        assert not any("created" in k or "bucket" in k or "cache_config" in k or "http" in k for k in m)

    def test_empty_payload(self):
        assert parse_vllm_metrics("") == {}


def test_metric_names_exist_in_installed_vllm():
    """Guards against vLLM renaming a family: read its loggers.py as text, without importing vLLM (which needs a
    GPU driver)."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("vllm is not installed")
    path = os.path.join(list(spec.submodule_search_locations)[0], "v1", "metrics", "loggers.py")
    source = open(path, encoding="utf-8").read()
    missing = [name for name in METRIC_FAMILIES if f'"{name}"' not in source]
    assert not missing, f"metric families not defined by the installed vLLM: {missing}"


# ------------------------------------------------------------------ window


class TestWindow:
    def test_gauges_use_active_samples_only(self):
        w = VllmMetricsWindow()
        w.rebase({"a": snap(), "b": snap()})
        w.add_round({"a": snap(running=10, kv=0.4), "b": snap(running=30, waiting=4, kv=0.8)}, 0, 5.0, 0.01)
        w.add_round({"a": snap(running=20, kv=0.6), "b": snap(running=50, kv=1.0)}, 0, 5.0, 0.01)
        w.add_round({"a": snap(), "b": snap()}, 0, 5.0, 0.01)  # idle: not counted
        m = w.summarize()
        assert m["vllm/samples"] == 2
        assert m["vllm/num_requests_running/mean"] == pytest.approx((20 + 35) / 2)  # per-server average
        assert m["vllm/num_requests_running/max"] == 50
        assert m["vllm/num_requests_waiting/mean"] == pytest.approx(1.0)
        assert m["vllm/num_requests_waiting/max"] == 4
        assert m["vllm/kv_cache_usage/mean"] == pytest.approx((0.6 + 0.8) / 2)
        assert m["vllm/kv_cache_usage/max"] == pytest.approx(1.0)

    def test_counter_deltas_histogram_means_and_rates(self):
        w = VllmMetricsWindow()
        w.rebase({"a": snap(preempt=5, prompt=100, gen=1000, success=2, pq=10, ph=5, ttft=(1.0, 2))})
        w.add_round(
            {"a": snap(running=8, preempt=7, prompt=300, gen=3000, success=6, pq=30, ph=20, ttft=(5.0, 6))},
            0,
            10.0,
            0.01,
        )
        m = w.summarize()
        assert m["vllm/num_preemptions"] == 2
        assert m["vllm/prompt_tokens"] == 200
        assert m["vllm/generation_tokens"] == 2000
        assert m["vllm/requests_finished"] == 4
        assert m["vllm/prefix_cache_hit_rate"] == pytest.approx(15 / 20)
        assert m["vllm/ttft_s/mean"] == pytest.approx(4.0 / 4)
        assert m["vllm/generation_throughput"] == pytest.approx(2000 / 10.0)
        assert m["vllm/scrape_time_s/mean"] == pytest.approx(0.01)

    def test_counters_are_summed_over_servers(self):
        w = VllmMetricsWindow()
        w.rebase({"a": snap(gen=10), "b": snap(gen=20)})
        w.add_round({"a": snap(running=1, gen=110), "b": snap(running=1, gen=220)}, 0, 1.0, 0.0)
        assert w.summarize()["vllm/generation_tokens"] == 300

    def test_counter_reset_counts_from_zero(self):
        w = VllmMetricsWindow()
        w.rebase({"a": snap(gen=5000)})
        w.add_round({"a": snap(running=1, gen=40)}, 0, 1.0, 0.0)  # server restarted
        assert w.summarize()["vllm/generation_tokens"] == 40

    def test_first_sight_of_a_server_only_sets_its_baseline(self):
        w = VllmMetricsWindow()
        w.add_round({"new": snap(running=1, gen=999)}, 0, 1.0, 0.0)
        assert w.summarize()["vllm/generation_tokens"] == 0

    def test_tail_round_with_tokens_but_no_running_requests_counts_as_active_time(self):
        w = VllmMetricsWindow()
        w.rebase({"a": snap(gen=0)})
        w.add_round({"a": snap(running=4, gen=100)}, 0, 5.0, 0.0)
        w.add_round({"a": snap(gen=150)}, 0, 1.0, 0.0)  # last requests finished between the two rounds
        m = w.summarize()
        assert m["vllm/samples"] == 1
        assert m["vllm/generation_throughput"] == pytest.approx(150 / 6.0)

    def test_empty_window_has_only_health_keys(self):
        w = VllmMetricsWindow()
        assert set(w.summarize()) == HEALTH_KEYS

    def test_idle_window_has_no_nan_and_no_gauges(self):
        w = VllmMetricsWindow()
        w.rebase({"a": snap()})
        w.add_round({"a": snap()}, 1, 5.0, 0.0)
        m = w.summarize()
        assert m["vllm/scrape_errors"] == 1
        assert not any("mean" in k and "scrape" not in k for k in m)
        assert "vllm/prefix_cache_hit_rate" not in m and "vllm/generation_throughput" not in m
        assert all(not math.isnan(v) for v in m.values())

    def test_reset_keeps_baselines(self):
        w = VllmMetricsWindow()
        w.rebase({"a": snap(gen=0)})
        w.add_round({"a": snap(running=1, gen=100)}, 0, 1.0, 0.0)
        w.reset()
        w.add_round({"a": snap(running=1, gen=130)}, 0, 1.0, 0.0)
        assert w.summarize()["vllm/generation_tokens"] == 30


# ------------------------------------------------------------------ monitor


class FakeServers:
    """fetch() stand-in: per-address payloads, settable from the test; unknown or failing addresses raise."""

    def __init__(self, **payloads):
        self.payloads = payloads
        self.failing = set()
        self.calls = []
        self.lock = threading.Lock()

    def fetch(self, address):
        with self.lock:
            self.calls.append(address)
            if address in self.failing or address not in self.payloads:
                raise ConnectionError(address)
            return dict(self.payloads[address])


def make_monitor(servers, addresses, active=True, interval_s=3600.0):
    """A monitor whose background thread effectively never fires: rounds happen only in pause() / test code."""
    return VllmMetricsMonitor(lambda: list(addresses), interval_s=interval_s, active=active, fetch=servers.fetch)


class TestMonitor:
    def test_resume_takes_baselines_and_pause_closes_the_window(self):
        servers = FakeServers(a=snap(gen=1000))
        mon = make_monitor(servers, ["a"], active=False)
        try:
            servers.payloads["a"] = snap(gen=1500)  # work done before the window: not counted
            mon.resume()
            servers.payloads["a"] = snap(gen=1900)
            mon.pause()
            m = mon.collect()
            assert m["vllm/generation_tokens"] == 400
        finally:
            mon.stop()

    def test_holds_nest_by_reason(self):
        servers = FakeServers(a=snap())
        mon = make_monitor(servers, ["a"])
        try:
            mon.pause("validation")
            mon.pause("save")
            mon.resume("validation")
            assert not mon.active
            mon.resume("save")
            assert mon.active
            mon.resume("save")  # unknown / already released reason: no-op
            assert mon.active
        finally:
            mon.stop()

    def test_paused_work_is_excluded(self):
        servers = FakeServers(a=snap(gen=0))
        mon = make_monitor(servers, ["a"])
        try:
            servers.payloads["a"] = snap(gen=100)
            mon.pause("validation")  # counts 100
            servers.payloads["a"] = snap(gen=10_000)  # validation tokens
            mon.resume("validation")  # re-baselines at 10_000
            servers.payloads["a"] = snap(gen=10_050)
            mon.pause()
            assert mon.collect()["vllm/generation_tokens"] == 150
        finally:
            mon.stop()

    def test_collect_starts_a_new_window_without_losing_counts(self):
        servers = FakeServers(a=snap(gen=0))
        mon = make_monitor(servers, ["a"])
        try:
            servers.payloads["a"] = snap(running=1, gen=70)
            with mon._lock:
                mon._add_round_locked()
            assert mon.collect()["vllm/generation_tokens"] == 70
            servers.payloads["a"] = snap(gen=100)
            mon.pause()
            assert mon.collect()["vllm/generation_tokens"] == 30
            assert set(mon.collect()) == HEALTH_KEYS  # nothing since
        finally:
            mon.stop()

    def test_failed_servers_are_counted_and_skipped(self):
        servers = FakeServers(a=snap(gen=0), b=snap(gen=0))
        mon = make_monitor(servers, ["a", "b"])
        try:
            servers.failing.add("b")
            servers.payloads["a"] = snap(running=2, gen=10)
            mon.pause()
            m = mon.collect()
            assert m["vllm/scrape_errors"] == 1
            assert m["vllm/generation_tokens"] == 10
        finally:
            mon.stop()

    def test_addresses_are_re_read_every_round(self):
        servers = FakeServers(a=snap(), b=snap())
        addresses = ["a", "b"]
        mon = make_monitor(servers, addresses)
        try:
            addresses.remove("b")
            servers.calls.clear()
            mon.pause()
            assert servers.calls == ["a"]
        finally:
            mon.stop()

    def test_background_thread_polls_while_active(self):
        servers = FakeServers(a=snap(running=3, kv=0.5))
        mon = make_monitor(servers, ["a"], interval_s=0.02)
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and len(servers.calls) < 4:
                time.sleep(0.01)
            m = mon.collect()
            assert m["vllm/samples"] >= 2
            assert m["vllm/num_requests_running/mean"] == 3
            mon.pause()
            n = len(servers.calls)
            time.sleep(0.1)
            assert len(servers.calls) == n  # paused: no polling
        finally:
            mon.stop()
        assert not mon._thread.is_alive()


class _MetricsHandler(BaseHTTPRequestHandler):
    payload = PAYLOAD

    def do_GET(self):
        if self.path != "/metrics":
            self.send_error(404)
            return
        body = self.server.payload.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class _ErrorHandler(_MetricsHandler):
    def do_GET(self):
        self.send_error(500)


@pytest.fixture
def http_servers():
    started = []

    def start(handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.payload = PAYLOAD
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        return server, f"127.0.0.1:{server.server_address[1]}"

    yield start
    for server in started:
        server.shutdown()
        server.server_close()


def test_monitor_over_http(http_servers):
    """End to end with the real fetch: a live /metrics endpoint and one returning HTTP 500."""
    good, good_address = http_servers(_MetricsHandler)
    _, bad_address = http_servers(_ErrorHandler)
    mon = VllmMetricsMonitor(lambda: [good_address, bad_address], interval_s=0.05)
    try:
        good.payload = PAYLOAD.replace("5000.0", "5600.0")  # +600 generation tokens on engine 0
        deadline = time.monotonic() + 10.0
        tokens, errors, active = 0.0, 0.0, None
        while time.monotonic() < deadline and not (tokens and active):
            time.sleep(0.1)
            m = mon.collect()
            tokens += m.get("vllm/generation_tokens", 0.0)
            errors += m["vllm/scrape_errors"]
            if m["vllm/samples"] >= 1:
                active = m
        assert tokens == 600
        assert active["vllm/num_requests_running/mean"] == 60
        assert active["vllm/kv_cache_usage/max"] == pytest.approx(0.7)
        assert errors > 0
    finally:
        mon.stop()


# ------------------------------------------------------------------ enable rule


def rollout_cfg(**kw):
    return OmegaConf.create({"name": "vllm", "disable_log_stats": False, **kw})


class TestEnableRule:
    @pytest.mark.parametrize(
        "cfg",
        [rollout_cfg(disable_log_stats=True), rollout_cfg(name="sglang"), OmegaConf.create({"name": "vllm"})],
        ids=["stats-off", "sglang", "default-stats-off"],
    )
    def test_off(self, cfg, monkeypatch):
        monkeypatch.delenv(INTERVAL_ENV, raising=False)
        assert maybe_create_vllm_metrics_monitor(cfg, lambda: []) is None

    def test_interval_zero_disables(self, monkeypatch):
        monkeypatch.setenv(INTERVAL_ENV, "0")
        assert maybe_create_vllm_metrics_monitor(rollout_cfg(), lambda: []) is None

    @pytest.mark.parametrize("raw,expected", [(None, 5.0), ("", 5.0), ("2.5", 2.5)])
    def test_on(self, raw, expected, monkeypatch):
        if raw is None:
            monkeypatch.delenv(INTERVAL_ENV, raising=False)
        else:
            monkeypatch.setenv(INTERVAL_ENV, raw)
        mon = maybe_create_vllm_metrics_monitor(rollout_cfg(), lambda: [], active=False)
        try:
            assert isinstance(mon, VllmMetricsMonitor)
            assert mon._interval_s == expected
            assert not mon.active
        finally:
            mon.stop()

    def test_dataclass_style_config(self, monkeypatch):
        monkeypatch.delenv(INTERVAL_ENV, raising=False)

        class Cfg:
            name = "vllm"
            disable_log_stats = False

        mon = maybe_create_vllm_metrics_monitor(Cfg(), lambda: [])
        try:
            assert mon is not None
        finally:
            mon.stop()

    def test_negative_interval_is_rejected(self, monkeypatch):
        monkeypatch.setenv(INTERVAL_ENV, "-1")
        with pytest.raises(ValueError, match=INTERVAL_ENV):
            vllm_metrics.get_interval_s()


# ------------------------------------------------------------------ sync trainer wiring


class _RecordingMonitor:
    def __init__(self):
        self.events = []
        self.active = False

    def resume(self, reason="default"):
        self.events.append(("resume", reason))
        self.active = True

    def pause(self, reason="default"):
        self.events.append(("pause", reason))
        self.active = False


class TestSyncTrainerWiring:
    @pytest.fixture
    def trainer(self):
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.seen_active = []

        class Manager:
            def generate_sequences(self, batch):
                trainer.seen_active.append(trainer.vllm_metrics_monitor.active)
                if batch == "boom":
                    raise RuntimeError("generation failed")
                return f"out:{batch}"

        trainer.async_rollout_manager = Manager()
        return trainer

    def test_generation_is_bracketed_by_the_window(self, trainer):
        trainer.vllm_metrics_monitor = _RecordingMonitor()
        assert trainer._generate_sequences_with_vllm_metrics("b") == "out:b"
        assert trainer.seen_active == [True]
        assert trainer.vllm_metrics_monitor.events == [("resume", "default"), ("pause", "default")]

    def test_window_closes_when_generation_fails(self, trainer):
        trainer.vllm_metrics_monitor = _RecordingMonitor()
        with pytest.raises(RuntimeError):
            trainer._generate_sequences_with_vllm_metrics("boom")
        assert trainer.vllm_metrics_monitor.events[-1] == ("pause", "default")

    def test_without_monitor(self, trainer):
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        assert RayPPOTrainer.vllm_metrics_monitor is None

        class Manager:
            def generate_sequences(self, batch):
                return f"out:{batch}"

        trainer.async_rollout_manager = Manager()
        assert trainer._generate_sequences_with_vllm_metrics("b") == "out:b"
        trainer._stop_vllm_metrics_monitor()  # no-op
