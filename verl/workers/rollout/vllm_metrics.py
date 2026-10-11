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
"""Online vLLM engine metrics for the training logger (``vllm/*`` keys).

With ``rollout.disable_log_stats=False`` every vLLM server fills its Prometheus registry and serves it at
``http://<server address>/metrics``. :class:`VllmMetricsMonitor` polls those endpoints from a daemon thread and turns
the samples into per-window scalars that the trainers add to the step's metrics:

* gauges (running / waiting requests per server, KV-cache usage) as mean and max over the *active* samples of the
  window, i.e. those with at least one running or waiting request, so idle and sleeping time does not dilute them;
* counters (preemptions, prompt / generation tokens, finished requests, prefix-cache hits) as deltas over the window,
  summed over servers;
* histograms (TTFT, inter-token latency, queue time, end-to-end latency, tokens per engine step) as window means,
  delta(sum) / delta(count).

A thread rather than a coroutine, because FullyAsyncRollouter._validate blocks the rollouter's event loop. Kept free
of vLLM imports so it can be tested on CPU-only machines.
"""

import logging
import os
import threading
import time
import urllib.request
from typing import Callable, Optional

from prometheus_client.parser import text_string_to_metric_families

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

INTERVAL_ENV = "VERL_VLLM_METRICS_INTERVAL_S"
DEFAULT_INTERVAL_S = 5.0
SCRAPE_TIMEOUT_S = 1.0

# Prometheus family names, as defined in vllm/v1/metrics/loggers.py (vLLM 0.24).
GAUGES = {
    "vllm:num_requests_running": "num_requests_running",
    "vllm:num_requests_waiting": "num_requests_waiting",
    "vllm:kv_cache_usage_perc": "kv_cache_usage",
}
COUNTERS = {
    "vllm:num_preemptions": "num_preemptions",
    "vllm:prompt_tokens": "prompt_tokens",
    "vllm:generation_tokens": "generation_tokens",
    "vllm:request_success": "requests_finished",
    "vllm:prefix_cache_queries": None,
    "vllm:prefix_cache_hits": None,
}
HISTOGRAMS = {
    "vllm:time_to_first_token_seconds": "ttft_s",
    "vllm:inter_token_latency_seconds": "itl_s",
    "vllm:request_queue_time_seconds": "queue_time_s",
    "vllm:e2e_request_latency_seconds": "e2e_latency_s",
    "vllm:iteration_tokens_total": "iteration_tokens",
}
METRIC_FAMILIES = (*GAUGES, *COUNTERS, *HISTOGRAMS)

# Exposed sample names: counters get "_total", histograms "_sum" / "_count" (buckets and "_created" are dropped).
_SAMPLES = {
    **{name: name for name in GAUGES},
    **{f"{name}_total": name for name in COUNTERS},
    **{f"{name}_sum": f"{name}_sum" for name in HISTOGRAMS},
    **{f"{name}_count": f"{name}_count" for name in HISTOGRAMS},
}
# Per-engine fractions are averaged over label sets (data-parallel engines), everything else is summed.
_AVERAGED = {"vllm:kv_cache_usage_perc"}
_CUMULATIVE = [name for name in _SAMPLES.values() if name not in GAUGES]


def parse_vllm_metrics(text: str) -> dict[str, float]:
    """Parse a vLLM ``/metrics`` payload into {family or family_sum / family_count: value}.

    Values are summed over label sets (``model_name``, ``engine``), except KV-cache usage, which is averaged.
    Families we do not log are filtered out before parsing, which keeps parsing cheap.
    """
    kept = []
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        end = len(line)
        for sep in ("{", " "):
            pos = line.find(sep)
            if pos != -1:
                end = min(end, pos)
        if line[:end] in _SAMPLES:
            kept.append(line)
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for family in text_string_to_metric_families("\n".join(kept) + "\n"):
        for sample in family.samples:
            key = _SAMPLES.get(sample.name)
            if key is None:
                continue
            sums[key] = sums.get(key, 0.0) + float(sample.value)
            counts[key] = counts.get(key, 0) + 1
    return {key: value / counts[key] if key in _AVERAGED else value for key, value in sums.items()}


def fetch_vllm_metrics(address: str, timeout: float = SCRAPE_TIMEOUT_S) -> dict[str, float]:
    """GET and parse ``http://{address}/metrics``; ``address`` is a replica's ``host:port`` (``[v6]:port``)."""
    with urllib.request.urlopen(f"http://{address}/metrics", timeout=timeout) as response:
        return parse_vllm_metrics(response.read().decode("utf-8", errors="replace"))


class VllmMetricsWindow:
    """Accumulates scrape rounds between two summarize() calls; the per-server counter baselines persist."""

    def __init__(self):
        self.baselines: dict[str, dict[str, float]] = {}
        self.reset()

    def reset(self):
        self.rounds = 0
        self.scrape_errors = 0
        self.scrape_time = 0.0
        self.active_samples = 0
        self.active_seconds = 0.0
        self.gauge_sum = {key: 0.0 for key in GAUGES}
        self.gauge_max = {key: 0.0 for key in GAUGES}
        self.deltas = {key: 0.0 for key in _CUMULATIVE}

    def rebase(self, per_server: dict[str, dict[str, float]]):
        """Take these values as the counter baselines without counting anything (window start)."""
        for address, values in per_server.items():
            self.baselines[address] = {key: values[key] for key in _CUMULATIVE if key in values}

    def add_round(self, per_server: dict[str, dict[str, float]], errors: int, seconds: float, scrape_time: float):
        """Add one scrape round: per-server parsed payloads (failed servers omitted), the number of failed servers,
        the time since the previous round and the time the round took."""
        self.rounds += 1
        self.scrape_errors += errors
        self.scrape_time += scrape_time
        generated = 0.0
        for address, values in per_server.items():
            baseline = self.baselines.setdefault(address, {})
            for key in _CUMULATIVE:
                if key not in values:
                    continue
                current = values[key]
                previous = baseline.get(key)
                if previous is not None:
                    # A counter that went down means the server restarted: count from zero.
                    delta = current - previous if current >= previous else current
                    self.deltas[key] += delta
                    if key == "vllm:generation_tokens":
                        generated += delta
                baseline[key] = current
        if not per_server:
            return
        running = sum(values.get("vllm:num_requests_running", 0.0) for values in per_server.values())
        waiting = sum(values.get("vllm:num_requests_waiting", 0.0) for values in per_server.values())
        if running + waiting > 0 or generated > 0:
            self.active_seconds += seconds
        if running + waiting <= 0:
            return
        self.active_samples += 1
        for key in GAUGES:
            per_server_values = [values.get(key, 0.0) for values in per_server.values()]
            self.gauge_sum[key] += sum(per_server_values) / len(per_server_values)
            self.gauge_max[key] = max(self.gauge_max[key], max(per_server_values))

    def summarize(self) -> dict[str, float]:
        out = {
            "vllm/samples": float(self.active_samples),
            "vllm/scrape_errors": float(self.scrape_errors),
        }
        if self.rounds == 0:
            return out
        out["vllm/scrape_time_s/mean"] = self.scrape_time / self.rounds
        if self.active_samples > 0:
            for name, key in GAUGES.items():
                out[f"vllm/{key}/mean"] = self.gauge_sum[name] / self.active_samples
                out[f"vllm/{key}/max"] = self.gauge_max[name]
        for name, key in COUNTERS.items():
            if key is not None:
                out[f"vllm/{key}"] = self.deltas[name]
        queries = self.deltas["vllm:prefix_cache_queries"]
        if queries > 0:
            out["vllm/prefix_cache_hit_rate"] = self.deltas["vllm:prefix_cache_hits"] / queries
        if self.active_seconds > 0:
            out["vllm/generation_throughput"] = self.deltas["vllm:generation_tokens"] / self.active_seconds
        for name, key in HISTOGRAMS.items():
            count = self.deltas[f"{name}_count"]
            if count > 0:
                out[f"vllm/{key}/mean"] = self.deltas[f"{name}_sum"] / count
        return out


class VllmMetricsMonitor:
    """Polls the vLLM servers' ``/metrics`` every ``interval_s`` seconds while active.

    ``get_addresses`` is re-read on every round, so replicas added or removed at run time are followed.
    ``pause(reason)`` / ``resume(reason)`` nest by reason: the monitor polls only while no reason holds it. Leaving
    the last hold scrapes once to take counter baselines, so work done while paused (validation, a save freeze, the
    non-rollout phases of a sync step) is not counted; entering the first hold scrapes once more, so counters are
    exact at the pause. ``collect()`` returns the scalars accumulated since the previous call and starts a new window;
    counter baselines carry over, so a continuously running monitor loses no counts between windows.
    """

    def __init__(
        self,
        get_addresses: Callable[[], list[str]],
        interval_s: float = DEFAULT_INTERVAL_S,
        active: bool = True,
        fetch: Callable[[str], dict[str, float]] = fetch_vllm_metrics,
    ):
        self._get_addresses = get_addresses
        self._interval_s = interval_s
        self._fetch = fetch
        self._lock = threading.Lock()
        self._window = VllmMetricsWindow()
        self._holds = {"default"}
        self._last_round = time.monotonic()
        self._stop = threading.Event()
        if active:
            self.resume()
        self._thread = threading.Thread(target=self._run, name="vllm-metrics-monitor", daemon=True)
        self._thread.start()

    def _scrape(self) -> tuple[dict[str, dict[str, float]], int, float]:
        start = time.monotonic()
        per_server, errors = {}, 0
        for address in list(self._get_addresses() or []):
            try:
                per_server[address] = self._fetch(address)
            except Exception as e:  # a slow or restarting server must not break training
                errors += 1
                logger.debug("vllm metrics scrape of %s failed: %s", address, e)
        return per_server, errors, time.monotonic() - start

    def _add_round_locked(self):
        per_server, errors, scrape_time = self._scrape()
        now = time.monotonic()
        self._window.add_round(per_server, errors, now - self._last_round, scrape_time)
        self._last_round = now

    def _run(self):
        while not self._stop.wait(self._interval_s):
            with self._lock:
                if not self._holds:
                    self._add_round_locked()

    @property
    def active(self) -> bool:
        return not self._holds

    def resume(self, reason: str = "default"):
        with self._lock:
            if reason not in self._holds:
                return
            self._holds.discard(reason)
            if self._holds:
                return
            per_server, _, _ = self._scrape()
            self._window.rebase(per_server)
            self._last_round = time.monotonic()

    def pause(self, reason: str = "default"):
        with self._lock:
            if reason in self._holds:
                return
            if not self._holds:
                self._add_round_locked()
            self._holds.add(reason)

    def collect(self) -> dict[str, float]:
        with self._lock:
            out = self._window.summarize()
            self._window.reset()
            return out

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self._interval_s + 2 * SCRAPE_TIMEOUT_S)


def get_interval_s() -> float:
    """Poll interval from VERL_VLLM_METRICS_INTERVAL_S (default 5 s; 0 disables the monitor)."""
    raw = os.getenv(INTERVAL_ENV, "")
    if raw.strip() == "":
        return DEFAULT_INTERVAL_S
    interval = float(raw)
    if interval < 0:
        raise ValueError(f"{INTERVAL_ENV} must be >= 0, got {raw!r}")
    return interval


def maybe_create_vllm_metrics_monitor(
    rollout_config, get_addresses: Callable[[], list[str]], active: bool = True
) -> Optional[VllmMetricsMonitor]:
    """A monitor iff the rollout is vLLM with stats enabled (rollout.disable_log_stats=False) and the poll interval
    is positive; None otherwise."""

    def get(key, default):  # DictConfig or a RolloutConfig dataclass
        if hasattr(rollout_config, "get"):
            return rollout_config.get(key, default)
        return getattr(rollout_config, key, default)

    if get("name", None) != "vllm" or get("disable_log_stats", True):
        return None
    interval_s = get_interval_s()
    if interval_s == 0:
        return None
    logger.warning("vLLM metrics monitor: polling /metrics every %.1f s into vllm/* metrics", interval_s)
    return VllmMetricsMonitor(get_addresses, interval_s=interval_s, active=active)
