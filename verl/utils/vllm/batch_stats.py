# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helpers for recording vLLM per-forward-step batch-size statistics.

Used by the rollout worker extension to count, for every model forward pass,
how many sequences were in the running batch (total width) and how many of
those were emitting a decode token. The driver turns the resulting histograms
(Counters keyed by batch size) into per-GPU TensorBoard histograms to study the
decode long-tail. See ``verl/workers/rollout/vllm_rollout/utils.py``.
"""

from collections import Counter
from typing import Optional


def forward_batch_sizes(scheduler_output) -> Optional[tuple[int, int]]:
    """Extract ``(num_reqs, num_decode)`` for one vLLM forward step.

    ``num_reqs`` is every scheduled request (prefill + decode); ``num_decode`` is the
    requests emitting a single decode token (``num_scheduled_tokens == 1``), the
    long-tail signal. Computed *before* ``execute_model`` runs so a possibly consumed
    ``scheduler_output`` is never read afterward.

    Access is defensive: if the expected vLLM V1 fields are missing (e.g. after a
    version bump), returns ``None`` so the caller becomes a no-op rather than breaking
    generation.
    """
    num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not num_scheduled_tokens:
        return None
    num_reqs = len(num_scheduled_tokens)
    total_tokens = getattr(scheduler_output, "total_num_scheduled_tokens", num_reqs)
    if total_tokens == num_reqs:
        num_decode = num_reqs
    else:
        num_decode = sum(1 for v in num_scheduled_tokens.values() if v == 1)
    return num_reqs, num_decode


def record_forward(
    decode_counter: Counter,
    total_counter: Counter,
    decode_time: Counter,
    total_time: Counter,
    num_reqs: int,
    num_decode: int,
    dt: float,
) -> None:
    """Record one forward step's batch sizes and elapsed time into the running histograms.

    Updates both the call-count counters (one count per forward, keyed by batch size) and
    the time counters (the forward's wall time ``dt`` added to the same batch-size bin).
    """
    total_counter[num_reqs] += 1
    decode_counter[num_decode] += 1
    total_time[num_reqs] += dt
    decode_time[num_decode] += dt


def counter_to_histogram_raw(counter: dict) -> tuple[Optional[dict], float]:
    """Turn a histogram Counter ({batch_size: weight}) into the arguments for
    ``SummaryWriter.add_histogram_raw``, as a normalized, unbinned (one bucket per
    integer batch size) probability mass function.

    The weight per batch size is either a call count (int) or accumulated forward time
    (float); the function works for both. Returns ``(hist_kwargs, total)`` where
    ``hist_kwargs`` has the keys ``min, max, num, sum, sum_squares, bucket_limits,
    bucket_counts`` with ``bucket_counts`` summing to 1, and ``total`` is the sum of the
    counter values (number of forward calls, or total seconds). Returns ``(None, 0)`` for
    an empty counter.
    """
    total = sum(counter.values())
    if total == 0:
        return None, 0
    lo, hi = min(counter), max(counter)
    values = list(range(lo, hi + 1))
    probs = [counter.get(v, 0) / total for v in values]
    bucket_limits = [v + 0.5 for v in values]
    mean = float(sum(v * p for v, p in zip(values, probs, strict=True)))
    sum_squares = float(sum(v * v * p for v, p in zip(values, probs, strict=True)))
    hist_kwargs = {
        "min": float(lo),
        "max": float(hi),
        "num": 1.0,
        "sum": mean,
        "sum_squares": sum_squares,
        "bucket_limits": bucket_limits,
        "bucket_counts": probs,
    }
    return hist_kwargs, total
