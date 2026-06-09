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
"""Helpers for recording vLLM per-forward-step batch-size statistics.

Used by the rollout worker extension to count, for every model forward pass,
how many sequences were in the running batch (total width) and how many of
those were emitting a decode token. The driver turns the resulting histograms
(Counters keyed by batch size) into per-GPU TensorBoard histograms to study the
decode long-tail. See ``verl/workers/rollout/vllm_rollout/utils.py``.
"""

from collections import Counter

import numpy as np


def record_forward(decode_counter: Counter, total_counter: Counter, scheduler_output) -> None:
    """Record one vLLM forward step into the running histograms.

    Bins by the number of requests in the forward pass. ``total_counter`` counts
    every scheduled request (prefill + decode), while ``decode_counter`` counts
    only requests emitting a single decode token (``num_scheduled_tokens == 1``),
    which is the long-tail signal.

    Access to ``scheduler_output`` is defensive: if the expected vLLM V1 fields
    are missing (e.g. after a version bump), this becomes a no-op rather than
    breaking generation.
    """
    num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not num_scheduled_tokens:
        return
    num_reqs = len(num_scheduled_tokens)
    total_tokens = getattr(scheduler_output, "total_num_scheduled_tokens", num_reqs)
    # Fast path: in steady-state decode every request schedules exactly one token,
    # so total == num_reqs means all requests are decoding.
    if total_tokens == num_reqs:
        num_decode = num_reqs
    else:
        num_decode = sum(1 for v in num_scheduled_tokens.values() if v == 1)
    total_counter[num_reqs] += 1
    decode_counter[num_decode] += 1


def counter_to_samples(counter: dict) -> np.ndarray:
    """Expand a histogram Counter ({batch_size: num_forwards}) into a flat sample
    array suitable for ``SummaryWriter.add_histogram``. Returns an empty array if
    the counter is empty."""
    if not counter:
        return np.empty(0, dtype=np.int64)
    keys = np.fromiter(counter.keys(), dtype=np.int64, count=len(counter))
    counts = np.fromiter(counter.values(), dtype=np.int64, count=len(counter))
    return np.repeat(keys, counts)
