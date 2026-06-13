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

from collections import Counter
from types import SimpleNamespace

from verl.utils.vllm.batch_stats import (
    build_cdf_figure,
    counter_to_cdf,
    counter_to_histogram_raw,
    forward_batch_sizes,
    record_forward,
)


def _scheduler_output(num_scheduled_tokens):
    total = sum(num_scheduled_tokens.values())
    return SimpleNamespace(num_scheduled_tokens=num_scheduled_tokens, total_num_scheduled_tokens=total)


def test_forward_batch_sizes_all_decode():
    # 3 requests, all decoding one token each: num_reqs=3, num_decode=3.
    assert forward_batch_sizes(_scheduler_output({"a": 1, "b": 1, "c": 1})) == (3, 3)


def test_forward_batch_sizes_mixed_prefill_decode():
    # b is doing (chunked) prefill (5 tokens); a and c are decoding.
    assert forward_batch_sizes(_scheduler_output({"a": 1, "b": 5, "c": 1})) == (3, 2)


def test_forward_batch_sizes_missing_fields_is_none():
    assert forward_batch_sizes(SimpleNamespace()) is None  # no num_scheduled_tokens
    assert forward_batch_sizes(_scheduler_output({})) is None  # empty batch


def test_record_forward_counts_and_times_accumulate():
    decode, total, decode_time, total_time = Counter(), Counter(), Counter(), Counter()
    # Two forwards at width 2 (dt 0.1, 0.2), one at width 1 (dt 0.5).
    record_forward(decode, total, decode_time, total_time, 2, 2, 0.1)
    record_forward(decode, total, decode_time, total_time, 2, 2, 0.2)
    record_forward(decode, total, decode_time, total_time, 1, 1, 0.5)
    assert dict(total) == {2: 2, 1: 1}
    assert dict(decode) == {2: 2, 1: 1}
    assert total_time[2] == 0.1 + 0.2 and total_time[1] == 0.5
    assert decode_time[2] == 0.1 + 0.2 and decode_time[1] == 0.5


def test_counter_to_histogram_raw_normalizes_to_one():
    # 3 forwards at width 2, 1 forward at width 4.
    hist, n_calls = counter_to_histogram_raw({2: 3, 4: 1})
    assert n_calls == 4
    # Contiguous per-integer buckets over [2, 4]: values 2, 3, 4.
    assert hist["min"] == 2.0 and hist["max"] == 4.0
    assert hist["bucket_limits"] == [2.5, 3.5, 4.5]
    assert hist["bucket_counts"] == [0.75, 0.0, 0.25]  # 3/4, gap, 1/4
    assert abs(sum(hist["bucket_counts"]) - 1.0) < 1e-9
    # Probability-mass convention: num=1, sum=mean.
    assert hist["num"] == 1.0
    assert abs(hist["sum"] - (2 * 0.75 + 4 * 0.25)) < 1e-9  # mean = 2.5
    assert abs(hist["sum_squares"] - (4 * 0.75 + 16 * 0.25)) < 1e-9
    assert len(hist["bucket_limits"]) == len(hist["bucket_counts"])


def test_counter_to_histogram_raw_single_value():
    hist, n_calls = counter_to_histogram_raw({190: 7})
    assert n_calls == 7
    assert hist["bucket_counts"] == [1.0]
    assert hist["bucket_limits"] == [190.5]


def test_counter_to_histogram_raw_empty():
    hist, total = counter_to_histogram_raw({})
    assert hist is None
    assert total == 0


def test_counter_to_histogram_raw_float_weights():
    # Time-weighted counter ({batch_size: seconds}); normalizes the same way.
    hist, total = counter_to_histogram_raw({2: 0.6, 4: 0.2})
    assert abs(total - 0.8) < 1e-9
    assert hist["bucket_limits"] == [2.5, 3.5, 4.5]
    expected = [0.75, 0.0, 0.25]  # 0.6/0.8, gap, 0.2/0.8
    assert all(abs(a - b) < 1e-9 for a, b in zip(hist["bucket_counts"], expected, strict=True))
    assert abs(sum(hist["bucket_counts"]) - 1.0) < 1e-9


def test_counter_to_cdf_normalizes_to_one():
    # 3 forwards at width 2, 1 forward at width 4: CDF over distinct observed sizes.
    xs, ys = counter_to_cdf({2: 3, 4: 1})
    assert xs == [2.0, 4.0]  # exact observed batch sizes, no zero-fill
    assert all(abs(a - b) < 1e-9 for a, b in zip(ys, [0.75, 1.0], strict=True))
    assert ys[-1] == 1.0


def test_counter_to_cdf_single_value():
    xs, ys = counter_to_cdf({190: 7})
    assert xs == [190.0]
    assert ys == [1.0]


def test_counter_to_cdf_empty():
    assert counter_to_cdf({}) == ([], [])


def test_counter_to_cdf_float_weights():
    # Time-weighted counter ({batch_size: seconds}); normalizes the same way.
    xs, ys = counter_to_cdf({2: 0.6, 4: 0.2})
    assert xs == [2.0, 4.0]
    assert all(abs(a - b) < 1e-9 for a, b in zip(ys, [0.75, 1.0], strict=True))
    assert ys[-1] == 1.0


def test_build_cdf_figure_all_empty_is_none():
    # No data on any curve -> None (also None if matplotlib is unavailable).
    assert build_cdf_figure("t", "x", [("a", {}), ("b", {})]) is None
