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

from verl.utils.vllm.batch_stats import counter_to_histogram_raw, record_forward


def _scheduler_output(num_scheduled_tokens):
    total = sum(num_scheduled_tokens.values())
    return SimpleNamespace(num_scheduled_tokens=num_scheduled_tokens, total_num_scheduled_tokens=total)


def test_record_forward_all_decode():
    decode, total = Counter(), Counter()
    record_forward(decode, total, _scheduler_output({"a": 1, "b": 1, "c": 1}))
    # 3 requests, all decoding one token each.
    assert dict(total) == {3: 1}
    assert dict(decode) == {3: 1}


def test_record_forward_mixed_prefill_decode():
    decode, total = Counter(), Counter()
    # b is doing (chunked) prefill (5 tokens); a and c are decoding.
    record_forward(decode, total, _scheduler_output({"a": 1, "b": 5, "c": 1}))
    assert dict(total) == {3: 1}  # batch width counts all 3 requests
    assert dict(decode) == {2: 1}  # only 2 are emitting a decode token


def test_record_forward_accumulates():
    decode, total = Counter(), Counter()
    record_forward(decode, total, _scheduler_output({"a": 1, "b": 1}))
    record_forward(decode, total, _scheduler_output({"a": 1, "b": 1}))
    record_forward(decode, total, _scheduler_output({"a": 1}))
    assert dict(total) == {2: 2, 1: 1}
    assert dict(decode) == {2: 2, 1: 1}


def test_record_forward_missing_fields_is_noop():
    decode, total = Counter(), Counter()
    record_forward(decode, total, SimpleNamespace())  # no num_scheduled_tokens
    record_forward(decode, total, _scheduler_output({}))  # empty batch
    assert dict(total) == {}
    assert dict(decode) == {}


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
    hist, n_calls = counter_to_histogram_raw({})
    assert hist is None
    assert n_calls == 0
