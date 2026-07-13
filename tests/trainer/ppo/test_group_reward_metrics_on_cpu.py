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
"""
Tests for compute_group_reward_metrics (groups/all_correct*, groups/all_wrong* metrics).

Samples sharing non_tensor_batch["uid"] form a GRPO group; the metrics count groups whose
samples are uniformly correct or uniformly wrong (zero within-group advantage signal).
"""

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_group_reward_metrics


def _batch(uids, acc=None, seq_scores=None):
    non_tensors = {"uid": np.array(uids, dtype=object)}
    if acc is not None:
        non_tensors["acc"] = np.array(acc, dtype=object)
    tensors = {}
    if seq_scores is not None:
        # one response token per sample; the metric sums token_level_scores over tokens
        tensors["token_level_scores"] = torch.tensor(seq_scores, dtype=torch.float32).unsqueeze(-1)
    if not tensors:
        # DataProto.from_dict requires at least one tensor to size the batch
        tensors["dummy"] = torch.zeros(len(uids))
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def test_counts_from_acc():
    # group a: all correct, group b: all wrong, group c: mixed
    batch = _batch(
        uids=["a", "a", "b", "b", "c", "c"],
        acc=[1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    )
    m = compute_group_reward_metrics(batch)
    assert m["groups/count"] == 3
    assert m["groups/all_correct"] == 1
    assert m["groups/all_wrong"] == 1
    assert m["groups/all_correct_ratio"] == 1 / 3
    assert m["groups/all_wrong_ratio"] == 1 / 3


def test_all_groups_uniform():
    batch = _batch(uids=["a", "a", "b", "b"], acc=[1, 1, 0, 0])
    m = compute_group_reward_metrics(batch)
    assert m["groups/all_correct"] == 1 and m["groups/all_correct_ratio"] == 0.5
    assert m["groups/all_wrong"] == 1 and m["groups/all_wrong_ratio"] == 0.5


def test_fallback_to_sequence_scores():
    # no acc info: positive sequence score counts as correct (math_dapo-style 1/-1 rewards)
    batch = _batch(
        uids=["a", "a", "b", "b", "c", "c"],
        seq_scores=[1.0, 1.0, -1.0, -1.0, 1.0, -1.0],
    )
    m = compute_group_reward_metrics(batch)
    assert m["groups/count"] == 3
    assert m["groups/all_correct"] == 1
    assert m["groups/all_wrong"] == 1


def test_non_numeric_acc_falls_back_to_scores():
    batch = _batch(
        uids=["a", "a", "b", "b"],
        acc=["yes", None, "no", None],
        seq_scores=[1.0, 1.0, -1.0, -1.0],
    )
    m = compute_group_reward_metrics(batch)
    assert m["groups/all_correct"] == 1
    assert m["groups/all_wrong"] == 1


def test_bool_acc():
    batch = _batch(uids=["a", "a"], acc=[True, True])
    m = compute_group_reward_metrics(batch)
    assert m["groups/all_correct"] == 1 and m["groups/all_wrong"] == 0


def test_no_uid_returns_empty():
    batch = DataProto.from_dict(tensors={"dummy": torch.zeros(2)}, non_tensors={})
    assert compute_group_reward_metrics(batch) == {}


def test_no_signal_returns_empty():
    # uid present but neither acc nor token_level_scores
    batch = _batch(uids=["a", "a"])
    assert compute_group_reward_metrics(batch) == {}
