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

from verl.experimental.agent_loop.batch_gate import GateCounter, classify_discarded_rollouts


def test_gate_counter_accepts_exactly_target():
    gate = GateCounter(target=3)
    # First `target` offers accepted, the rest rejected.
    assert [gate.offer() for _ in range(5)] == [True, True, True, False, False]
    assert gate.count == 3
    assert gate.done


def test_gate_counter_done_flips_at_target():
    gate = GateCounter(target=2)
    assert not gate.done
    assert gate.offer() is True
    assert not gate.done
    assert gate.offer() is True
    assert gate.done


def test_gate_counter_zero_target_is_done_and_rejects():
    gate = GateCounter(target=0)
    assert gate.done
    assert gate.offer() is False
    assert gate.count == 0


def test_classify_discarded_rollouts():
    # Groups of n=2 rollouts: A accepted; B aborted mid-generation (one rollout has tokens);
    # C never started (all rollouts at 0 tokens); D has an errored rollout (no result row),
    # so it is discarded but not classified as never-started.
    group_rows = {"A": [0, 1], "B": [2, 3], "C": [4, 5], "D": [6, 7]}
    token_counts = {2: 0, 3: 5, 4: 0, 5: 0, 7: 0}
    stats = classify_discarded_rollouts(group_rows, accepted_rows=[0, 1], token_counts=token_counts)
    assert stats == {
        "discarded_groups": 3,
        "never_started_groups": 1,
        "discarded_rollouts": 6,
        "never_started_rollouts": 4,
        "discarded_tokens": 5,
    }


def test_classify_discarded_rollouts_all_accepted():
    stats = classify_discarded_rollouts({"A": [0, 1]}, accepted_rows=[0, 1], token_counts={})
    assert stats["discarded_groups"] == 0
    assert stats["never_started_groups"] == 0
    assert stats["discarded_tokens"] == 0


def test_gate_counter_accumulates_discard_stats_across_workers():
    gate = GateCounter(target=2)
    gate.report_discard({"discarded_groups": 2, "never_started_groups": 1, "discarded_tokens": 10})
    gate.report_discard({"discarded_groups": 1, "never_started_groups": 0, "discarded_tokens": 7})
    assert gate.discard_stats == {"discarded_groups": 3, "never_started_groups": 1, "discarded_tokens": 17}
