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

from verl.experimental.agent_loop.batch_gate import GateCounter


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
