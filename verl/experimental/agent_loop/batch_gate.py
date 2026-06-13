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
"""Shared global counter for over-sample-and-discard rollout generation.

The agent-loop workers generate ``k * train_batch_size`` prompt-groups concurrently and report each
*complete* group (all ``rollout.n`` rollouts finished) to a single shared ``BatchGate`` actor. The
gate accepts exactly ``target`` groups globally and rejects the rest, so the fastest
``train_batch_size`` groups are kept and the slow long-tail is discarded. ``wait_until_done`` lets a
worker that has no more completing groups still get cancelled promptly once the global target is hit.

See ``verl/experimental/agent_loop/agent_loop.py`` (``AgentLoopWorker.generate_sequences``) and
``verl/experimental/one_step_off_policy/ray_trainer.py``.
"""

import asyncio

import ray


class GateCounter:
    """Ray-free accept-counter logic (unit-testable without a Ray runtime)."""

    def __init__(self, target: int):
        self.target = target
        self.count = 0

    def offer(self) -> bool:
        """Accept one group if still under ``target``; otherwise reject."""
        if self.count >= self.target:
            return False
        self.count += 1
        return True

    @property
    def done(self) -> bool:
        return self.count >= self.target


@ray.remote(num_cpus=0)
class BatchGate:
    """Atomic global accept-counter for the fastest ``target`` complete prompt-groups.

    Runs as a Ray async actor, so every ``offer`` call is processed serially on the actor's event
    loop (atomic increment, no lock needed).
    """

    def __init__(self, target: int):
        self._counter = GateCounter(target)
        self._done = asyncio.Event()
        if self._counter.done:
            self._done.set()

    def offer(self) -> bool:
        """Try to accept one complete group. Returns True if accepted (still under target),
        False once ``target`` groups have already been accepted (caller discards the group)."""
        accepted = self._counter.offer()
        if self._counter.done:
            self._done.set()
        return accepted

    async def wait_until_done(self) -> None:
        """Resolve once ``target`` groups have been accepted globally."""
        await self._done.wait()

    def reset(self, target: int) -> None:
        """Reset the counter for reuse across rollout steps."""
        self._counter = GateCounter(target)
        self._done = asyncio.Event()
        if self._counter.done:
            self._done.set()
