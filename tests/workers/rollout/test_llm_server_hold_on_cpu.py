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
"""Load-balancer hold gate for partial-rollout resumes.

While ``GlobalRequestLoadBalancer.set_hold(True)`` is in effect, a request aborted by
``abort_all_requests`` must not resubmit its continuation: FullyAsyncLLMServerClient keeps the partial
output and waits until the hold is cleared. This is what lets a stop-the-world pause (serialized
validation, checkpoint saves) freeze in-flight training generation while the servers stay available.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.workers.rollout import llm_server
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, GlobalRequestLoadBalancer
from verl.workers.rollout.replica import TokenOutput


class FakeLoadBalancer:
    """``is_held.remote()`` as the client awaits it; releases the hold after ``held_polls`` polls."""

    def __init__(self, held_polls: int, events: list):
        self.held_polls = held_polls
        self.polls = 0
        self.events = events
        self.is_held = SimpleNamespace(remote=self._is_held)

    async def _is_held(self):
        self.polls += 1
        held = self.polls <= self.held_polls
        self.events.append(("poll", held))
        return held


class FakeServer:
    """Aborts the first ``aborts`` attempts after 3 tokens each, then finishes with 2 more tokens."""

    def __init__(self, aborts: int, events: list):
        self.aborts = aborts
        self.calls: list[int] = []
        self.events = events

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        self.calls.append(len(prompt_ids))
        self.events.append(("generate", len(prompt_ids)))
        if len(self.calls) <= self.aborts:
            return TokenOutput(token_ids=[1, 2, 3], log_probs=[-0.1] * 3, stop_reason="aborted")
        return TokenOutput(token_ids=[4, 5], log_probs=[-0.2] * 2, stop_reason="completed")


@pytest.fixture
def events():
    return []


@pytest.fixture
def no_sleep(monkeypatch):
    real_sleep = asyncio.sleep

    async def fast(_delay, *args, **kwargs):
        return await real_sleep(0, *args, **kwargs)

    monkeypatch.setattr(llm_server.asyncio, "sleep", fast)


def _patch_server(monkeypatch, server):
    monkeypatch.setattr(llm_server.LLMServerClient, "generate", server.generate)


def _config(partial_rollout=True):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"response_length": 64, "name": "vllm"}},
            "async_training": {"partial_rollout": partial_rollout},
        }
    )


def _run(client):
    return asyncio.run(client.generate(request_id="r", prompt_ids=[0] * 10, sampling_params={}))


class TestLoadBalancerHold:
    def test_default_not_held(self):
        assert GlobalRequestLoadBalancer(servers={}).is_held() is False

    def test_set_and_clear(self):
        lb = GlobalRequestLoadBalancer(servers={})
        lb.set_hold(True)
        assert lb.is_held() is True
        lb.set_hold(False)
        assert lb.is_held() is False

    def test_hold_does_not_affect_routing(self):
        lb = GlobalRequestLoadBalancer(servers={"s0": "h0"})
        lb.set_hold(True)
        assert lb.acquire_server("req") == ("s0", "h0")


class TestClientWaitsWhileHeld:
    def test_resume_waits_until_released(self, monkeypatch, no_sleep, events):
        server = FakeServer(aborts=1, events=events)
        _patch_server(monkeypatch, server)
        lb = FakeLoadBalancer(held_polls=3, events=events)
        out = _run(FullyAsyncLLMServerClient(config=_config(), load_balancer_handle=lb))
        # first attempt, three held polls, the release poll, then the resume with the partial output
        assert events == [
            ("generate", 10),
            ("poll", True),
            ("poll", True),
            ("poll", True),
            ("poll", False),
            ("generate", 13),
        ]
        assert out.token_ids == [1, 2, 3, 4, 5]
        assert out.log_probs == [-0.1] * 3 + [-0.2] * 2

    def test_not_held_resumes_immediately(self, monkeypatch, no_sleep, events):
        server = FakeServer(aborts=2, events=events)
        _patch_server(monkeypatch, server)
        lb = FakeLoadBalancer(held_polls=0, events=events)
        out = _run(FullyAsyncLLMServerClient(config=_config(), load_balancer_handle=lb))
        assert server.calls == [10, 13, 16]
        assert lb.polls == 2  # one check per resume
        assert out.token_ids == [1, 2, 3, 1, 2, 3, 4, 5]

    def test_completed_request_never_polls(self, monkeypatch, no_sleep, events):
        server = FakeServer(aborts=0, events=events)
        _patch_server(monkeypatch, server)
        lb = FakeLoadBalancer(held_polls=100, events=events)
        _run(FullyAsyncLLMServerClient(config=_config(), load_balancer_handle=lb))
        assert lb.polls == 0

    def test_no_partial_rollout_drops_without_polling(self, monkeypatch, no_sleep, events):
        server = FakeServer(aborts=1, events=events)
        _patch_server(monkeypatch, server)
        lb = FakeLoadBalancer(held_polls=100, events=events)
        out = _run(FullyAsyncLLMServerClient(config=_config(partial_rollout=False), load_balancer_handle=lb))
        assert lb.polls == 0
        assert out.stop_reason == "aborted"

    def test_without_load_balancer(self, monkeypatch, no_sleep, events):
        server = FakeServer(aborts=1, events=events)
        _patch_server(monkeypatch, server)
        out = _run(FullyAsyncLLMServerClient(config=_config(), load_balancer_handle=None))
        assert out.token_ids == [1, 2, 3, 4, 5]
