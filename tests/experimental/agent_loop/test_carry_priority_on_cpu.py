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

from __future__ import annotations

from typing import Any, Optional

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.experimental.agent_loop.carry_utils import carry_request_priority
from verl.experimental.agent_loop.single_turn_carry_agent_loop import PartialCarryAgentLoop
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.replica import TokenOutput


class _FakeTokenizer:
    padding_side = "right"

    def apply_chat_template(self, messages, **kwargs) -> list[int]:
        del messages, kwargs
        return [101, 102]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return "<decoded>"


# ---------------------------------------------------------------------------
# carry_request_priority (pure helper)
# ---------------------------------------------------------------------------


def test_vllm_priority_carried_request_gets_minus_one():
    assert carry_request_priority(3, "vllm", "priority") == -1


def test_vllm_priority_fresh_request_gets_zero():
    assert carry_request_priority(0, "vllm", "priority") == 0


def test_vllm_fcfs_returns_none():
    assert carry_request_priority(3, "vllm", "fcfs") is None


def test_sglang_returns_none_even_with_priority_policy():
    assert carry_request_priority(3, "sglang", "priority") is None


def test_missing_backend_returns_none():
    assert carry_request_priority(3, None, "priority") is None
    assert carry_request_priority(3, "vllm", None) is None


# ---------------------------------------------------------------------------
# PartialCarryAgentLoop wiring
# ---------------------------------------------------------------------------


class _RecordingServerManager:
    """Records every generate() call's kwargs (accepts anything, like the vLLM server)."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def generate(self, request_id: str, **kwargs: Any) -> TokenOutput:
        self.calls.append(kwargs)
        return TokenOutput(token_ids=[11, 12], log_probs=[0.0, 0.0], stop_reason="stop")


class _StrictSglangLikeServerManager:
    """generate() with sglang's fixed signature: no priority param, no **kwargs.

    Passing priority= to this raises TypeError, proving the loop omits the kwarg
    when the backend/policy combination does not support it.
    """

    def __init__(self):
        self.calls = 0

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> TokenOutput:
        del request_id, prompt_ids, sampling_params, image_data, video_data, audio_data, mm_processor_kwargs
        self.calls += 1
        return TokenOutput(token_ids=[11, 12], log_probs=[0.0, 0.0], stop_reason="stop")


def _make_loop(server_manager, scheduling_policy: str = "priority") -> PartialCarryAgentLoop:
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "scheduling_policy": scheduling_policy,
                    "prompt_length": 16,
                    "response_length": 16,
                    "multi_turn": {"tool_config_path": None},
                },
                "model": {},
            },
            "data": {
                "tool_config_path": None,
                "apply_chat_template_kwargs": {},
            },
        }
    )
    return PartialCarryAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=server_manager,
        tokenizer=_FakeTokenizer(),
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
    )


_RAW_PROMPT = [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_carried_row_sends_priority_minus_one():
    manager = _RecordingServerManager()
    loop = _make_loop(manager)
    out = await loop.run(
        sampling_params={},
        raw_prompt=_RAW_PROMPT,
        prefix_response_ids=[5, 6],
        prefix_logprobs=[-0.1, -0.2],
    )
    assert len(manager.calls) == 1
    assert manager.calls[0]["priority"] == -1
    assert out.response_ids == [5, 6, 11, 12]


@pytest.mark.asyncio
async def test_fresh_row_sends_priority_zero():
    manager = _RecordingServerManager()
    loop = _make_loop(manager)
    out = await loop.run(sampling_params={}, raw_prompt=_RAW_PROMPT)
    assert len(manager.calls) == 1
    assert manager.calls[0]["priority"] == 0
    assert out.response_ids == [11, 12]


@pytest.mark.asyncio
async def test_fcfs_policy_omits_priority_kwarg():
    # Strict sglang-like signature: the call only succeeds if the loop did NOT pass priority=.
    manager = _StrictSglangLikeServerManager()
    loop = _make_loop(manager, scheduling_policy="fcfs")
    out = await loop.run(
        sampling_params={},
        raw_prompt=_RAW_PROMPT,
        prefix_response_ids=[5, 6],
        prefix_logprobs=[-0.1, -0.2],
    )
    assert manager.calls == 1
    assert out.response_ids == [5, 6, 11, 12]


@pytest.mark.asyncio
async def test_carry_done_row_never_generates():
    manager = _RecordingServerManager()
    loop = _make_loop(manager)
    out = await loop.run(
        sampling_params={},
        raw_prompt=_RAW_PROMPT,
        prefix_response_ids=[5, 6],
        prefix_logprobs=[-0.1, -0.2],
        carry_done=True,
    )
    assert manager.calls == []
    assert out.response_ids == [5, 6]
    assert out.response_logprobs == [-0.1, -0.2]
