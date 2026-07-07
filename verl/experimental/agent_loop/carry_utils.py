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
"""Pure helpers for partial-rollout carry-over (dependency-free, unit-testable on CPU).

See :class:`verl.experimental.agent_loop.single_turn_carry_agent_loop.PartialCarryAgentLoop`
for the agent loop that consumes them.
"""

from typing import Optional


def stitch_carry_response(
    prefix_ids: list[int],
    prefix_logprobs: list[float],
    new_token_ids: list[int],
    new_logprobs: Optional[list[float]],
    response_length: int,
) -> tuple[list[int], Optional[list[float]]]:
    """Stitch a carried prefix and newly generated tokens into the full running response.

    Returns ``(response_ids, response_logprobs)`` where ``response_ids = prefix + new`` capped to
    ``response_length``. ``response_logprobs`` stays aligned with ``response_ids`` whenever the
    available log-probs cover it, and is ``None`` otherwise (a ``None`` disables Layer-1
    rollout-log-prob correctness for this row, so we only give up when the data is truly missing):

    - no new tokens (pass-through, ``remaining <= 0``, or a request aborted before its first
      token): the prefix log-probs alone cover the response -- keep them;
    - new tokens with log-probs (``new_logprobs`` is not ``None``): concatenate, provided the
      prefix log-probs are aligned with the prefix (or the prefix is empty);
    - new tokens without log-probs, or a misaligned prefix: ``None``.
    """
    response_ids = (list(prefix_ids) + list(new_token_ids))[:response_length]
    prefix_lp_ok = len(prefix_logprobs) == len(prefix_ids)
    if not new_token_ids:
        response_logprobs = list(prefix_logprobs)[:response_length] if prefix_lp_ok else None
    elif prefix_lp_ok and new_logprobs is not None:
        response_logprobs = (list(prefix_logprobs) + list(new_logprobs))[:response_length]
    else:
        response_logprobs = None
    return response_ids, response_logprobs


def carry_request_priority(prefix_len: int, backend: Optional[str], scheduling_policy: Optional[str]) -> Optional[int]:
    """Per-request scheduler priority for a carry-mode generation request.

    Returns ``None`` unless the rollout backend is vLLM with ``scheduling_policy="priority"`` --
    the caller must then omit the ``priority`` kwarg entirely (other backends, e.g. sglang, do not
    accept it, and under fcfs vLLM ignores it anyway). Otherwise returns ``-1`` for a carried
    request (non-empty prefix) and ``0`` (vLLM's default) for a fresh one.

    vLLM v1 priority semantics: a LOWER value is scheduled earlier (ties broken by arrival time),
    and under KV-cache pressure the numerically highest-priority request is preempted first. So
    carried partials start decoding before any fresh request and never lose their slot to one --
    they retire in one extra step instead of being aborted repeatedly (each extra carry hop adds
    another policy version to the response's token mix and re-pays prefill on a longer prefix).
    A flat -1 suffices: carried survivors are re-submitted in stable oldest-first pool order, so
    the arrival-time tiebreak already orders them; switch to ``-prefix_len`` if carried requests
    ever starve each other.
    """
    if backend != "vllm" or scheduling_policy != "priority":
        return None
    return -1 if prefix_len > 0 else 0
