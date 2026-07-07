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
