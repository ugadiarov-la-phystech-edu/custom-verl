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
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_carry")
class PartialCarryAgentLoop(SingleTurnAgentLoop):
    """Single-turn agent loop that RESUMES a partially-generated response (partial-rollout carry-over).

    Used by the colocated sync trainer's ``VERL_PARTIAL_ROLLOUT`` mode. When a rollout does not finish
    within the step that harvests the fastest ``train_batch_size`` complete groups, the trainer stashes
    the tokens generated so far and re-submits the prompt on the next step with three extra per-sample
    fields:

    - ``prefix_response_ids``: accumulated response token ids generated in earlier steps.
    - ``prefix_logprobs``: their per-token rollout log-probs (the behavior-policy denominator that keeps
      the off-policy importance ratio per-token-correct across the policy-version boundary).
    - ``carry_done``: ``True`` if the rollout already finished (EOS / length) and only needs to pass
      through unchanged this step.

    This loop re-prefills ``prompt_ids + prefix_response_ids``, caps ``max_tokens`` so the running
    response stays within ``response_length``, and returns ``prefix + new`` as the full response /
    logprobs — mirroring ``fully_async_policy``'s client-side resume (``prompt_ids + token_ids``), but
    driven step-to-step by the sync trainer's carry buffer rather than a long-lived client loop. With no
    carry fields present it behaves exactly like :class:`SingleTurnAgentLoop`.
    """

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        prefix_ids = list(kwargs.get("prefix_response_ids") or [])
        prefix_logprobs = list(kwargs.get("prefix_logprobs") or [])
        carry_done = bool(kwargs.get("carry_done", False))
        remaining = self.response_length - len(prefix_ids)

        metrics: dict[str, Any] = {}
        new_token_ids: list[int] = []
        new_logprobs = None
        num_preempted = 0
        stop_reason = "stop"
        if not carry_done and remaining > 0:
            gen_sampling_params = dict(sampling_params)
            gen_sampling_params["max_tokens"] = remaining
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids + prefix_ids,
                    sampling_params=gen_sampling_params,
                    image_data=images,
                    video_data=videos,
                    audio_data=audios,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
            new_token_ids = output.token_ids
            new_logprobs = output.log_probs
            num_preempted = output.num_preempted
            stop_reason = getattr(output, "stop_reason", None)
        metrics.setdefault("generate_sequences", 0.0)
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = num_preempted if num_preempted is not None else -1

        response_ids = (prefix_ids + list(new_token_ids))[: self.response_length]
        have_new_lp = new_logprobs is not None
        prefix_lp_ok = len(prefix_logprobs) == len(prefix_ids)
        if not prefix_ids and have_new_lp:
            response_logprobs = list(new_logprobs)[: self.response_length]
        elif prefix_lp_ok and (carry_done or remaining <= 0):
            response_logprobs = prefix_logprobs[: self.response_length]
        elif prefix_lp_ok and have_new_lp:
            response_logprobs = (prefix_logprobs + list(new_logprobs))[: self.response_length]
        else:
            response_logprobs = None
        response_mask = [1] * len(response_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields={},
        )
        output.extra_fields.update({"turn_scores": [], "tool_rewards": [], "stop_reason": stop_reason})
        return output
