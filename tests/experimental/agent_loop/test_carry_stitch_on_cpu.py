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

from verl.experimental.agent_loop.carry_utils import stitch_carry_response

RL = 8  # response_length used throughout


def test_fresh_rollout_with_logprobs():
    ids, lps = stitch_carry_response([], [], [1, 2, 3], [-0.1, -0.2, -0.3], RL)
    assert ids == [1, 2, 3]
    assert lps == [-0.1, -0.2, -0.3]


def test_fresh_rollout_without_logprobs():
    ids, lps = stitch_carry_response([], [], [1, 2, 3], None, RL)
    assert ids == [1, 2, 3]
    assert lps is None


def test_resume_concatenates_prefix_and_new():
    ids, lps = stitch_carry_response([1, 2], [-0.1, -0.2], [3, 4], [-0.3, -0.4], RL)
    assert ids == [1, 2, 3, 4]
    assert lps == [-0.1, -0.2, -0.3, -0.4]


def test_pass_through_keeps_prefix_logprobs():
    # carry_done / remaining <= 0: no generation attempted, new parts empty.
    ids, lps = stitch_carry_response([1, 2, 3], [-0.1, -0.2, -0.3], [], None, RL)
    assert ids == [1, 2, 3]
    assert lps == [-0.1, -0.2, -0.3]


def test_abort_before_first_token_keeps_prefix_logprobs():
    # A request aborted before emitting its first token returns token_ids=[] (log_probs=[] when
    # logprobs were requested). The prefix log-probs alone cover the response -- keep them.
    ids, lps = stitch_carry_response([1, 2, 3], [-0.1, -0.2, -0.3], [], [], RL)
    assert ids == [1, 2, 3]
    assert lps == [-0.1, -0.2, -0.3]


def test_fresh_abort_before_first_token_is_empty_but_aligned():
    ids, lps = stitch_carry_response([], [], [], [], RL)
    assert ids == []
    assert lps == []


def test_new_tokens_without_logprobs_drops_logprobs():
    ids, lps = stitch_carry_response([1, 2], [-0.1, -0.2], [3], None, RL)
    assert ids == [1, 2, 3]
    assert lps is None


def test_misaligned_prefix_drops_logprobs():
    ids, lps = stitch_carry_response([1, 2, 3], [-0.1], [4], [-0.4], RL)
    assert ids == [1, 2, 3, 4]
    assert lps is None


def test_caps_to_response_length():
    ids, lps = stitch_carry_response(
        list(range(6)), [-0.1] * 6, [10, 11, 12, 13], [-0.5] * 4, RL
    )
    assert ids == [0, 1, 2, 3, 4, 5, 10, 11]
    assert lps == [-0.1] * 6 + [-0.5] * 2
    assert len(ids) == RL == len(lps)
