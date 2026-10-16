# Copyright 2026 custom-verl authors
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
"""verl/experimental/agent_loop/agent_loop.py:build_sampling_params - the per-request vLLM sampling params."""

import pytest

from verl.experimental.agent_loop.agent_loop import build_sampling_params
from verl.workers.config import RolloutConfig
from verl.workers.config.rollout import SamplingConfig

ORZ_STOP = ["User:", "Human:", "Assistant:", "</answer>"]


def _config(**kw):
    kw.setdefault("val_kwargs", SamplingConfig(temperature=0.6, top_p=0.95, top_k=20))
    return RolloutConfig(name="vllm", temperature=1.0, top_p=1.0, top_k=-1, calculate_log_probs=True, **kw)


def test_training_params_without_stop_are_unchanged():
    assert build_sampling_params(_config(), validate=False) == dict(
        temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0, logprobs=True
    )


def test_validation_params_without_stop_use_val_kwargs():
    assert build_sampling_params(_config(), validate=True) == dict(
        temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0, logprobs=True
    )


@pytest.mark.parametrize("validate", [False, True])
def test_stop_strings_apply_to_training_and_validation(validate):
    params = build_sampling_params(_config(stop=ORZ_STOP), validate=validate)
    assert params["stop"] == ORZ_STOP
    assert params["include_stop_str_in_output"] is True
    without = build_sampling_params(_config(), validate=validate)
    assert {k: v for k, v in params.items() if k not in ("stop", "include_stop_str_in_output")} == without


def test_stop_is_copied_not_aliased():
    config = _config(stop=ORZ_STOP)
    params = build_sampling_params(config, validate=False)
    params["stop"].append("X")
    assert list(config.stop) == ORZ_STOP


def test_dictconfig_rollout_config_works_too():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        dict(temperature=1.0, top_p=1.0, top_k=-1, calculate_log_probs=False, stop=["</answer>"],
             val_kwargs=dict(temperature=1.0, top_p=1.0, top_k=-1))
    )
    params = build_sampling_params(cfg, validate=True)
    assert params["stop"] == ["</answer>"] and params["logprobs"] is False
    cfg.stop = None
    assert "stop" not in build_sampling_params(cfg, validate=False)


def test_builds_real_vllm_sampling_params():
    vllm = pytest.importorskip("vllm")
    params = build_sampling_params(_config(stop=ORZ_STOP), validate=False)
    params["logprobs"] = 0 if params.pop("logprobs") else None  # as vllm_async_server.generate does
    sp = vllm.SamplingParams(max_tokens=16, **params)
    assert sp.stop == ORZ_STOP and sp.include_stop_str_in_output is True
