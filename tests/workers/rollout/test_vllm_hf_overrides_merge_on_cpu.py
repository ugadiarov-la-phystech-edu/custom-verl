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
"""A user's engine_kwargs.vllm.hf_overrides is merged into verl's hf_overrides instead of replacing it.

Regression: Qwen2.5-Math-7B needs hf_overrides={"max_position_embeddings": 32768}; with an FP8 rollout that value
replaced verl's FP8 block quantization_config, and the first weight sync crashed on weight_block_size[0] (None).
"""

import inspect
import json

import pytest
from omegaconf import OmegaConf

from verl.workers.rollout.vllm_rollout import vllm_async_server as vserver
from verl.workers.rollout.vllm_rollout.vllm_async_server import merge_hf_overrides

FP8 = {
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "ignored_layers": ["model.layers.0.mlp.gate"],
    }
}
MATH_POSITIONS = {"max_position_embeddings": 32768}


def test_none_keeps_verl_s_dict():
    assert merge_hf_overrides(FP8, None) == FP8


def test_user_dict_is_added_and_the_fp8_block_config_survives():
    merged = merge_hf_overrides(FP8, MATH_POSITIONS)
    assert merged["max_position_embeddings"] == 32768
    assert merged["quantization_config"] == FP8["quantization_config"]
    assert merged["quantization_config"]["weight_block_size"] == [128, 128]


def test_json_string_from_the_command_line():
    # what +actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides='{"max_position_embeddings":32768}' delivers
    merged = merge_hf_overrides(FP8, json.dumps(MATH_POSITIONS))
    assert merged == {**FP8, **MATH_POSITIONS}


def test_empty_string_is_no_override():
    assert merge_hf_overrides(FP8, "  ") == FP8


def test_omegaconf_mapping():
    user = OmegaConf.create({"max_position_embeddings": 32768, "quantization_config": {"activation_scheme": "static"}})
    merged = merge_hf_overrides(FP8, user)
    assert merged["max_position_embeddings"] == 32768
    assert merged["quantization_config"]["activation_scheme"] == "static"
    assert merged["quantization_config"]["weight_block_size"] == [128, 128]


def test_nested_merge_user_wins_only_on_the_keys_it_names():
    merged = merge_hf_overrides(FP8, {"quantization_config": {"weight_block_size": [64, 64]}})
    q = merged["quantization_config"]
    assert q["weight_block_size"] == [64, 64]
    assert (q["quant_method"], q["fmt"], q["activation_scheme"]) == ("fp8", "e4m3", "dynamic")


def test_user_scalar_replaces_a_verl_dict_explicitly():
    assert merge_hf_overrides(FP8, {"quantization_config": None}) == {"quantization_config": None}


def test_no_quantization_just_passes_the_user_dict():
    assert merge_hf_overrides({}, MATH_POSITIONS) == MATH_POSITIONS


def test_inputs_are_not_mutated():
    base = json.loads(json.dumps(FP8))
    user = {"quantization_config": {"weight_block_size": [64, 64]}, "rope_scaling": {"type": "linear"}}
    merged = merge_hf_overrides(base, user)
    merged["quantization_config"]["fmt"] = "e5m2"
    merged["rope_scaling"]["type"] = "dynamic"
    assert base == FP8 and user["rope_scaling"]["type"] == "linear"


@pytest.mark.parametrize("bad", [[1, 2], 3, "[1, 2]"])
def test_non_object_is_rejected(bad):
    with pytest.raises(TypeError, match="hf_overrides must be a dict"):
        merge_hf_overrides(FP8, bad)


def test_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        merge_hf_overrides(FP8, "{max_position_embeddings: 32768")


def test_launch_path_pops_and_merges_the_user_value_before_building_args():
    # the server must not pass engine_kwargs["hf_overrides"] through **engine_kwargs (that replaced verl's dict)
    src = inspect.getsource(vserver)
    i_apply = src.index("quantization, hf_overrides = self._apply_quantization()")
    i_pop = src.index('engine_kwargs.pop("hf_overrides", None)')
    i_merge = src.index("hf_overrides = merge_hf_overrides(hf_overrides, user_hf_overrides)")
    i_args = src.index('"hf_overrides": hf_overrides,')
    i_spread = src.index("**engine_kwargs,", i_args)
    assert i_apply < i_pop < i_merge < i_args < i_spread
