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
"""actor_rollout_ref.rollout.stop: the stop strings vLLM samples with (Open-Reasoner-Zero's format needs them)."""

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import RolloutConfig

REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / "verl" / "trainer" / "config"
ORZ_STOP = ["User:", "Human:", "Assistant:", "</answer>"]


def _rollout(**kw):
    return RolloutConfig(name="vllm", **kw)


def test_default_is_none():
    assert _rollout().stop is None


@pytest.mark.parametrize("stop", [ORZ_STOP, ["</answer>"], ("User:", "</answer>")])
def test_lists_of_strings_are_accepted(stop):
    assert list(_rollout(stop=stop).stop) == list(stop)


@pytest.mark.parametrize(
    "stop, message",
    [
        ("</answer>", "must be a list of strings or null"),  # a bare string would be split into characters
        (5, "must be a list of strings or null"),
        (["</answer>", ""], "non-empty strings"),
        (["</answer>", 3], "non-empty strings"),
    ],
)
def test_invalid_values_are_rejected(stop, message):
    with pytest.raises(ValueError, match=message):
        _rollout(stop=stop)


def test_hydra_list_override_reaches_the_dataclass():
    # what the launcher's actor_rollout_ref.rollout.stop='["User:",...]' override produces
    cfg = OmegaConf.merge(
        OmegaConf.structured(RolloutConfig),
        OmegaConf.from_dotlist(['name=vllm', 'stop=["User:","Human:","Assistant:","</answer>"]']),
    )
    rollout = omega_conf_to_dataclass(cfg, dataclass_type=RolloutConfig)
    assert list(rollout.stop) == ORZ_STOP


def test_rollout_yaml_default_is_null():
    assert yaml.safe_load((CONFIG_DIR / "rollout" / "rollout.yaml").read_text())["stop"] is None


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("_generated_*.yaml")), ids=lambda p: p.name)
def test_generated_configs_carry_the_null_default(path):
    rollout = yaml.safe_load(path.read_text())["actor_rollout_ref"]["rollout"]
    assert "stop" in rollout and rollout["stop"] is None
