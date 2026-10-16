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
"""Chat templates from files (resolve_chat_template) and the Open-Reasoner-Zero template on Qwen2.5-7B's tokenizer.

The ORZ arm keeps Qwen2.5-7B's tokenizer and swaps only the chat template for ORZ's
(verl/utils/tokenizer/chat_templates/orz.jinja). The tokenizer tests use the local Hugging Face cache and skip when
the tokenizers are not there.
"""

import os
from pathlib import Path

import pytest

from verl.utils.tokenizer.chat_template import resolve_chat_template

REPO = Path(__file__).resolve().parents[2]
ORZ_TEMPLATE = REPO / "verl" / "utils" / "tokenizer" / "chat_templates" / "orz.jinja"
ORZ_SYSTEM = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant "
    "first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning "
    "process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
)
# The user message of orz-math-72k / aime-*-orz rows (scripts/convert_orz72k_to_verl_parquet.py).
ORZ_USER = (
    "You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final "
    "answer will be extracted automatically by the \\boxed{} tag.\nThis is the problem:\nWhat is $1+1$?"
)
MESSAGES = [{"role": "user", "content": ORZ_USER}]


# --------------------------------------------------------------------------- resolve_chat_template


def test_none_and_inline_templates_pass_through():
    assert resolve_chat_template(None) is None
    inline = "{% for m in messages %}{{ m.content }}{% endfor %}"
    assert resolve_chat_template(inline) == inline


def test_a_jinja_file_is_read(tmp_path):
    path = tmp_path / "t.jinja"
    path.write_text("{{ messages[0].content }}\n", encoding="utf-8")
    assert resolve_chat_template(str(path)) == "{{ messages[0].content }}\n"
    assert resolve_chat_template(f"  {path}  ") == "{{ messages[0].content }}\n"


def test_a_missing_jinja_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="custom_chat_template file not found"):
        resolve_chat_template(str(tmp_path / "missing.jinja"))


def test_an_inline_template_mentioning_jinja_is_not_a_path():
    inline = "{{ 'see orz.jinja' }}\n{{ messages[0].content }} x.jinja"
    assert resolve_chat_template(inline) == inline


def test_the_shipped_orz_template_resolves():
    text = resolve_chat_template(str(ORZ_TEMPLATE))
    assert ORZ_SYSTEM in text and "'User: '" in text and "Assistant: <think>" in text


# --------------------------------------------------------------------------- the ORZ template on real tokenizers


# Qwen2.5 checkpoints of every size ship the same tokenizer.json, so any cached one stands in for Qwen2.5-7B.
QWEN25_REPOS = ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-32B")
ORZ_REPOS = ("Open-Reasoner-Zero/Open-Reasoner-Zero-7B",)


def _snapshot_with_tokenizer(repo_ids):
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "hub")
    for repo_id in repo_ids:
        for snap in sorted((hub.expanduser() / f"models--{repo_id.replace('/', '--')}" / "snapshots").glob("*")):
            if (snap / "tokenizer.json").is_file() and (snap / "tokenizer_config.json").is_file():
                return snap
    return None


def _tokenizer(repo_ids):
    transformers = pytest.importorskip("transformers")
    snap = _snapshot_with_tokenizer(repo_ids)
    if snap is None:
        pytest.skip(f"none of {repo_ids} has a cached tokenizer")
    return transformers.AutoTokenizer.from_pretrained(str(snap), local_files_only=True)


@pytest.fixture(scope="module")
def qwen25():
    return _tokenizer(QWEN25_REPOS)


@pytest.fixture(scope="module")
def orz7b():
    return _tokenizer(ORZ_REPOS)


def test_the_file_is_orz7b_s_own_template(orz7b):
    assert ORZ_TEMPLATE.read_text(encoding="utf-8") == orz7b.chat_template


def test_qwen25_with_the_orz_template_renders_orz_s_prompt(qwen25):
    text = qwen25.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True, chat_template=resolve_chat_template(str(ORZ_TEMPLATE))
    )
    assert text == ORZ_SYSTEM + "User: " + ORZ_USER + "\nAssistant: <think>"
    assert "<|im_start|>" not in text and "You are a helpful assistant" not in text
    # Qwen2.5-7B's own template would have wrapped the message in ChatML
    assert "<|im_start|>" in qwen25.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)


def test_qwen25_tokenizer_plus_template_equals_the_orz_tokenizer(qwen25, orz7b):
    template = resolve_chat_template(str(ORZ_TEMPLATE))
    ids_qwen = qwen25.apply_chat_template(MESSAGES, tokenize=True, add_generation_prompt=True, chat_template=template)
    ids_orz = orz7b.apply_chat_template(MESSAGES, tokenize=True, add_generation_prompt=True)
    assert list(ids_qwen) == list(ids_orz)
    assert qwen25.eos_token == orz7b.eos_token == "<|endoftext|>"
    assert qwen25.get_vocab() == orz7b.get_vocab()


@pytest.mark.parametrize("stop", ["User:", "Human:", "Assistant:", "</answer>"])
def test_stop_strings_round_trip_through_the_tokenizer(qwen25, stop):
    # the reward decodes the response token ids: a stop string kept in the output survives decoding
    response = "<think>x</think> <answer> \\boxed{2} </answer>" + ("" if stop == "</answer>" else f"\n{stop}")
    assert qwen25.decode(qwen25.encode(response), skip_special_tokens=True) == response


def test_hf_model_config_applies_the_template_file(qwen25, monkeypatch):
    from verl.workers.config.model import HFModelConfig

    snap = _snapshot_with_tokenizer(QWEN25_REPOS)
    if not (snap / "config.json").is_file():
        pytest.skip(f"{snap} has no config.json")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = HFModelConfig(path=str(snap), custom_chat_template=str(ORZ_TEMPLATE))
    assert cfg.custom_chat_template == ORZ_TEMPLATE.read_text(encoding="utf-8")
    assert cfg.tokenizer.chat_template == cfg.custom_chat_template
    text = cfg.tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    assert text.startswith(ORZ_SYSTEM) and text.endswith("\nAssistant: <think>")


def test_hf_model_config_without_a_template_keeps_the_tokenizer_s(qwen25, monkeypatch):
    from verl.workers.config.model import HFModelConfig

    snap = _snapshot_with_tokenizer(QWEN25_REPOS)
    if not (snap / "config.json").is_file():
        pytest.skip(f"{snap} has no config.json")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = HFModelConfig(path=str(snap))
    assert cfg.custom_chat_template is None and cfg.tokenizer.chat_template == qwen25.chat_template


def test_hf_model_config_rejects_a_missing_template_file(qwen25, monkeypatch, tmp_path):
    from verl.workers.config.model import HFModelConfig

    snap = _snapshot_with_tokenizer(QWEN25_REPOS)
    if not (snap / "config.json").is_file():
        pytest.skip(f"{snap} has no config.json")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(FileNotFoundError):
        HFModelConfig(path=str(snap), custom_chat_template=str(tmp_path / "nope.jinja"))
