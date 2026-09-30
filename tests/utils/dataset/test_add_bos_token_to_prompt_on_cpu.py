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
"""``data.add_bos_token_to_prompt``: the BOS the openPangu tokenizer expects, and nothing else.

Pinned with stub tokenizers (no network, no checkpoint) that model the three relevant shapes:

* Pangu:   ``add_bos_token=True``, template never emits BOS  -> the flag adds exactly one BOS;
* Qwen:    no BOS token at all                                -> always a no-op;
* Llama-3: BOS rendered by the template itself                -> never doubled.

Covered: the helper's decision table, the real ``SingleTurnAgentLoop.apply_chat_template`` (the
prompt ids sent to vLLM), ``RLHFDataset``'s overlong-prompt filter, and the config default.
"""

import asyncio
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from verl.utils.dataset.prompt_utils import maybe_prepend_bos, tokenizer_wants_bos

BOS_ID = 1
EOS_ID = 2
MESSAGES = [{"role": "user", "content": "What is 1+1?"}]


class _StubTokenizer:
    """Whitespace tokenizer with a chat template and configurable BOS behaviour.

    Like HF: ``apply_chat_template(tokenize=True)`` never adds special tokens, while
    ``encode(add_special_tokens=True)`` / ``__call__`` prepend BOS when ``add_bos_token`` is set.
    """

    def __init__(self, add_bos_token=True, bos_token="<s>", bos_in_template=False):
        self.add_bos_token = add_bos_token
        self.bos_token = bos_token
        self.bos_token_id = BOS_ID if bos_token is not None else None
        self.eos_token = "[eos]"
        self.eos_token_id = EOS_ID
        self.pad_token_id = 0
        self.chat_template = "stub"
        self._bos_in_template = bos_in_template
        self._vocab = {"<s>": BOS_ID, "[eos]": EOS_ID}

    def apply_chat_template(
        self, messages, add_generation_prompt=True, tokenize=True, tools=None, return_dict=False, **kwargs
    ):
        text = "".join(f"[{m['role']}] {m['content']} [eos] " for m in messages)
        if add_generation_prompt:
            text += "[assistant]"
        if self._bos_in_template:
            text = f"{self.bos_token} {text}"
        return self.encode(text, add_special_tokens=False) if tokenize else text

    def encode(self, text, add_special_tokens=True):
        ids = []
        for tok in text.split():
            ids.append(self._vocab.setdefault(tok, 10 + len(self._vocab)))
        if add_special_tokens and self.add_bos_token and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        return ids

    def __call__(self, text, add_special_tokens=True, **kwargs):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def __len__(self):
        return 1000


def pangu():
    return _StubTokenizer(add_bos_token=True)


def qwen():
    return _StubTokenizer(add_bos_token=False, bos_token=None)


def llama3():
    return _StubTokenizer(add_bos_token=True, bos_in_template=True)


def _render(tok, messages=MESSAGES):
    raw = tok.apply_chat_template(messages, tokenize=False)
    return raw, tok.encode(raw, add_special_tokens=False)


# --------------------------------------------------------------------------- helper


class TestTokenizerWantsBos:
    @pytest.mark.parametrize(
        "tok, expected",
        [(pangu(), True), (qwen(), False), (_StubTokenizer(add_bos_token=False), False), (llama3(), True)],
    )
    def test_matches_what_tokenizer_text_would_do(self, tok, expected):
        assert tokenizer_wants_bos(tok) is expected

    def test_missing_attributes_mean_no(self):
        assert tokenizer_wants_bos(object()) is False


class TestMaybePrependBos:
    def test_pangu_gets_exactly_one_bos_matching_the_reference_path(self):
        tok = pangu()
        raw, ids = _render(tok)
        out = maybe_prepend_bos(tok, raw, ids, enabled=True)
        assert out == [BOS_ID] + ids
        # the official recipe: render with tokenize=False, then a plain tokenizer(text) call
        assert out == tok(raw)["input_ids"]

    @pytest.mark.parametrize("make_tok", [pangu, qwen, llama3])
    def test_flag_off_is_always_a_noop(self, make_tok):
        tok = make_tok()
        raw, ids = _render(tok)
        assert maybe_prepend_bos(tok, raw, ids, enabled=False) == ids

    def test_tokenizer_without_bos_is_a_noop(self):
        tok = qwen()
        raw, ids = _render(tok)
        assert maybe_prepend_bos(tok, raw, ids, enabled=True) == ids

    def test_add_bos_token_false_is_a_noop(self):
        tok = _StubTokenizer(add_bos_token=False)
        raw, ids = _render(tok)
        assert maybe_prepend_bos(tok, raw, ids, enabled=True) == ids

    def test_template_that_renders_bos_is_not_doubled(self):
        tok = llama3()
        raw, ids = _render(tok)
        assert ids[0] == BOS_ID
        out = maybe_prepend_bos(tok, raw, ids, enabled=True)
        assert out == ids and out.count(BOS_ID) == 1

    def test_ids_already_starting_with_bos_are_not_doubled(self):
        tok = pangu()
        raw, ids = _render(tok)
        assert maybe_prepend_bos(tok, raw, [BOS_ID] + ids, enabled=True) == [BOS_ID] + ids

    def test_empty_prompt_gets_bos(self):
        assert maybe_prepend_bos(pangu(), "", [], enabled=True) == [BOS_ID]

    def test_returns_a_new_list_and_accepts_any_sequence(self):
        tok = pangu()
        raw, ids = _render(tok)
        as_tuple = tuple(ids)
        out = maybe_prepend_bos(tok, raw, as_tuple, enabled=False)
        assert isinstance(out, list) and out == ids
        out.append(99)
        assert list(as_tuple) == ids


# --------------------------------------------------------------------------- agent loop


def _agent_loop_prompt_ids(tok, flag, messages=MESSAGES, remove_system_prompt=False):
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
    from verl.utils.dataset.rl_dataset import RLHFDataset

    trainer_config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"prompt_length": 512, "response_length": 64}}})
    data_config = OmegaConf.create(
        {
            "apply_chat_template_kwargs": {},
            "mm_processor_kwargs": {},
            "continuous_token": {"enable": False, "model_family": "auto"},
            "add_bos_token_to_prompt": flag,
        }
    )

    async def run():
        loop = SingleTurnAgentLoop(
            trainer_config=DictConfigWrap(trainer_config),
            server_manager=None,
            tokenizer=tok,
            processor=None,
            dataset_cls=RLHFDataset,
            data_config=DictConfigWrap(data_config),
        )
        return await loop.apply_chat_template(messages, remove_system_prompt=remove_system_prompt)

    return asyncio.run(run())


class TestAgentLoop:
    @pytest.mark.parametrize("make_tok", [pangu, qwen, llama3])
    def test_flag_off_is_token_identical_to_tokenize_true(self, make_tok):
        tok = make_tok()
        assert _agent_loop_prompt_ids(tok, flag=False) == tok.apply_chat_template(MESSAGES, tokenize=True)

    def test_pangu_prompt_starts_with_a_single_bos(self):
        tok = pangu()
        _, ids = _render(tok)
        assert _agent_loop_prompt_ids(tok, flag=True) == [BOS_ID] + ids

    @pytest.mark.parametrize("make_tok", [qwen, llama3])
    def test_flag_on_is_a_noop_where_bos_is_absent_or_already_rendered(self, make_tok):
        tok = make_tok()
        assert _agent_loop_prompt_ids(tok, flag=True) == tok.apply_chat_template(MESSAGES, tokenize=True)

    def test_continuation_turns_never_get_a_bos(self):
        # remove_system_prompt=True is how tool loops tokenize follow-up turns
        tok = pangu()
        on = _agent_loop_prompt_ids(tok, flag=True, remove_system_prompt=True)
        off = _agent_loop_prompt_ids(tok, flag=False, remove_system_prompt=True)
        assert on == off
        assert BOS_ID not in on

    def test_agent_loop_and_dataset_filter_agree_on_length(self, tmp_path):
        tok = pangu()
        ids = _agent_loop_prompt_ids(tok, flag=True)
        ds = _dataset(tmp_path, tok, flag=True, max_prompt_length=len(ids))
        assert len(ds) == 1
        ds = _dataset(tmp_path, tok, flag=True, max_prompt_length=len(ids) - 1)
        assert len(ds) == 0


# --------------------------------------------------------------------------- dataset filter


def _dataset(tmp_path: Path, tok, flag, max_prompt_length, messages=MESSAGES):
    from verl.utils.dataset.rl_dataset import RLHFDataset

    path = tmp_path / "train.parquet"
    pd.DataFrame({"prompt": [messages], "data_source": ["unit"], "reward_model": [{"ground_truth": "2"}]}).to_parquet(
        path
    )
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": max_prompt_length,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 1,
            "cache_dir": str(tmp_path / "cache"),
            "add_bos_token_to_prompt": flag,
        }
    )
    return RLHFDataset(data_files=str(path), tokenizer=tok, config=config)


class TestDatasetFilter:
    def test_flag_off_keeps_a_prompt_that_fits_without_bos(self, tmp_path):
        tok = pangu()
        n = len(_render(tok)[1])
        assert len(_dataset(tmp_path, tok, flag=False, max_prompt_length=n)) == 1

    def test_flag_on_counts_the_bos(self, tmp_path):
        tok = pangu()
        n = len(_render(tok)[1])
        # exactly at the limit without BOS: the BOS pushes it one token over
        assert len(_dataset(tmp_path, tok, flag=True, max_prompt_length=n)) == 0
        assert len(_dataset(tmp_path, tok, flag=True, max_prompt_length=n + 1)) == 1

    @pytest.mark.parametrize("make_tok", [qwen, llama3])
    def test_flag_on_does_not_change_lengths_without_a_missing_bos(self, tmp_path, make_tok):
        tok = make_tok()
        n = len(tok.apply_chat_template(MESSAGES, tokenize=True))
        assert len(_dataset(tmp_path, tok, flag=True, max_prompt_length=n)) == 1

    def test_dataset_reads_the_flag(self, tmp_path):
        assert _dataset(tmp_path, pangu(), flag=True, max_prompt_length=512).add_bos_token_to_prompt is True
        assert _dataset(tmp_path, pangu(), flag=False, max_prompt_length=512).add_bos_token_to_prompt is False


# --------------------------------------------------------------------------- config


def test_config_default_is_off():
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="ppo_trainer")
    assert cfg.data.add_bos_token_to_prompt is False
