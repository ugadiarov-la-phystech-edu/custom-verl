# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Checkpoint ``save_contents`` / ``load_contents`` semantics, and unlimited retention.

Motivation: the ``run/baselines`` scripts save export-only checkpoints with
``save_contents=['model','hf_model']`` and ``max_actor_ckpt_to_keep=null``. Both choices are
load-bearing and easy to regress silently -- dropping ``'model'`` from the list yields *empty*
checkpoints on the mbridge megatron path rather than an error (see
``test_hf_model_alone_does_not_request_a_model_save``).
"""

import os
import shutil
import tempfile

import pytest


def _manager(monkeypatch, save_contents=None, load_contents=None):
    """A minimal BaseCheckpointManager carrying the given contents lists."""
    import torch.distributed

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)

    from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager

    checkpoint_config = None
    if save_contents is not None or load_contents is not None:
        checkpoint_config = {}
        if save_contents is not None:
            checkpoint_config["save_contents"] = save_contents
        if load_contents is not None:
            checkpoint_config["load_contents"] = load_contents

    class _Mock:
        pass

    return BaseCheckpointManager(
        model=_Mock(),
        optimizer=_Mock(),
        lr_scheduler=None,
        processing_class=None,
        checkpoint_config=checkpoint_config,
    )


# ------------------------------------------------------------------ save_contents


def test_baseline_contents_save_model_and_hf_model_only(monkeypatch):
    """['model','hf_model'] -> weights (HF format via the bridge), no optimizer, no extra."""
    m = _manager(monkeypatch, save_contents=["model", "hf_model"])
    assert m.should_save_model is True
    assert m.should_save_hf_model is True
    assert m.should_save_optimizer is False
    assert m.should_save_extra is False


def test_hf_model_alone_does_not_request_a_model_save(monkeypatch):
    """Regression guard for the run/baselines choice.

    On the mbridge megatron path ``use_hf_checkpoint`` is True, so the HF export runs inside
    ``if self.should_save_model:`` (megatron_checkpoint_manager.py) and the
    ``should_save_hf_model and not use_hf_checkpoint`` branch is unreachable. With ``['hf_model']``
    alone every save predicate below is False, i.e. the checkpoint would contain no weights at all.
    That is why the scripts pass ``['model','hf_model']``.
    """
    m = _manager(monkeypatch, save_contents=["hf_model"])
    assert m.should_save_model is False, "'hf_model' alone must not imply a model save"
    assert m.should_save_optimizer is False
    assert m.should_save_extra is False
    assert m.should_save_hf_model is True


def test_default_contents_are_model_optimizer_extra(monkeypatch):
    m = _manager(monkeypatch)
    assert m.should_save_model is True
    assert m.should_save_optimizer is True
    assert m.should_save_extra is True
    assert m.should_save_hf_model is False


def test_unknown_content_tokens_are_ignored_silently(monkeypatch):
    """There is no validation of the list; a typo degrades to 'save nothing'."""
    m = _manager(monkeypatch, save_contents=["hf-model", "moldel"])
    assert m.should_save_model is False
    assert m.should_save_hf_model is False
    assert m.should_save_optimizer is False
    assert m.should_save_extra is False


# ------------------------------------------------------------------ load_contents


def test_load_contents_defaults_independently_of_save_contents(monkeypatch):
    """At the manager level the two lists are independent; the *interpolation* that makes
    load follow save lives in actor.yaml (``load_contents: ${.save_contents}``), which is
    covered by the script-config test."""
    m = _manager(monkeypatch, save_contents=["model", "hf_model"])
    assert m.should_load_model is True
    assert m.should_load_optimizer is True
    assert m.should_load_extra is True


def test_baseline_load_contents_carry_no_optimizer(monkeypatch):
    """What the composed scripts actually produce: load mirrors save, so a resume would restore
    weights but neither optimizer nor RNG -- the reason the scripts also set resume_mode=disable."""
    m = _manager(monkeypatch, save_contents=["model", "hf_model"], load_contents=["model", "hf_model"])
    assert m.should_load_model is True
    assert m.should_load_optimizer is False
    assert m.should_load_extra is False
    assert not hasattr(m, "should_load_hf_model"), "there is no load-side counterpart for hf_model"


# ------------------------------------------------------------------ retention


class TestUnlimitedRetention:
    """max_actor_ckpt_to_keep=null must never delete a checkpoint."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.test_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _mk(self, step: int) -> str:
        path = os.path.join(self.test_dir, f"global_step_{step}")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "checkpoint.txt"), "w") as f:
            f.write(f"step={step}")
        return path

    def test_none_keeps_every_checkpoint(self, monkeypatch):
        m = _manager(monkeypatch)
        paths = []
        for step in range(1, 6):
            p = self._mk(step)
            m.ensure_checkpoint_capacity(None)
            m.register_checkpoint(p, None)
            paths.append(p)
        assert all(os.path.exists(p) for p in paths), "None must disable retention entirely"
        assert len(m.previous_saved_paths) == 5

    def test_one_deletes_the_previous_checkpoint(self, monkeypatch):
        """Contrast case: this is what the scripts did before the export-only policy."""
        m = _manager(monkeypatch)
        first, second = self._mk(1), self._mk(2)
        m.ensure_checkpoint_capacity(1)
        m.register_checkpoint(first, 1)
        assert os.path.exists(first), "the trim happens after the next save, not before"
        m.ensure_checkpoint_capacity(1)
        m.register_checkpoint(second, 1)
        assert not os.path.exists(first)
        assert os.path.exists(second)
