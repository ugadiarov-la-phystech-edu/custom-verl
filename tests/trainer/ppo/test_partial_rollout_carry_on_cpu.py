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
"""
Tests for partial-rollout carry-over (VERL_PARTIAL_ROLLOUT=1) fresh-prompt sourcing on
RayPPOTrainer, with mixed (non-integer) gen_batch_size / train_batch_size ratios.

Covers `_carry_epoch_source` (cold-start over-sample, leftover buffering across steps and
epochs, exact sample accounting) and `validate_partial_rollout_config` (the config gate,
including the num_workers-divides-gen_batch_size requirement).
"""

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, validate_partial_rollout_config


class _FakeLoader:
    """Stands in for the train StatefulDataLoader: yields `num_batches` dict-batches of
    `batch_size` distinct sample ids (0..N-1) per epoch, like a non-shuffling sampler with
    drop_last=True."""

    def __init__(self, num_batches: int, batch_size: int):
        self.num_batches = num_batches
        self.batch_size = batch_size

    def __iter__(self):
        for b in range(self.num_batches):
            start = b * self.batch_size
            yield {"idx": torch.arange(start, start + self.batch_size)}

    def __len__(self):
        return self.num_batches


def _make_trainer(tbs: int, gen_bs: int, num_batches: int) -> RayPPOTrainer:
    # _carry_epoch_source touches only config.data, train_dataloader and the two carry
    # attributes reset at the top of fit(); no worker setup needed.
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create({"data": {"train_batch_size": tbs, "gen_batch_size": gen_bs}})
    trainer.train_dataloader = _FakeLoader(num_batches, tbs)
    trainer._carry_first_step = True
    trainer._carry_leftover = None
    return trainer


def _run_epochs(trainer: RayPPOTrainer, epochs: int) -> list[list[int]]:
    """Drive the source the way fit() does (a fresh generator per epoch) and return the
    yielded draws as lists of sample ids."""
    draws = []
    for _ in range(epochs):
        for batch in trainer._carry_epoch_source():
            draws.append(batch.batch["idx"].tolist())
    return draws


def _cold_start_extra_steps(gen_bs: int, tbs: int) -> int:
    # Mirrors the total_training_steps adjustment in _create_dataloader.
    return (gen_bs + tbs - 1) // tbs - 1


# ---------------------------------------------------------------------------
# _carry_epoch_source
# ---------------------------------------------------------------------------


def test_integer_multiple_cold_start():
    # Regression: gen_bs = 2 * tbs behaves as before — first draw is two dataloader
    # batches, every later draw is one, nothing buffered.
    trainer = _make_trainer(tbs=4, gen_bs=8, num_batches=5)
    draws = _run_epochs(trainer, epochs=1)
    assert [len(d) for d in draws] == [8, 4, 4, 4]
    assert draws[0] == list(range(8))
    assert [d[0] for d in draws[1:]] == [8, 12, 16]
    assert trainer._carry_leftover is None


def test_fractional_cold_start_buffers_surplus():
    # gen_bs = 1.5 * tbs: the cold start pulls ceil(6/4)=2 batches, uses 6 samples and
    # buffers the remaining 2 for the next draw.
    trainer = _make_trainer(tbs=4, gen_bs=6, num_batches=5)
    gen = trainer._carry_epoch_source()
    first = next(gen)
    assert first.batch["idx"].tolist() == [0, 1, 2, 3, 4, 5]
    assert trainer._carry_first_step is False
    assert trainer._carry_leftover is not None
    assert trainer._carry_leftover.batch["idx"].tolist() == [6, 7]
    second = next(gen)
    # The buffered surplus is consumed before pulling batch 2.
    assert second.batch["idx"].tolist() == [6, 7, 8, 9]


def test_fractional_no_sample_lost_or_duplicated_within_epoch():
    trainer = _make_trainer(tbs=4, gen_bs=6, num_batches=5)
    draws = _run_epochs(trainer, epochs=1)
    # 20 samples: cold start 6, then 4 per step -> 4 full draws, 2 samples buffered.
    assert [len(d) for d in draws] == [6, 4, 4, 4]
    consumed = [i for d in draws for i in d]
    assert consumed == list(range(18))
    assert trainer._carry_leftover.batch["idx"].tolist() == [18, 19]


def test_leftover_persists_across_epochs():
    trainer = _make_trainer(tbs=4, gen_bs=6, num_batches=5)
    draws = _run_epochs(trainer, epochs=2)
    # Epoch 1: 4 draws (2 samples buffered). Epoch 2: buffered [18,19] + fresh 20 -> 5 draws,
    # ending with 2 samples buffered again.
    assert [len(d) for d in draws] == [6, 4, 4, 4, 4, 4, 4, 4, 4]
    # Epoch 2's first draw starts with epoch 1's leftover, then continues with fresh ids.
    assert draws[4] == [18, 19, 0, 1]
    consumed = [i for d in draws for i in d]
    assert consumed == list(range(18)) + [18, 19] + list(range(18))
    assert trainer._carry_leftover.batch["idx"].tolist() == [18, 19]


@pytest.mark.parametrize(
    "tbs,gen_bs,num_batches,epochs",
    [
        (4, 8, 5, 2),  # integer k=2
        (4, 6, 5, 2),  # k=1.5
        (4, 9, 5, 2),  # k=2.25
        (4, 5, 3, 3),  # k=1.25
        (2, 6, 4, 1),  # k=3 integer, single epoch
    ],
)
def test_total_steps_matches_dataloader_adjustment(tbs, gen_bs, num_batches, epochs):
    # The number of draws the source actually produces must equal the total_training_steps
    # computed in _create_dataloader: len(dataloader)*epochs - (ceil(gen/tbs) - 1).
    trainer = _make_trainer(tbs=tbs, gen_bs=gen_bs, num_batches=num_batches)
    draws = _run_epochs(trainer, epochs=epochs)
    assert len(draws) == num_batches * epochs - _cold_start_extra_steps(gen_bs, tbs)
    # Every draw is full-sized (pool == gen_batch_size groups invariant).
    assert [len(d) for d in draws] == [gen_bs] + [tbs] * (len(draws) - 1)


def test_cold_start_spans_epochs_on_tiny_dataset():
    # One epoch holds fewer samples than gen_batch_size: the cold start stashes what it got,
    # stays pending, and completes with the next epoch's data instead of yielding a short pool.
    trainer = _make_trainer(tbs=2, gen_bs=6, num_batches=2)
    gen = trainer._carry_epoch_source()
    assert list(gen) == []
    assert trainer._carry_first_step is True
    assert trainer._carry_leftover.batch["idx"].tolist() == [0, 1, 2, 3]
    draws = _run_epochs(trainer, epochs=1)
    assert [len(d) for d in draws] == [6, 2]
    assert draws[0] == [0, 1, 2, 3, 0, 1]


# ---------------------------------------------------------------------------
# validate_partial_rollout_config
# ---------------------------------------------------------------------------


def _cfg(
    tbs=4,
    gen_bs=8,
    num_workers=1,
    calculate_log_probs=True,
    default_agent_loop="single_turn_agent",
    adv_estimator="grpo",
    name="vllm",
    scheduling_policy="priority",
):
    return OmegaConf.create(
        {
            "data": {"train_batch_size": tbs, "gen_batch_size": gen_bs},
            "algorithm": {"adv_estimator": adv_estimator},
            "actor_rollout_ref": {
                "rollout": {
                    "agent": {
                        "default_agent_loop": default_agent_loop,
                        "agent_loop_config_path": None,
                        "num_workers": num_workers,
                    },
                    "calculate_log_probs": calculate_log_probs,
                    "name": name,
                    "scheduling_policy": scheduling_policy,
                }
            },
        }
    )


def test_validate_accepts_integer_and_fractional_multiples():
    validate_partial_rollout_config(_cfg(tbs=4, gen_bs=8, num_workers=4))
    validate_partial_rollout_config(_cfg(tbs=4, gen_bs=6, num_workers=2))
    validate_partial_rollout_config(_cfg(tbs=4, gen_bs=6, num_workers=3))  # 3 | 6 though 3 ∤ 4
    validate_partial_rollout_config(_cfg(tbs=384, gen_bs=576, num_workers=8))  # k=1.5


def test_validate_rejects_gen_bs_not_greater_than_tbs():
    with pytest.raises(ValueError, match="gen_batch_size > data.train_batch_size"):
        validate_partial_rollout_config(_cfg(tbs=4, gen_bs=4))
    with pytest.raises(ValueError, match="gen_batch_size > data.train_batch_size"):
        validate_partial_rollout_config(_cfg(tbs=4, gen_bs=2))


def test_validate_num_workers_must_divide_gen_batch_size():
    with pytest.raises(ValueError, match="num_workers to divide"):
        validate_partial_rollout_config(_cfg(tbs=4, gen_bs=6, num_workers=4))  # 4 | tbs but 4 ∤ 6


def test_validate_rejects_oversample_discard_combo():
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_partial_rollout_config(_cfg(), oversample_discard=True)


def test_validate_rejects_remax():
    with pytest.raises(ValueError, match="remax"):
        validate_partial_rollout_config(_cfg(adv_estimator="remax"))


def test_validate_requires_rollout_log_probs():
    with pytest.raises(ValueError, match="calculate_log_probs=True"):
        validate_partial_rollout_config(_cfg(calculate_log_probs=False))


def test_validate_requires_single_turn_agent_loop():
    with pytest.raises(ValueError, match="single-turn agent loop"):
        validate_partial_rollout_config(_cfg(default_agent_loop="tool_agent"))


def test_validate_fcfs_hint_does_not_raise(capsys):
    validate_partial_rollout_config(_cfg(scheduling_policy="fcfs"))
    assert "scheduling_policy=priority" in capsys.readouterr().out
