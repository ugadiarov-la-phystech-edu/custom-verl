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
"""Splitting one decoupled-objective replay update into its K optimizer mini-batches (no GPU, no Ray cluster).

- replay_sizing.stratified_minibatch_buckets: whole groups per mini-batch, equal sizes, fresh / replayed counts
  within one of each other, replayed groups stratified by staleness; fresh-only, replay-only and k=1 updates;
  seeded and deterministic.
- replay_sizing.order_rows_for_minibatches: the rank-major row layout under the actor dispatch (contiguous equal
  chunks per DP rank, read in order as the mini-batches), token balance per rank inside each mini-batch.
- FullyAsyncTrainer._build_replay_batch / _fit_replay_step: the assembled update, chunked and read the way the
  workers read it, yields exactly the planned mini-batches with every per-row tensor still aligned; the bypass and
  one-mini-batch paths keep the trainer's usual balancing.
"""

import asyncio

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.detach_utils import RolloutSample
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer as _TrainerActor
from verl.experimental.fully_async_policy.replay_buffer import ReplayBuffer
from verl.experimental.fully_async_policy.replay_sizing import (
    order_rows_for_minibatches,
    stratified_minibatch_buckets,
)
from verl.protocol import DataProto

FullyAsyncTrainer = (
    _TrainerActor.__ray_metadata__.modified_class if hasattr(_TrainerActor, "__ray_metadata__") else _TrainerActor
)

PROMPT_LEN = 3
RESP_LEN = 8


def _rng(seed=0):
    return np.random.default_rng(seed)


def _staleness(n, seed=1):
    return list(np.random.default_rng(seed).integers(1, 33, size=n))


# ------------------------------------------------------------------ the bucket planner


COUNT_CASES = [
    (72, 72, 4),
    (100, 44, 4),
    (144, 0, 4),  # fresh-only
    (0, 144, 4),  # replay-only
    (36, 36, 2),  # the 72-group first update: 2 mini-batches
    (5, 3, 4),
    (1, 7, 4),
    (7, 1, 4),
    (3, 3, 3),
]


@pytest.mark.parametrize("n_fresh, n_replay, k", COUNT_CASES)
def test_buckets_partition_the_update_into_equal_whole_group_minibatches(n_fresh, n_replay, k):
    buckets = stratified_minibatch_buckets(n_fresh, _staleness(n_replay), k, _rng())
    assert len(buckets) == k
    flat = [g for b in buckets for g in b]
    assert sorted(flat) == list(range(n_fresh + n_replay))  # every group exactly once
    assert {len(b) for b in buckets} == {(n_fresh + n_replay) // k}
    assert all(b == sorted(b) for b in buckets)


@pytest.mark.parametrize("n_fresh, n_replay, k", COUNT_CASES)
def test_fresh_and_replayed_counts_differ_by_at_most_one(n_fresh, n_replay, k):
    buckets = stratified_minibatch_buckets(n_fresh, _staleness(n_replay), k, _rng())
    fresh = [sum(g < n_fresh for g in b) for b in buckets]
    replay = [sum(g >= n_fresh for g in b) for b in buckets]
    assert max(fresh) - min(fresh) <= 1 and sum(fresh) == n_fresh
    assert max(replay) - min(replay) <= 1 and sum(replay) == n_replay


def test_72_plus_72_gives_18_plus_18_per_step():
    buckets = stratified_minibatch_buckets(72, _staleness(72), 4, _rng())
    assert [(sum(g < 72 for g in b), sum(g >= 72 for g in b)) for b in buckets] == [(18, 18)] * 4


def test_fresh_only_update():
    buckets = stratified_minibatch_buckets(144, [], 4, _rng())
    assert [len(b) for b in buckets] == [36] * 4
    assert sorted(g for b in buckets for g in b) == list(range(144))


def test_one_minibatch_is_the_whole_update():
    assert stratified_minibatch_buckets(3, [5, 1], 1, _rng()) == [[0, 1, 2, 3, 4]]


@pytest.mark.parametrize("n_fresh", [0, 1, 2, 3, 5, 72])
def test_every_staleness_stratum_covers_every_minibatch(n_fresh):
    k = 4
    # 40 - n_fresh % 4 replayed groups: a whole number of mini-batches, and a last stratum shorter than k whenever the
    # fresh count leaves an offset
    staleness = _staleness(40 - n_fresh % k, seed=7)
    buckets = stratified_minibatch_buckets(n_fresh, staleness, k, _rng(3))
    where = {g: m for m, b in enumerate(buckets) for g in b}
    ranked = sorted(range(len(staleness)), key=lambda i: (staleness[i], i))
    for start in range(0, len(ranked), k):
        block = ranked[start : start + k]
        assert len({where[n_fresh + i] for i in block}) == len(block), "a stratum shares a mini-batch"


def _mean_staleness_spread(buckets, n_fresh, staleness):
    means = [np.mean([staleness[g - n_fresh] for g in b if g >= n_fresh]) for b in buckets]
    return max(means) - min(means)


def test_stratification_bounds_the_staleness_spread():
    k, n_fresh = 4, 72
    staleness = _staleness(72, seed=11)
    ranked = sorted(staleness)
    worst_block = max(ranked[i + k - 1] - ranked[i] for i in range(0, len(ranked), k))
    for seed in range(50):
        buckets = stratified_minibatch_buckets(n_fresh, staleness, k, _rng(seed))
        assert _mean_staleness_spread(buckets, n_fresh, staleness) <= worst_block + 1e-9


def test_stratified_spread_is_smaller_than_a_random_deal():
    k, n_fresh, n_replay = 4, 72, 72
    staleness = _staleness(n_replay, seed=5)
    strat, rand = [], []
    for seed in range(200):
        buckets = stratified_minibatch_buckets(n_fresh, staleness, k, _rng(seed))
        strat.append(_mean_staleness_spread(buckets, n_fresh, staleness))
        perm = np.random.default_rng(10_000 + seed).permutation(n_replay)
        random_buckets = [[n_fresh + int(i) for i in perm[m::k]] for m in range(k)]
        rand.append(_mean_staleness_spread(random_buckets, n_fresh, staleness))
    assert max(strat) <= max(rand)
    assert np.mean(strat) < 0.5 * np.mean(rand)


def test_ties_in_staleness():
    buckets = stratified_minibatch_buckets(2, [4, 4, 4, 4, 9, 9], 4, _rng())
    assert sorted(g for b in buckets for g in b) == list(range(8))
    replay = [sum(g >= 2 for g in b) for b in buckets]
    assert max(replay) - min(replay) <= 1


def test_seeded_and_deterministic():
    staleness = _staleness(40)
    a = stratified_minibatch_buckets(32, staleness, 4, _rng(42))
    b = stratified_minibatch_buckets(32, staleness, 4, _rng(42))
    c = stratified_minibatch_buckets(32, staleness, 4, _rng(43))
    assert a == b and a != c
    counts = lambda bs: [(sum(g < 32 for g in x), sum(g >= 32 for g in x)) for x in bs]  # noqa: E731
    assert counts(a) == counts(c)


@pytest.mark.parametrize(
    "args, match",
    [((3, [1, 2], 4), "do not split"), ((4, [], 0), "k must be"), ((-1, [1], 1), "n_fresh")],
)
def test_planner_rejects_bad_input(args, match):
    n_fresh, staleness, k = args
    with pytest.raises(ValueError, match=match):
        stratified_minibatch_buckets(n_fresh, staleness, k, _rng())


# ------------------------------------------------------------------ the row layout


def _read_like_the_workers(order, dp, k):
    """Rows of each optimizer step: dispatch chunks the batch contiguously over dp ranks, each rank reads its chunk
    in order as k equal mini-batches (tu.make_iterator, no shuffle)."""
    per_rank = len(order) // dp
    per_step = per_rank // k
    steps = [[] for _ in range(k)]
    per_rank_steps = [[None] * k for _ in range(dp)]
    for r in range(dp):
        chunk = order[r * per_rank : (r + 1) * per_rank]
        for m in range(k):
            piece = chunk[m * per_step : (m + 1) * per_step]
            steps[m].extend(piece)
            per_rank_steps[r][m] = piece
    return steps, per_rank_steps


def _layout_case(n_groups=16, rows_per_group=4, k=4, dp=2, seed=0):
    g = np.random.default_rng(seed)
    group_rows = [list(range(i * rows_per_group, (i + 1) * rows_per_group)) for i in range(n_groups)]
    seqlens = list(g.integers(20, 2000, size=n_groups * rows_per_group))
    buckets = stratified_minibatch_buckets(n_groups // 2, _staleness(n_groups - n_groups // 2), k, _rng(seed))
    return buckets, group_rows, seqlens


@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("dp", [1, 2, 4])
def test_each_step_is_exactly_its_bucket(balance, dp):
    buckets, group_rows, seqlens = _layout_case(dp=dp)
    order = order_rows_for_minibatches(buckets, group_rows, seqlens, dp=dp, balance=balance)
    assert sorted(order) == list(range(len(seqlens)))  # a permutation
    steps, _ = _read_like_the_workers(order, dp, len(buckets))
    for m, bucket in enumerate(buckets):
        assert sorted(steps[m]) == sorted(r for gidx in bucket for r in group_rows[gidx])


def test_tokens_are_balanced_over_ranks_inside_each_step():
    dp, k = 4, 4
    buckets, group_rows, seqlens = _layout_case(n_groups=32, rows_per_group=8, k=k, dp=dp, seed=3)
    balanced = order_rows_for_minibatches(buckets, group_rows, seqlens, dp=dp, balance=True)
    plain = order_rows_for_minibatches(buckets, group_rows, seqlens, dp=dp, balance=False)

    def worst_ratio(order):
        _, per_rank = _read_like_the_workers(order, dp, k)
        ratios = []
        for m in range(k):
            sums = [sum(seqlens[r] for r in per_rank[rank][m]) for rank in range(dp)]
            ratios.append(max(sums) / min(sums))
        return max(ratios)

    assert worst_ratio(balanced) < 1.25
    assert worst_ratio(balanced) <= worst_ratio(plain)


def test_unbalanced_split_is_contiguous_inside_each_bucket():
    buckets, group_rows, seqlens = _layout_case(dp=2)
    order = order_rows_for_minibatches(buckets, group_rows, seqlens, dp=2, balance=False)
    _, per_rank = _read_like_the_workers(order, 2, len(buckets))
    for m, bucket in enumerate(buckets):
        rows = [r for gidx in bucket for r in group_rows[gidx]]
        assert per_rank[0][m] == rows[: len(rows) // 2] and per_rank[1][m] == rows[len(rows) // 2 :]


def test_layout_rejects_uneven_buckets_and_rows_not_divisible_by_dp():
    group_rows = [[0, 1], [2, 3], [4, 5]]
    with pytest.raises(ValueError, match="different row counts"):
        order_rows_for_minibatches([[0, 1], [2]], group_rows, [10] * 6, dp=1)
    with pytest.raises(ValueError, match="does not split"):
        order_rows_for_minibatches([[0], [1], [2]], group_rows, [10] * 6, dp=4)


# ------------------------------------------------------------------ the trainer


def _group(uid, rows, version, seed):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, RESP_LEN + 1, (rows,), generator=g)
    mask = (torch.arange(RESP_LEN)[None, :] < lengths[:, None]).long()
    scores = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(rows)], dtype=np.float32)
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.ones(rows, PROMPT_LEN, dtype=torch.long),
            "responses": torch.ones(rows, RESP_LEN, dtype=torch.long),
            "input_ids": torch.ones(rows, PROMPT_LEN + RESP_LEN, dtype=torch.long),
            "attention_mask": torch.cat([torch.ones(rows, PROMPT_LEN, dtype=torch.long), mask], dim=1),
            "response_mask": mask,
            # encodes (group, row) so alignment after the reorder can be checked exactly
            "rollout_log_probs": -(torch.full((rows, RESP_LEN), float(seed)) + torch.arange(rows)[:, None] / 100.0),
        },
        non_tensors={
            "uid": np.array([uid] * rows, dtype=object),
            "row_in_group": np.arange(rows, dtype=object),
            "min_global_steps": np.array([version] * rows, dtype=object),
            "max_global_steps": np.array([version] * rows, dtype=object),
            "reward_scalar": scores,
            "advantage_scalar": (scores - scores.mean() + seed).astype(np.float32),
        },
        meta_info={"metrics": [{"generate_sequences": 0.1, "tool_calls": 0.0}] * rows},
    )
    return RolloutSample(full_batch=batch, sample_id=uid, epoch=0, rollout_status={}, group_version=version)


def _trainer(objective="decoupled", minibatches=4, mini=4, rows=4, dp=2, balance=True):
    t = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
    t.replay_objective = objective
    t.replay_update_minibatches = minibatches
    t.replay_updates_done = 0
    t.replay_split_seed = 7
    t.tokenizer = None
    t.balance_calls = []
    t._balance_batch = lambda batch, metrics: t.balance_calls.append(len(batch))
    t.config = OmegaConf.create(
        {
            "trainer": {"balance_batch": balance, "nnodes": 1, "n_gpus_per_node": dp},
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": mini, "strategy": "fsdp", "ulysses_sequence_parallel_size": 1},
                "rollout": {"temperature": 0.7, "n": rows, "multi_turn": {"enable": False}},
            },
        }
    )
    return t


def _update(n_fresh, n_replay, rows=4, current=40):
    """Composed entries (fresh first) and their compose info, as ReplayBuffer.compose_minibatch returns them."""
    buf = ReplayBuffer(tau=8.0, staleness_threshold=64, seed=0)
    staleness = _staleness(n_replay, seed=2)
    entries = []
    for i in range(n_fresh):
        entries.append(buf.add(_group(f"f{i}", rows, current - 1, seed=i), current - 1))
    for j, s in enumerate(staleness):
        entries.append(buf.add(_group(f"r{j}", rows, current - int(s), seed=1000 + j), current - int(s)))
    info = {
        "n_new": n_fresh,
        "n_replayed": n_replay,
        "staleness": [1] * n_fresh + [int(s) for s in staleness],
        "fresh_staleness": [1] * n_fresh,
        "times_trained": [0] * (n_fresh + n_replay),
    }
    return entries, info


def _steps_by_uid(batch, dp, k):
    order = list(range(len(batch)))
    steps, _ = _read_like_the_workers(order, dp, k)
    uids = batch.non_tensor_batch["uid"]
    return [[uids[r] for r in step] for step in steps]


@pytest.mark.parametrize(
    "n_fresh, n_replay, k",
    [(8, 8, 4), (10, 6, 4), (16, 0, 4), (0, 16, 4), (4, 4, 2)],
    ids=["half-half", "uneven", "fresh-only", "replay-only", "first-update-2-steps"],
)
def test_build_replay_batch_yields_the_planned_minibatches(n_fresh, n_replay, k):
    rows, dp = 4, 2
    t = _trainer(mini=(n_fresh + n_replay) // k, rows=rows, dp=dp)
    entries, info = _update(n_fresh, n_replay, rows=rows)
    batch = t._build_replay_batch(entries, info)
    assert t.balance_calls == []  # the stratified layout replaces _balance_batch
    assert len(batch) == (n_fresh + n_replay) * rows
    steps = _steps_by_uid(batch, dp, k)
    seen = set()
    for step in steps:
        groups = set(step)
        assert all(step.count(u) == rows for u in groups), "a group is split across optimizer steps"
        assert not groups & seen
        seen |= groups
    fresh = [sum(u.startswith("f") for u in set(s)) for s in steps]
    replay = [sum(u.startswith("r") for u in set(s)) for s in steps]
    assert sum(fresh) == n_fresh and max(fresh) - min(fresh) <= 1
    assert sum(replay) == n_replay and max(replay) - min(replay) <= 1
    m = t._replay_split_metrics
    assert m["replay/minibatches_in_update"] == k
    assert (m["replay/step_fresh_groups_min"], m["replay/step_fresh_groups_max"]) == (min(fresh), max(fresh))
    assert (m["replay/step_replayed_groups_min"], m["replay/step_replayed_groups_max"]) == (min(replay), max(replay))
    assert ("replay/step_replay_staleness_mean_min" in m) == (n_replay > 0)


def test_per_row_tensors_stay_aligned_after_the_reorder():
    rows = 4
    t = _trainer(mini=4, rows=rows, dp=2)
    entries, info = _update(8, 8, rows=rows)
    batch = t._build_replay_batch(entries, info)
    by_uid = {e.sample.full_batch.non_tensor_batch["uid"][0]: e.sample.full_batch for e in entries}
    for r in range(len(batch)):
        uid = batch.non_tensor_batch["uid"][r]
        i = int(batch.non_tensor_batch["row_in_group"][r])
        src = by_uid[uid]
        assert torch.equal(batch.batch["rollout_log_probs"][r], src.batch["rollout_log_probs"][i])
        assert torch.equal(batch.batch["response_mask"][r], src.batch["response_mask"][i])
        length = int(src.batch["response_mask"][i].sum())
        advantage = float(src.non_tensor_batch["advantage_scalar"][i])
        assert batch.batch["advantages"][r, 0].item() == pytest.approx(advantage)
        assert batch.batch["token_level_scores"][r, length - 1].item() == pytest.approx(
            float(src.non_tensor_batch["reward_scalar"][i])
        )


def test_replay_staleness_is_stratified_across_steps():
    rows, dp, k = 4, 2, 4
    t = _trainer(mini=8, rows=rows, dp=dp)
    entries, info = _update(8, 24, rows=rows)
    batch = t._build_replay_batch(entries, info)
    steps = _steps_by_uid(batch, dp, k)
    uids = [e.sample.full_batch.non_tensor_batch["uid"][0] for e in entries]
    staleness = dict(zip(uids, info["staleness"], strict=True))
    ranked = sorted(info["staleness"][8:])
    worst_block = max(ranked[i + k - 1] - ranked[i] for i in range(0, len(ranked), k))
    means = [np.mean([staleness[u] for u in set(s) if u.startswith("r")]) for s in steps]
    assert max(means) - min(means) <= worst_block + 1e-9
    m = t._replay_split_metrics
    assert m["replay/step_replay_staleness_mean_max"] - m["replay/step_replay_staleness_mean_min"] == pytest.approx(
        max(means) - min(means)
    )


def test_the_split_changes_per_update_but_is_reproducible():
    entries, info = _update(8, 8)
    t1, t2 = _trainer(mini=4), _trainer(mini=4)
    a = t1._build_replay_batch(entries, info).non_tensor_batch["uid"].tolist()
    b = t2._build_replay_batch(entries, info).non_tensor_batch["uid"].tolist()
    t2.replay_updates_done = 1
    c = t2._build_replay_batch(entries, info).non_tensor_batch["uid"].tolist()
    assert a == b and a != c


@pytest.mark.parametrize(
    "kw, info_given",
    [
        ({"objective": "bypass", "minibatches": 1}, True),  # bypass: one mini-batch per update
        ({"objective": "decoupled", "minibatches": 1}, True),  # decoupled with K=1
        ({"objective": "decoupled", "minibatches": 4, "mini": 16}, True),  # update of exactly one mini-batch
        ({"objective": "decoupled", "minibatches": 4}, False),  # no compose info: legacy call
    ],
)
def test_single_minibatch_paths_keep_the_usual_balancing(kw, info_given):
    t = _trainer(**{"mini": 4, **kw})
    entries, info = _update(8, 8)
    batch = t._build_replay_batch(entries, info if info_given else None)
    assert t.balance_calls == [len(batch)]  # _balance_batch ran (the stub keeps the concatenation order)
    expected = [e.sample.full_batch.non_tensor_batch["uid"][0] for e in entries for _ in range(4)]
    assert batch.non_tensor_batch["uid"].tolist() == expected
    assert t._replay_split_metrics == {}


def test_balance_off_still_stratifies():
    t = _trainer(mini=4, balance=False)
    entries, info = _update(8, 8)
    batch = t._build_replay_batch(entries, info)
    assert t.balance_calls == []
    steps = _steps_by_uid(batch, 2, 4)
    assert all(all(s.count(u) == 4 for u in set(s)) for s in steps)


def test_info_must_cover_the_update():
    t = _trainer(mini=4)
    entries, info = _update(8, 8)
    info = {**info, "staleness": info["staleness"][:-1]}
    with pytest.raises(AssertionError, match="composition info"):
        t._build_replay_batch(entries, info)


def test_fit_replay_step_passes_the_compose_info_and_logs_the_split():
    rows = 4
    t = _trainer(mini=4, rows=rows, dp=2)
    entries, info = _update(10, 6, rows=rows)
    calls = []
    t.metrics = {}
    t.global_steps, t.epoch = 1, 0
    t.local_trigger_step, t.trigger_parameter_sync_step = 1, 1
    t.current_param_version = 40
    t.replay_buffer = ReplayBuffer(tau=8.0, staleness_threshold=64, seed=0)
    t.replay_buffer.entries = list(entries)
    t._replay_fresh_wait_s, t._replay_fresh_floor_waived = 0.0, 0
    t.replay_min_fresh_ratio = 0.0
    t.config = OmegaConf.merge(t.config, {"global_profiler": {"steps": None}})

    async def acquire():
        return entries, info

    t._acquire_replay_minibatch = acquire
    t._collect_metrics_from_samples = lambda batch, metrics: None
    t._fit_start_profile = lambda **k: None
    t._fit_stop_profile = lambda **k: None
    t._fit_compute_log_prob = lambda batch: batch
    t._replay_apply_rollout_correction = lambda batch: batch
    t._update_actor = lambda batch, mini_batch_size=None: (
        calls.append(("update_actor", len(batch), mini_batch_size))
        or DataProto(meta_info={"metrics": {"actor/pg_loss": [0.1]}})
    )

    async def update_weights():
        return None

    async def validate():
        return None

    t._fit_update_weights = update_weights
    t._fit_validate = validate
    t._fit_dump_data = lambda batch: None
    t._record_train_resource_utilization = lambda allocated_time: None
    t._fit_save_checkpoint = lambda: None
    t._fit_collect_metrics = lambda batch: None
    t._advance_virtual_clock = lambda: None
    t._open_virtual_step = lambda *a, **k: None
    t._fit_postprocess_step = lambda: calls.append(("metrics", dict(t.metrics)))
    asyncio.run(t._fit_replay_step())
    assert calls[0] == ("update_actor", 16 * rows, 4 * rows)
    metrics = calls[1][1]
    assert metrics["replay/minibatches_in_update"] == 4.0
    assert metrics["replay/step_fresh_groups_max"] - metrics["replay/step_fresh_groups_min"] <= 1
    assert metrics["replay/step_replayed_groups_max"] - metrics["replay/step_replayed_groups_min"] <= 1

