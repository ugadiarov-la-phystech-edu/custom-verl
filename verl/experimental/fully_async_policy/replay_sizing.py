# Copyright 2025 Meituan Ltd. and/or its affiliates
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
"""Sizing helpers shared by the fully-async TRAINER and ROLLOUTER (no ray/torch imports).

- trainer_dp_size / first_minibatch_groups / first_update_groups: the trainer's data-parallel size and the
  size of the first replay update when replay_buffer.requires_mini_batches < 1 (see fully_async_trainer.py).
- parse_concurrency_ramp / concurrency_cap: the rollouter's staged in-flight cap
  (async_training.concurrency_ramp), keyed on the number of groups DELIVERED to the message queue.
- stratified_minibatch_buckets / order_rows_for_minibatches: how the trainer splits one decoupled-objective
  update (K optimizer mini-batches) into its mini-batches: whole groups, fresh and replayed groups spread evenly,
  replayed ones stratified by staleness, rows laid out so each DP rank's contiguous chunk reads as the K
  mini-batches in order.
"""

import math

__all__ = [
    "trainer_dp_size",
    "first_minibatch_groups",
    "first_update_groups",
    "parse_concurrency_ramp",
    "concurrency_cap",
    "stratified_minibatch_buckets",
    "order_rows_for_minibatches",
]


def trainer_dp_size(config) -> int:
    """Data-parallel size of the TRAINER worker group, from the config alone (the trainer
    never queries the workers' parallel state): trainer GPUs divided by the model-parallel
    degree — tp x pp x cp for Megatron, ulysses_sequence_parallel_size for fsdp/fsdp2."""
    gpus = int(config.trainer.nnodes) * int(config.trainer.n_gpus_per_node)
    actor_cfg = config.actor_rollout_ref.actor
    strategy = str(actor_cfg.get("strategy", "megatron"))
    if strategy == "megatron":
        mp = int(actor_cfg.megatron.get("tensor_model_parallel_size", 1))
        mp *= int(actor_cfg.megatron.get("pipeline_model_parallel_size", 1))
        mp *= int(actor_cfg.megatron.get("context_parallel_size", 1))
    else:
        mp = int(actor_cfg.get("ulysses_sequence_parallel_size", 1))
    if mp < 1 or gpus % mp != 0:
        raise ValueError(f"trainer GPUs ({gpus}) are not divisible by the model-parallel degree ({mp})")
    return gpus // mp


def first_minibatch_groups(requires_mini_batches: float, mini_size: int, n: int, dp: int) -> int | None:
    """replay_buffer.requires_mini_batches in (0, 1): the size, in GROUPS, of the FIRST
    replay mini-batch — the smallest g >= requires_mini_batches x mini_size whose g x n
    sequences split evenly over the dp trainer ranks, capped at mini_size. None (no special
    first mini-batch) for values >= 1, which keep the watermark semantics. E.g. 0.5 x 33
    = 16.5 -> 17; 17 x 16 = 272 does not divide by dp=3, 18 x 16 = 288 does -> 18."""
    rmb = float(requires_mini_batches)
    if rmb <= 0:
        raise ValueError(f"replay_buffer.requires_mini_batches must be > 0, got {requires_mini_batches!r}")
    if rmb >= 1:
        return None
    if mini_size < 1 or n < 1 or dp < 1:
        raise ValueError(f"invalid mini_size={mini_size}, n={n}, dp={dp}")
    g = max(1, math.ceil(rmb * mini_size - 1e-9))
    while g < mini_size and (g * n) % dp != 0:
        g += 1
    return min(g, mini_size)


def first_update_groups(
    requires_mini_batches: float, mini_size: int, minibatches_per_update: int, n: int, dp: int
) -> int | None:
    """The size, in GROUPS, of the FIRST replay update when requires_mini_batches is in (0, 1); None otherwise.

    One mini-batch per update (minibatches_per_update=1, the bypass objective) is exactly
    first_minibatch_groups. With K > 1 optimizer mini-batches per update (the decoupled objective,
    async_training.require_batches=K) the first update is requires_mini_batches x K x mini_size groups rounded UP
    to a whole number of mini-batches, so every optimizer step of it is a full mini_size-group mini-batch (whose
    sequences already split over the dp ranks). E.g. 0.5 x 4 x 36 = 72 groups = 2 mini-batches."""
    k = int(minibatches_per_update)
    if k < 1:
        raise ValueError(f"minibatches_per_update must be >= 1, got {minibatches_per_update!r}")
    if k == 1:
        return first_minibatch_groups(requires_mini_batches, mini_size, n, dp)
    rmb = float(requires_mini_batches)
    if rmb <= 0:
        raise ValueError(f"replay_buffer.requires_mini_batches must be > 0, got {requires_mini_batches!r}")
    if rmb >= 1:
        return None
    if mini_size < 1 or n < 1 or dp < 1:
        raise ValueError(f"invalid mini_size={mini_size}, n={n}, dp={dp}")
    minibatches = max(1, math.ceil(rmb * k - 1e-9))
    return min(minibatches, k) * mini_size


def parse_concurrency_ramp(value) -> list[int]:
    """async_training.concurrency_ramp -> list of per-engine caps for the warm-up stages.

    Accepts None / "" / "null" / [] (ramp OFF -> []), a list (or OmegaConf ListConfig) of positive
    ints, or the string form "[4,8,16]" / "4,8,16". Stages must be non-decreasing."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in ("", "null", "none", "[]"):
            return []
        items = [t for t in text.strip("[]() ").replace(";", ",").split(",") if t.strip()]
    else:
        try:
            items = list(value)
        except TypeError as e:
            raise ValueError(f"async_training.concurrency_ramp must be a list of positive ints, got {value!r}") from e
    ramp: list[int] = []
    for item in items:
        try:
            f = float(item)
        except (TypeError, ValueError) as e:
            raise ValueError(f"async_training.concurrency_ramp entries must be ints, got {item!r}") from e
        if f != int(f) or int(f) < 1:
            raise ValueError(f"async_training.concurrency_ramp entries must be positive ints, got {item!r}")
        ramp.append(int(f))
    for a, b in zip(ramp, ramp[1:], strict=False):
        if b < a:
            raise ValueError(f"async_training.concurrency_ramp must be non-decreasing, got {ramp}")
    return ramp


def concurrency_cap(
    delivered: int,
    ramp: list[int],
    n_engines: int,
    first_size: int,
    mini_size: int,
    full_cap: int,
    hard_cap: int | None = None,
) -> int:
    """In-flight group cap for the rollouter given `delivered` groups pushed to the queue so far.

    Stage i (cap ramp[i] x n_engines) is active while delivered < T_i, with T_0 = first_size (the
    trainer's first mini-batch) and T_i = first_size + i x mini_size; after the last stage the cap is
    `full_cap` (n_engines x bsz_per_dp_rank, today's value). Every stage is bounded by `hard_cap`
    (the staleness quota max_required_samples) and never below 1. An empty ramp returns full_cap."""
    cap = full_cap
    for i, per_engine in enumerate(ramp):
        threshold = first_size + i * mini_size
        if delivered < threshold:
            cap = per_engine * n_engines
            break
    if hard_cap is not None:
        cap = min(cap, hard_cap)
    return max(1, int(cap))


def stratified_minibatch_buckets(n_fresh: int, replay_staleness, k: int, rng) -> list[list[int]]:
    """Assign the groups of one composed update to its k optimizer mini-batches ("buckets").

    The update is ``ReplayBuffer.compose_minibatch``'s list: positions ``0..n_fresh-1`` are the fresh groups,
    positions ``n_fresh..`` the replayed ones, whose staleness (in versions) is ``replay_staleness``. Returns k lists of
    group positions, each sorted ascending.

    * Fresh groups are shuffled (``rng.permutation``).
    * Replayed groups are sorted by staleness (stable: ties keep composition order), cut into consecutive blocks of k
      (one staleness stratum each) and shuffled within each block.
    * The fresh sequence followed by the replayed one is dealt round-robin: position j -> bucket j % k. Every bucket
      gets the same number of groups (the caller guarantees ``total % k == 0``); fresh and replayed counts differ by
      at most one between buckets; each stratum's k groups land in k distinct buckets whatever offset the fresh
      count leaves, so every bucket sees the whole staleness range.

    A fresh-only update (empty ``replay_staleness``), a replay-only one (``n_fresh=0``) and k=1 need no special case.
    ``rng`` is a ``numpy.random.Generator``.
    """
    k = int(k)
    n_fresh = int(n_fresh)
    staleness = [float(x) for x in replay_staleness]
    total = n_fresh + len(staleness)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n_fresh < 0:
        raise ValueError(f"n_fresh must be >= 0, got {n_fresh}")
    if total % k != 0:
        raise ValueError(f"{total} groups do not split into {k} equal mini-batches")
    order = [int(i) for i in rng.permutation(n_fresh)] if n_fresh else []
    replay_sorted = sorted(range(len(staleness)), key=lambda i: (staleness[i], i))
    for start in range(0, len(replay_sorted), k):
        block = replay_sorted[start : start + k]
        order.extend(n_fresh + block[int(j)] for j in rng.permutation(len(block)))
    buckets = [[] for _ in range(k)]
    for j, g in enumerate(order):
        buckets[j % k].append(g)
    return [sorted(b) for b in buckets]


def order_rows_for_minibatches(buckets, group_rows, seqlens, dp: int, balance: bool = True) -> list[int]:
    """Row permutation laying the buckets out for the actor update.

    ``group_rows[g]`` is the list of row indices of group g in the assembled batch; ``seqlens[r]`` is row r's token
    count. The update reaches the dp ranks as contiguous equal chunks, and each rank reads its chunk as the
    mini-batches in order (``make_iterator``, no shuffle), so the returned order is RANK-MAJOR: rank 0's share of
    bucket 0, rank 0's share of bucket 1, ..., then rank 1's ... With that layout optimizer step m on every rank is
    exactly bucket m.

    Each bucket is split over the ranks into equal-size parts: token-balanced
    (``get_seqlen_balanced_partitions`` over ``calculate_workload``, as ``_balance_batch`` does) when ``balance``,
    else contiguously. Rows keep ascending index order inside a part.
    """
    dp = int(dp)
    if dp < 1:
        raise ValueError(f"dp must be >= 1, got {dp}")
    bucket_rows = [[r for g in b for r in group_rows[g]] for b in buckets]
    sizes = {len(rows) for rows in bucket_rows}
    if len(sizes) != 1:
        raise ValueError(f"mini-batches have different row counts: {[len(rows) for rows in bucket_rows]}")
    size = sizes.pop()
    if size % dp != 0:
        raise ValueError(f"a mini-batch of {size} rows does not split over {dp} data-parallel ranks")
    per_rank = size // dp
    parts = []  # parts[m][r] = rows of bucket m on rank r
    for rows in bucket_rows:
        if balance and dp > 1:
            import torch

            from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions

            workload = calculate_workload(torch.tensor([int(seqlens[r]) for r in rows])).tolist()
            partitions = get_seqlen_balanced_partitions(workload, k_partitions=dp, equal_size=True)
            parts.append([sorted(rows[i] for i in part) for part in partitions])
        else:
            parts.append([rows[r * per_rank : (r + 1) * per_rank] for r in range(dp)])
    return [row for r in range(dp) for m in range(len(parts)) for row in parts[m][r]]
