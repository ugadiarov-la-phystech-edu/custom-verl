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
"""ReplayBuffer and replay sizing helpers (pure Python), ported from custom_vcpo.

- ReplayBuffer: score formula, add/evict/rescore, one-shot freshness, mini-batch composition (fresh
  first, newest first, then a staleness x reuse weighted draw), state_dict round-trip incl. RNG state
- replay_sizing: first replay mini-batch size, trainer DP size, concurrency-ramp parsing and caps
"""

from types import SimpleNamespace

import numpy as np
import pytest
import ray.cloudpickle
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.replay_buffer import (
    GroupEntry,
    ReplayBuffer,
    reuse_score,
    sampling_weight,
    staleness_score,
)
from verl.experimental.fully_async_policy.replay_sizing import (
    concurrency_cap,
    first_minibatch_groups,
    parse_concurrency_ramp,
    trainer_dp_size,
)


def _sample(group_version=0):
    """Minimal stand-in for a RolloutSample as the buffer sees it."""
    return SimpleNamespace(group_version=group_version)


def _make_buffer(tau=4.0, staleness_threshold=8, seed=1234, **kwargs):
    return ReplayBuffer(tau=tau, staleness_threshold=staleness_threshold, seed=seed, **kwargs)


# ---------------------------------------------------------------- score & add


def test_staleness_score_halves_every_tau():
    assert staleness_score(0, tau=4.0) == pytest.approx(1.0)
    assert staleness_score(4, tau=4.0) == pytest.approx(0.5)
    assert staleness_score(8, tau=4.0) == pytest.approx(0.25)
    assert staleness_score(2, tau=2.0) == pytest.approx(0.5)


def test_add_stamps_entry_fields():
    buf = _make_buffer(tau=4.0)
    e0 = buf.add(_sample(group_version=6), current_version=10)
    e1 = buf.add(_sample(group_version=10), current_version=10)
    assert e0.times_trained == 0 and e1.times_trained == 0
    assert (e0.insert_seq, e1.insert_seq) == (0, 1)
    assert e0.group_version == 6
    assert e0.score == pytest.approx(staleness_score(4, 4.0))
    assert e1.score == pytest.approx(1.0)
    assert buf.total_added == 2
    assert buf.size() == 2 and buf.untrained_count() == 2
    assert buf.pending_fresh == [e0, e1]


# ---------------------------------------------------------------- evict / rescore / mark_used


def test_evict_boundary_and_unseen_counting():
    buf = _make_buffer(staleness_threshold=2)
    at_threshold = buf.add(_sample(group_version=8), current_version=8)  # staleness 2 at v=10
    over_trained = buf.add(_sample(group_version=7), current_version=8)  # staleness 3 at v=10
    buf.add(_sample(group_version=6), current_version=8)  # over_untrained: staleness 4 at v=10
    fresh = buf.add(_sample(group_version=10), current_version=10)
    buf.mark_trained([over_trained])
    evicted, evicted_unseen = buf.evict(current_version=10)
    assert (evicted, evicted_unseen) == (2, 1)  # over_trained + over_untrained; only the latter unseen
    remaining = {e.insert_seq for e in buf.entries}
    assert remaining == {at_threshold.insert_seq, fresh.insert_seq}
    assert buf.evicted_total == 2 and buf.evicted_unseen_total == 1
    assert buf.evicted_trained_once_total == 1  # over_trained was trained exactly once


def test_evict_counts_trained_once_separately_from_unseen_and_multi():
    buf = _make_buffer(staleness_threshold=0)
    once = buf.add(_sample(group_version=0), current_version=0)
    twice = buf.add(_sample(group_version=0), current_version=0)
    buf.add(_sample(group_version=0), current_version=0)  # never trained
    buf.mark_trained([once, twice])
    buf.mark_trained([twice])
    evicted, evicted_unseen = buf.evict(current_version=1)
    assert (evicted, evicted_unseen) == (3, 1)
    assert buf.evicted_trained_once_total == 1


def test_evict_purges_pending_fresh():
    buf = _make_buffer(staleness_threshold=2)
    stale_pending = buf.add(_sample(group_version=0), current_version=0)  # staleness 10 at v=10
    kept_pending = buf.add(_sample(group_version=10), current_version=10)
    buf.evict(current_version=10)
    assert buf.pending_fresh == [kept_pending]
    assert stale_pending not in buf.entries
    selected, info = buf.compose_minibatch(1, current_version=10)
    assert selected == [kept_pending] and info["n_new"] == 1


def test_recompute_scores_tracks_current_version():
    buf = _make_buffer(tau=4.0, staleness_threshold=100)
    entry = buf.add(_sample(group_version=0), current_version=0)
    assert entry.score == pytest.approx(1.0)
    buf.recompute_scores(current_version=4)
    assert entry.score == pytest.approx(0.5)
    buf.recompute_scores(current_version=8)
    assert entry.score == pytest.approx(0.25)


def test_mark_trained_increments_and_keeps_entry():
    buf = _make_buffer()
    entry = buf.add(_sample(), current_version=0)
    buf.mark_trained([entry])
    buf.mark_trained([entry])
    assert entry.times_trained == 2
    assert buf.size() == 1 and buf.untrained_count() == 0


# ---------------------------------------------------------------- composition


def test_compose_fresh_priority_newest_arrived_first():
    buf = _make_buffer()
    entries = [buf.add(_sample(group_version=i), current_version=5) for i in range(5)]
    selected, info = buf.compose_minibatch(3, current_version=5)
    assert [e.insert_seq for e in selected] == [4, 3, 2]
    assert info["n_new"] == 3 and info["n_replayed"] == 0
    assert info["staleness"] == [e.staleness(5) for e in entries[4:1:-1]]
    # freshness is one-shot: the pending list is cleared wholesale
    assert buf.pending_fresh == []


def test_compose_fresh_overflow_goes_to_replay_pool():
    buf = _make_buffer()
    entries = [buf.add(_sample(group_version=5), current_version=5) for _ in range(7)]
    selected, info = buf.compose_minibatch(5, current_version=5)
    # the 5 newest-arrived fresh groups fill the mini-batch...
    assert selected == entries[:1:-1] and info["n_new"] == 5
    # ...the overflow (the window's 2 earliest arrivals) stays buffered as
    # plain replay candidates
    assert buf.pending_fresh == []
    assert entries[0] in buf.entries and entries[1] in buf.entries
    # next composition with no new arrivals is pure replay
    selected2, info2 = buf.compose_minibatch(5, current_version=5)
    assert info2["n_new"] == 0 and info2["n_replayed"] == 5


def test_freshness_is_one_shot():
    buf = _make_buffer(tau=1.0, staleness_threshold=100)
    weak_overflow = buf.add(_sample(group_version=0), current_version=10)  # score 2^-10
    strong = buf.add(_sample(group_version=10), current_version=10)  # score 1
    selected, info = buf.compose_minibatch(1, current_version=10)
    # weak_overflow (the earlier arrival) is fresh overflow: unselected,
    # never trained, freshness spent
    assert selected == [strong] and info["n_new"] == 1
    picks = 0
    for trial in range(100):
        buf.rng = np.random.default_rng(trial)
        fresh = buf.add(_sample(group_version=10), current_version=10)
        selected, info = buf.compose_minibatch(2, current_version=10)
        # only this round's arrival counts as fresh...
        assert info["n_new"] == 1 and selected[0] is fresh
        picks += int(weak_overflow in selected)
        buf.entries.remove(fresh)
    # ...and the never-trained overflow holds no priority: it competes on its
    # ~2^-10 weight against strong's 1.0 (the old is_new rule would have
    # selected it deterministically every round)
    assert weak_overflow.times_trained == 0
    assert picks < 10


def test_compose_mixes_fresh_and_weighted_replay_without_duplicates():
    buf = _make_buffer()
    old_entries = [buf.add(_sample(group_version=0), current_version=0) for _ in range(4)]
    buf.compose_minibatch(4, current_version=0)  # consume their freshness
    buf.mark_trained(old_entries)
    new = [buf.add(_sample(group_version=3), current_version=3) for _ in range(2)]
    selected, info = buf.compose_minibatch(4, current_version=3)
    assert info["n_new"] == 2 and info["n_replayed"] == 2
    assert selected[:2] == new[::-1]  # fresh-first, newest-arrived first
    assert len({id(e) for e in selected}) == 4  # no duplicates
    assert all(e in old_entries for e in selected[2:])


def test_compose_pure_replay_when_no_new_groups():
    buf = _make_buffer()
    buf.add(_sample(), current_version=0)
    buf.add(_sample(), current_version=0)
    buf.add(_sample(), current_version=0)
    buf.compose_minibatch(3, current_version=0)  # consume freshness
    selected, info = buf.compose_minibatch(2, current_version=0)
    assert info["n_new"] == 0 and info["n_replayed"] == 2
    assert len({id(e) for e in selected}) == 2


def test_compose_is_seed_deterministic():
    def build():
        buf = _make_buffer(seed=42)
        used = [buf.add(_sample(group_version=i), current_version=6) for i in range(6)]
        buf.compose_minibatch(6, current_version=6)  # consume freshness
        buf.mark_trained(used)
        return buf

    sel_a, _ = build().compose_minibatch(3, current_version=6)
    sel_b, _ = build().compose_minibatch(3, current_version=6)
    assert [e.insert_seq for e in sel_a] == [e.insert_seq for e in sel_b]


def test_compose_sampling_prefers_high_scores():
    picks = {0: 0, 1: 0}
    for trial in range(200):
        buf = _make_buffer(tau=1.0, staleness_threshold=100, seed=trial)
        recent = buf.add(_sample(group_version=10), current_version=10)  # staleness 0, score 1
        stale = buf.add(_sample(group_version=0), current_version=10)  # staleness 10, score 2^-10
        buf.compose_minibatch(2, current_version=10)  # consume freshness
        buf.mark_trained([recent, stale])
        selected, _ = buf.compose_minibatch(1, current_version=10)
        picks[selected[0].insert_seq] += 1
    assert picks[0] > 190  # ~1000:1 odds per draw


# ---------------------------------------------------------------- reuse decay


def test_reuse_score_halves_every_halflife_and_is_one_when_off():
    assert reuse_score(0, 2.0) == 1.0
    assert reuse_score(2, 2.0) == pytest.approx(0.5)
    assert reuse_score(4, 2.0) == pytest.approx(0.25)
    assert reuse_score(7, 1.0) == pytest.approx(2.0**-7)
    assert reuse_score(100, None) == 1.0


def test_sampling_weight_is_staleness_score_times_reuse_score():
    entry = GroupEntry(sample=_sample(), group_version=0, score=0.5, insert_seq=0, times_trained=3)
    assert sampling_weight(entry, None) == pytest.approx(0.5)
    assert sampling_weight(entry, 1.0) == pytest.approx(0.5 * 2.0**-3)
    assert sampling_weight(entry, 3.0) == pytest.approx(0.25)


def test_reuse_halflife_off_is_the_default_and_rejects_non_positive_or_non_finite():
    assert _make_buffer().reuse_halflife is None
    assert _make_buffer(reuse_halflife=None).reuse_halflife is None
    assert _make_buffer(reuse_halflife=0).reuse_halflife is None  # <= 0 means off
    assert _make_buffer(reuse_halflife=-1.0).reuse_halflife is None
    assert _make_buffer(reuse_halflife=2).reuse_halflife == 2.0
    with pytest.raises(AssertionError, match="finite"):
        _make_buffer(reuse_halflife=float("inf"))


def _fill_mixed_pool(buf, n=12, current_version=10):
    """A pool with a spread of staleness AND reuse counts, freshness consumed."""
    entries = [
        buf.add(_sample(group_version=current_version - (i % 5)), current_version=current_version) for i in range(n)
    ]
    buf.compose_minibatch(n, current_version=current_version)  # consume freshness (all-fresh mini-batch)
    for i, e in enumerate(entries):
        for _ in range(i % 4):
            buf.mark_trained([e])
    return entries


def _draw_sequence(buf, compositions=20, mini=4, current_version=10):
    out = []
    for _ in range(compositions):
        selected, _info = buf.compose_minibatch(mini, current_version=current_version)
        out.append([e.insert_seq for e in selected])
    return out


def test_reuse_halflife_off_reproduces_exact_draws():
    """The knob defaults to off; off must be bit-for-bit today's staleness-only draw, and an enormous
    half-life must converge to it (the decay factor -> 1)."""
    baseline = _make_buffer(seed=3)
    explicit_off = _make_buffer(seed=3, reuse_halflife=None)
    limit = _make_buffer(seed=3, reuse_halflife=1e12)
    for b in (baseline, explicit_off, limit):
        _fill_mixed_pool(b)
    reference = _draw_sequence(baseline)  # drawn once: each call advances that buffer's rng
    assert _draw_sequence(explicit_off) == reference
    assert _draw_sequence(limit) == reference


def test_reuse_halflife_changes_the_draws_when_on():
    baseline, decayed = _make_buffer(seed=3), _make_buffer(seed=3, reuse_halflife=1.0)
    _fill_mixed_pool(baseline)
    _fill_mixed_pool(decayed)
    assert _draw_sequence(baseline) != _draw_sequence(decayed)


def test_reuse_penalty_prefers_less_trained_at_equal_staleness():
    """Mirror of test_compose_sampling_prefers_high_scores: same staleness, one group already
    trained three times -> at nu=1 its weight is 1/8 of the untrained twin's."""
    picks = {"untrained": 0, "trained3": 0}
    for trial in range(300):
        buf = _make_buffer(tau=4.0, staleness_threshold=100, seed=trial, reuse_halflife=1.0)
        untrained = buf.add(_sample(group_version=10), current_version=10)
        trained3 = buf.add(_sample(group_version=10), current_version=10)
        buf.compose_minibatch(2, current_version=10)  # consume freshness
        for _ in range(3):
            buf.mark_trained([trained3])
        selected, _ = buf.compose_minibatch(1, current_version=10)
        picks["untrained" if selected[0] is untrained else "trained3"] += 1
    # expected 8:1 odds -> ~33 of 300 for the trained group
    assert 10 <= picks["trained3"] <= 65, picks


def test_reuse_penalty_composes_with_staleness():
    """One training at nu=1 costs the same weight as tau updates of age: a stale-but-untrained
    group (staleness 4 = tau, score 2^-1) and a fresh-but-trained-once group (staleness 0, decay
    2^-1) carry equal weight and split the draws ~50/50."""
    picks = {"stale_untrained": 0, "fresh_trained": 0}
    for trial in range(400):
        buf = _make_buffer(tau=4.0, staleness_threshold=100, seed=trial, reuse_halflife=1.0)
        stale_untrained = buf.add(_sample(group_version=6), current_version=10)
        fresh_trained = buf.add(_sample(group_version=10), current_version=10)
        buf.compose_minibatch(2, current_version=10)
        buf.mark_trained([fresh_trained])
        w_stale = sampling_weight(stale_untrained, buf.reuse_halflife)
        w_fresh = sampling_weight(fresh_trained, buf.reuse_halflife)
        assert w_stale == pytest.approx(w_fresh)
        selected, _ = buf.compose_minibatch(1, current_version=10)
        picks["stale_untrained" if selected[0] is stale_untrained else "fresh_trained"] += 1
    assert 150 <= picks["stale_untrained"] <= 250, picks


def test_reuse_penalty_never_starves_the_draw():
    """Heavily re-trained pools still fill the mini-batch: the decay shrinks weights, never zeroes
    them, and the existing underflow fallback covers the extreme."""
    buf = _make_buffer(tau=1.0, staleness_threshold=10**6, seed=0, reuse_halflife=0.5)
    entries = [buf.add(_sample(group_version=0), current_version=0) for _ in range(4)]
    buf.compose_minibatch(4, current_version=0)
    for _ in range(10):
        buf.mark_trained(entries)
    selected, info = buf.compose_minibatch(3, current_version=0)
    assert len(selected) == 3 and info["n_replayed"] == 3
    # underflow of BOTH factors -> uniform fallback still fills
    buf.recompute_scores(current_version=5000)
    for _ in range(3000):
        buf.mark_trained(entries)
    selected, info = buf.compose_minibatch(3, current_version=5000)
    assert len(selected) == 3 and info["n_replayed"] == 3


def test_compose_info_reports_times_trained_before_marking():
    buf = _make_buffer(seed=1)
    a = buf.add(_sample(group_version=0), current_version=0)
    b = buf.add(_sample(group_version=0), current_version=0)
    _, info = buf.compose_minibatch(2, current_version=0)
    assert info["times_trained"] == [0, 0]
    buf.mark_trained([a, b])
    buf.mark_trained([a])
    selected, info = buf.compose_minibatch(2, current_version=0)
    assert info["times_trained"] == [e.times_trained for e in selected]
    assert sorted(info["times_trained"]) == [1, 2]
    assert buf.times_trained_list() == [2, 1]


def test_compose_uniform_fallback_when_scores_underflow():
    buf = _make_buffer(tau=1.0, staleness_threshold=10**6)
    used = [buf.add(_sample(group_version=0), current_version=0) for _ in range(3)]
    buf.compose_minibatch(3, current_version=0)  # consume freshness
    buf.mark_trained(used)
    buf.recompute_scores(current_version=5000)  # 2^-5000 underflows to 0.0
    assert all(e.score == 0.0 for e in buf.entries)
    selected, info = buf.compose_minibatch(2, current_version=5000)
    assert len(selected) == 2 and info["n_replayed"] == 2


def test_compose_raises_below_mini_size():
    buf = _make_buffer()
    buf.add(_sample(), current_version=0)
    with pytest.raises(ValueError, match="watermark"):
        buf.compose_minibatch(2, current_version=0)


# ---------------------------------------------------------------- checkpoint round-trip


def test_state_dict_roundtrip_restores_entries_counters_and_rng():
    buf = _make_buffer(seed=7)
    entries = [buf.add(_sample(group_version=i), current_version=8) for i in range(8)]
    buf.compose_minibatch(6, current_version=8)  # 6 fresh consumed, 2 overflow
    buf.mark_trained(entries[:6])
    buf.add(_sample(group_version=8), current_version=8)  # pending at save time
    buf.evict(current_version=8)
    state = ray.cloudpickle.loads(ray.cloudpickle.dumps(buf.state_dict()))

    restored = _make_buffer(seed=999)  # seed overwritten by the restored RNG state
    restored.load_state_dict(state)
    assert restored.size() == buf.size()
    assert restored.untrained_count() == buf.untrained_count()
    assert restored.total_added == buf.total_added
    assert restored.evicted_total == buf.evicted_total
    assert restored.evicted_trained_once_total == buf.evicted_trained_once_total
    assert [e.insert_seq for e in restored.entries] == [e.insert_seq for e in buf.entries]
    assert [e.score for e in restored.entries] == [e.score for e in buf.entries]
    assert [e.times_trained for e in restored.entries] == [e.times_trained for e in buf.entries]
    assert [e.insert_seq for e in restored.pending_fresh] == [e.insert_seq for e in buf.pending_fresh]
    # identical RNG continuation and identical fresh set: the next draw matches
    sel_orig, info_orig = buf.compose_minibatch(4, current_version=8)
    sel_rest, info_rest = restored.compose_minibatch(4, current_version=8)
    assert [e.insert_seq for e in sel_orig] == [e.insert_seq for e in sel_rest]
    assert info_orig["n_new"] == info_rest["n_new"] == 1


def test_reuse_halflife_stays_config_driven_across_restore():
    """The half-life is a config knob like tau: a checkpoint written with one value must not
    override the value the resumed run was launched with; the restored counts DO feed the weights."""
    buf = _make_buffer(seed=7, reuse_halflife=1.0)
    entries = [buf.add(_sample(group_version=8), current_version=8) for _ in range(4)]
    buf.compose_minibatch(4, current_version=8)
    buf.mark_trained(entries[:2])
    state = ray.cloudpickle.loads(ray.cloudpickle.dumps(buf.state_dict()))
    assert "reuse_halflife" not in state

    restored_off = _make_buffer(seed=7)
    restored_off.load_state_dict(state)
    assert restored_off.reuse_halflife is None
    restored_on = _make_buffer(seed=7, reuse_halflife=1.0)
    restored_on.load_state_dict(state)
    assert restored_on.reuse_halflife == 1.0
    weights_on = [sampling_weight(e, restored_on.reuse_halflife) for e in restored_on.entries]
    weights_off = [sampling_weight(e, restored_off.reuse_halflife) for e in restored_off.entries]
    assert weights_on[:2] == pytest.approx([w / 2 for w in weights_off[:2]])
    assert weights_on[2:] == pytest.approx(weights_off[2:])
    # the original and its exact restore draw identically
    sel_a, _ = buf.compose_minibatch(2, current_version=8)
    sel_b, _ = restored_on.compose_minibatch(2, current_version=8)
    assert [e.insert_seq for e in sel_a] == [e.insert_seq for e in sel_b]


def test_load_state_dict_tolerates_legacy_is_new_payload():
    # Pre-freshness checkpoints carried is_new per entry and no pending_seqs.
    state = {
        "next_insert_seq": 2,
        "total_added": 2,
        "evicted_total": 0,
        "evicted_unseen_total": 0,
        "rng_state": None,
        "entries": [
            {"sample": _sample(0), "group_version": 0, "is_new": True, "score": 1.0, "insert_seq": 0},
            {"sample": _sample(0), "group_version": 0, "is_new": False, "score": 1.0, "insert_seq": 1},
        ],
    }
    buf = _make_buffer()
    buf.load_state_dict(state)
    assert buf.pending_fresh == []  # legacy unseen backlog gets no fresh priority
    assert [e.times_trained for e in buf.entries] == [0, 1]  # waste counter keeps meaning
    selected, info = buf.compose_minibatch(2, current_version=0)
    assert info["n_new"] == 0


# ---------------------------------------------------------------- rollouter insertion gate


def test_pending_fresh_count_tracks_add_compose_and_evict():
    buf = _make_buffer(staleness_threshold=2)
    assert buf.pending_fresh_count() == 0
    e_old = buf.add(_sample(group_version=0), current_version=0)
    buf.add(_sample(group_version=3), current_version=3)
    assert buf.pending_fresh_count() == 2
    buf.evict(current_version=3)  # e_old: staleness 3 > 2 -> gone from the pending list too
    assert e_old not in buf.entries
    assert buf.pending_fresh_count() == 1
    buf.compose_minibatch(1, current_version=3)
    assert buf.pending_fresh_count() == 0  # one-shot freshness
    assert buf.untrained_count() == 1  # untrained is a different thing


def test_compose_info_carries_the_fresh_prefix_staleness():
    buf = _make_buffer()
    for v in (10, 8):
        buf.add(_sample(group_version=v), current_version=10)
    buf.compose_minibatch(2, current_version=10)
    buf.mark_trained(buf.entries)
    buf.add(_sample(group_version=9), current_version=10)  # the only fresh one
    selected, info = buf.compose_minibatch(3, current_version=10)
    assert info["n_new"] == 1
    assert info["fresh_staleness"] == info["staleness"][:1] == [1]
    assert len(info["staleness"]) == 3
    # pure replay -> empty prefix
    _, info = buf.compose_minibatch(3, current_version=10)
    assert info["n_new"] == 0 and info["fresh_staleness"] == []


class TestFirstMinibatchGroups:
    """requires_mini_batches in (0, 1) sizes ONLY the first mini-batch: the smallest group count
    >= rmb x mini_size whose sequences (groups x n) split evenly over the trainer DP ranks."""

    @pytest.mark.parametrize(
        "rmb, mini, n, dp, expected",
        [
            (0.5, 33, 16, 3, 18),  # 16.5 -> 17 (272 % 3 != 0) -> 18 (288 % 3 == 0): the user's case
            (0.5, 33, 16, 4, 17),  # 17 x 16 = 272 divides by 4
            (0.5, 33, 16, 1, 17),  # no divisibility constraint
            (0.5, 34, 16, 3, 18),  # exactly 17.0 -> 17 not divisible -> 18
            (0.9, 33, 16, 3, 30),  # 29.7 -> 30, 480 % 3 == 0
            (0.99, 33, 16, 3, 33),  # 32.67 -> 33 = cap at mini_size
            (0.01, 33, 16, 3, 3),  # 0.33 -> 1 -> 2 -> 3 (48 % 3 == 0)
            (0.5, 2, 3, 2, 1),  # 1 x 3 = 3 is odd -> 2? no: g=1 -> 3 % 2 != 0 -> g=2 = cap  (see below)
        ],
    )
    def test_rounding(self, rmb, mini, n, dp, expected):
        if (rmb, mini, n, dp) == (0.5, 2, 3, 2):
            expected = 2  # 1 group = 3 seqs (odd); 2 groups = 6 seqs -> the cap coincides with the fix
        assert first_minibatch_groups(rmb, mini, n, dp) == expected

    @pytest.mark.parametrize("rmb", [1.0, 1.5, 2, 2.0])
    def test_at_least_one_keeps_the_watermark_semantics(self, rmb):
        assert first_minibatch_groups(rmb, 33, 16, 3) is None

    @pytest.mark.parametrize("bad", [0, 0.0, -1, -0.5])
    def test_non_positive_is_an_error(self, bad):
        with pytest.raises(ValueError, match="requires_mini_batches"):
            first_minibatch_groups(bad, 33, 16, 3)

    def test_result_always_splits_over_dp_or_hits_the_cap(self):
        for mini in (2, 5, 33, 64):
            for n in (1, 4, 16):
                for dp in (1, 2, 3, 4, 8):
                    for rmb in (0.1, 0.3, 0.5, 0.75, 0.999):
                        g = first_minibatch_groups(rmb, mini, n, dp)
                        assert 1 <= g <= mini
                        assert g >= rmb * mini - 1e-9
                        assert (g * n) % dp == 0 or g == mini


class TestTrainerDpSize:
    @staticmethod
    def _cfg(gpus, strategy="megatron", tp=1, pp=1, cp=1, ulysses=1, nnodes=1):
        return OmegaConf.create(
            {
                "trainer": {"nnodes": nnodes, "n_gpus_per_node": gpus},
                "actor_rollout_ref": {
                    "actor": {
                        "strategy": strategy,
                        "megatron": {
                            "tensor_model_parallel_size": tp,
                            "pipeline_model_parallel_size": pp,
                            "context_parallel_size": cp,
                        },
                        "ulysses_sequence_parallel_size": ulysses,
                    }
                },
            }
        )

    def test_megatron_pure_dp(self):
        assert trainer_dp_size(self._cfg(3)) == 3  # the 5+3 arms

    def test_megatron_model_parallel_divides(self):
        assert trainer_dp_size(self._cfg(4, tp=2)) == 2
        assert trainer_dp_size(self._cfg(8, tp=2, pp=2)) == 2
        assert trainer_dp_size(self._cfg(4, nnodes=2, tp=2)) == 4
        assert trainer_dp_size(self._cfg(8, cp=2)) == 4
        assert trainer_dp_size(self._cfg(8, tp=2, cp=2)) == 2

    def test_megatron_non_divisible_is_an_error(self):
        with pytest.raises(ValueError, match="not divisible"):
            trainer_dp_size(self._cfg(3, tp=2))

    def test_fsdp_uses_ulysses(self):
        assert trainer_dp_size(self._cfg(4, strategy="fsdp2", ulysses=2)) == 2
        assert trainer_dp_size(self._cfg(4, strategy="fsdp")) == 4


# The 5+3 arm: 5 engines, mini-batch 33 groups, first mini-batch 18 (requires_mini_batches=0.5),
# full cap 5 x 33 = 165, staleness quota 33 x (32 + 1) = 1089.
ENGINES, FIRST, MINI, FULL, HARD = 5, 18, 33, 165, 1089
RAMP = [4, 8, 16]


class TestParseConcurrencyRamp:
    @pytest.mark.parametrize("off", [None, "", "null", "None", "[]", []])
    def test_off_forms(self, off):
        assert parse_concurrency_ramp(off) == []

    @pytest.mark.parametrize(
        "value",
        [[4, 8, 16], (4, 8, 16), "[4,8,16]", "[4, 8, 16]", "4,8,16", OmegaConf.create([4, 8, 16]), [4.0, 8.0, 16.0]],
    )
    def test_list_forms(self, value):
        assert parse_concurrency_ramp(value) == [4, 8, 16]

    def test_single_stage_and_plateaus_are_fine(self):
        assert parse_concurrency_ramp("[4]") == [4]
        assert parse_concurrency_ramp([4, 4, 8]) == [4, 4, 8]

    @pytest.mark.parametrize("bad", [[0, 4], [-1], [4, 2], "[4,x]", [2.5], 7])
    def test_rejects_non_positive_decreasing_or_non_int(self, bad):
        with pytest.raises(ValueError, match="concurrency_ramp"):
            parse_concurrency_ramp(bad)


class TestConcurrencyCap:
    @pytest.mark.parametrize(
        "delivered, expected",
        [(0, 20), (17, 20), (18, 40), (50, 40), (51, 80), (83, 80), (84, 165), (10_000, 165)],
    )
    def test_stage_table_for_the_5plus3_arm(self, delivered, expected):
        """Stage 0 until the 18-group first mini-batch is delivered, then one full mini-batch (33) per stage."""
        assert concurrency_cap(delivered, RAMP, ENGINES, FIRST, MINI, FULL, HARD) == expected

    def test_thresholds_follow_the_full_first_minibatch_when_rmb_is_at_least_one(self):
        first = first_minibatch_groups(1.0, MINI, 16, 3) or MINI
        assert first == MINI
        caps = [concurrency_cap(d, RAMP, ENGINES, first, MINI, FULL, HARD) for d in (0, 32, 33, 65, 66, 98, 99)]
        assert caps == [20, 20, 40, 40, 80, 80, 165]

    @pytest.mark.parametrize("delivered", [0, 18, 84, 10_000])
    def test_ramp_off_is_the_full_cap(self, delivered):
        assert concurrency_cap(delivered, [], ENGINES, FIRST, MINI, FULL, HARD) == FULL

    def test_hard_cap_bounds_every_stage(self):
        assert concurrency_cap(0, RAMP, ENGINES, FIRST, MINI, FULL, hard_cap=15) == 15
        assert concurrency_cap(18, RAMP, ENGINES, FIRST, MINI, FULL, hard_cap=30) == 30
        assert concurrency_cap(84, RAMP, ENGINES, FIRST, MINI, FULL, hard_cap=30) == 30
        assert concurrency_cap(0, RAMP, ENGINES, FIRST, MINI, FULL, hard_cap=None) == 20

    def test_single_stage_ramp(self):
        assert concurrency_cap(0, [4], ENGINES, FIRST, MINI, FULL, HARD) == 20
        assert concurrency_cap(18, [4], ENGINES, FIRST, MINI, FULL, HARD) == FULL

    def test_cap_is_monotone_in_deliveries_and_never_below_one(self):
        prev = 0
        for d in range(0, 300):
            cap = concurrency_cap(d, RAMP, ENGINES, FIRST, MINI, FULL, HARD)
            assert cap >= max(1, prev)
            prev = cap
        assert concurrency_cap(0, [1], 1, 1, 1, 1, hard_cap=1) == 1

    def test_first_stage_covers_the_first_minibatch_in_one_wave(self):
        """Design rule for choosing ramp[0]: ramp[0] x n_engines >= first mini-batch, else a second wave."""
        assert RAMP[0] * ENGINES >= FIRST
