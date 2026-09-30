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
"""MATH-500 in the DAPO answer format: the ``math500_dapo`` scorer, its routing, and the preprocessor."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from verl.utils.reward_score import default_compute_score, math500, math_dapo

REPO = Path(__file__).resolve().parents[3]


class TestScorer:
    @pytest.mark.parametrize(
        "solution, gt",
        [
            ("Let me think.\nAnswer: 2", "2"),
            ("Answer: 0.5", "\\frac{1}{2}"),
            ("Answer: 1/2", "\\frac{1}{2}"),
            ("Answer: \\dfrac{1}{2}", "\\frac{1}{2}"),
            ("Answer: x^2+2x", "2x+x^2"),
        ],
    )
    def test_correct_answers_score_plus_one(self, solution, gt):
        out = math500.compute_score(solution, gt)
        assert out["score"] == 1.0 and out["acc"] is True

    @pytest.mark.parametrize(
        "solution, gt",
        [("Answer: 3", "2"), ("no answer line at all", "2"), ("Answer: 0.6", "\\frac{1}{2}")],
    )
    def test_wrong_or_missing_answers_score_minus_one(self, solution, gt):
        out = math500.compute_score(solution, gt)
        assert out["score"] == -1.0 and out["acc"] is False

    def test_same_dict_shape_as_math_dapo(self):
        assert set(math500.compute_score("Answer: 2", "2")) == set(math_dapo.compute_score("Answer: 2", "2"))

    def test_exact_matches_skip_math_verify(self, monkeypatch):
        calls = []
        monkeypatch.setattr(math500._math_verify, "compute_score", lambda *a: calls.append(a) or 0.0)
        assert math500.compute_score("Answer: 2", "2")["acc"] is True
        assert calls == []

    def test_math_verify_sees_the_boxed_last_answer_line(self, monkeypatch):
        calls = []
        monkeypatch.setattr(math500._math_verify, "compute_score", lambda sol, gt: calls.append((sol, gt)) or 1.0)
        out = math500.compute_score("Answer: 7\nwait, recheck\nAnswer: 1/2", "\\frac{1}{2}")
        assert calls == [("\\boxed{1/2}", "\\frac{1}{2}")]
        assert out == {"score": 1.0, "acc": True, "pred": "1/2"}

    def test_only_the_last_300_chars_are_searched(self, monkeypatch):
        monkeypatch.setattr(math500._math_verify, "compute_score", lambda *a: 1.0)
        solution = "Answer: 1/2\n" + "x" * 400
        assert math500.compute_score(solution, "\\frac{1}{2}")["acc"] is False

    def test_without_math_verify_it_is_plain_math_dapo(self, monkeypatch):
        monkeypatch.setattr(math500, "_HAS_MATH_VERIFY", False)
        assert math500.compute_score("Answer: 0.5", "\\frac{1}{2}") == math_dapo.compute_score(
            "Answer: 0.5", "\\frac{1}{2}"
        )


class TestRouting:
    def test_math500_dapo_routes_to_the_math500_scorer(self):
        assert default_compute_score("math500_dapo", "Answer: 0.5", "\\frac{1}{2}")["acc"] is True

    def test_math_dapo_routing_is_unchanged(self):
        # plain math_dapo has no Math-Verify fallback, so the equivalent form still fails there
        assert default_compute_score("math_dapo", "Answer: 0.5", "\\frac{1}{2}")["acc"] is False
        assert default_compute_score("aime2025_dapo", "Answer: 2", "2")["acc"] is True


def test_preprocessor_writes_dapo_format_parquet(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        {"problem": "What is $1+1$?", "answer": "2", "solution": "...", "subject": "a", "level": 1, "unique_id": "u0"},
        {
            "problem": "Half of one?",
            "answer": "\\frac{1}{2}",
            "solution": "...",
            "subject": "b",
            "level": 2,
            "unique_id": "u1",
        },
    ]
    (raw / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out_dir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "examples" / "data_preprocess" / "math500.py"),
            "--local_dataset_path",
            str(raw),
            "--local_save_dir",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
    )
    df = pd.read_parquet(out_dir / "math500.parquet")
    assert list(df.columns) == ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    assert len(df) == 2
    assert set(df["data_source"]) == {"math500_dapo"}
    first = df.iloc[0]
    content = first["prompt"][0]["content"]
    assert first["prompt"][0]["role"] == "user"
    assert "What is $1+1$?" in content
    assert content.startswith("Solve the following math problem step by step.")
    assert content.endswith('Remember to put your answer on its own line after "Answer:".')
    assert first["reward_model"]["ground_truth"] == "2"
    assert df.iloc[1]["reward_model"]["ground_truth"] == "\\frac{1}{2}"
    assert [r["index"] for r in df["extra_info"]] == [0, 1]
