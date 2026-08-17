# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for the math500_dapo validation scorer (math_dapo string match
with an optional Math-Verify equivalence fallback)."""

from types import SimpleNamespace

from verl.utils.reward_score import default_compute_score, math500


class TestMath500StringMatch:
    def test_exact_match_scores_via_math_dapo(self):
        result = math500.compute_score("Some steps.\nAnswer: 42", "42")
        assert result == {"score": 1.0, "acc": True, "pred": "42"}

    def test_normalized_match(self):
        # Minerva normalization strips commas from large numbers.
        result = math500.compute_score("Answer: 1,000", "1000")
        assert result["acc"] is True

    def test_wrong_answer(self):
        result = math500.compute_score("Answer: 7", "42")
        assert result["score"] == -1.0
        assert result["acc"] is False

    def test_missing_answer_line(self):
        result = math500.compute_score("no final answer given", "42")
        assert result["acc"] is False
        assert result["pred"] == "[INVALID]"

    def test_dispatch_routes_math500_dapo(self):
        result = default_compute_score("math500_dapo", "Answer: 42", "42")
        assert result["acc"] is True


class TestMath500MathVerifyFallback:
    """Exercise the fallback union logic with a stubbed Math-Verify module,
    so the tests run whether or not the math-verify package is installed."""

    def _patch(self, monkeypatch, verify_result):
        calls = []

        def fake_compute_score(model_output, ground_truth):
            calls.append((model_output, ground_truth))
            return verify_result

        monkeypatch.setattr(math500, "_HAS_MATH_VERIFY", True)
        monkeypatch.setattr(math500, "_math_verify", SimpleNamespace(compute_score=fake_compute_score), raising=False)
        return calls

    def test_equivalent_form_accepted_via_math_verify(self, monkeypatch):
        calls = self._patch(monkeypatch, 1.0)
        result = math500.compute_score("Answer: 1/2", "\\frac{1}{2}")
        assert result == {"score": 1.0, "acc": True, "pred": "1/2"}
        # The extracted answer is boxed to force LaTeX extraction.
        assert calls == [("\\boxed{1/2}", "\\frac{1}{2}")]

    def test_non_equivalent_keeps_math_dapo_result(self, monkeypatch):
        self._patch(monkeypatch, 0.0)
        result = math500.compute_score("Answer: 1/3", "\\frac{1}{2}")
        assert result["score"] == -1.0
        assert result["acc"] is False

    def test_string_match_skips_math_verify(self, monkeypatch):
        calls = self._patch(monkeypatch, 0.0)
        result = math500.compute_score("Answer: 42", "42")
        assert result["acc"] is True
        assert calls == []

    def test_no_answer_line_skips_math_verify(self, monkeypatch):
        calls = self._patch(monkeypatch, 1.0)
        result = math500.compute_score("nothing to extract", "42")
        assert result["acc"] is False
        assert calls == []

    def test_last_answer_line_wins(self, monkeypatch):
        calls = self._patch(monkeypatch, 1.0)
        math500.compute_score("Answer: 1\nmore reasoning\nAnswer: 2", "3")
        assert calls == [("\\boxed{2}", "3")]

    def test_not_installed_falls_back_to_math_dapo(self, monkeypatch):
        monkeypatch.setattr(math500, "_HAS_MATH_VERIFY", False)
        result = math500.compute_score("Answer: 1/2", "\\frac{1}{2}")
        assert result["acc"] is False
