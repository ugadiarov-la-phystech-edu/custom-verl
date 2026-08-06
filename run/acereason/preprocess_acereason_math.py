#!/usr/bin/env python
"""Preprocess nvidia/AceReason-Math into a verl parquet for the math_dapo scorer.

Keeps only problems whose ground-truth answer normalizes to a plain
integer/decimal under math_dapo's normalize_final_answer -- the subset the
scorer's normalized string match handles robustly (~42.3k of 49.6k rows).
Output schema mirrors /data2/datasets/dapo/dapo-math-17k.parquet exactly.

Run from the verl repo root (needs `verl` importable and `datasets` installed):
    python run/acereason/preprocess_acereason_math.py --local_save_dir /data2/datasets/acereason
"""

import argparse
import os
import re
import sys
import uuid

import datasets
import pandas as pd

# make `verl` importable: walk up from the script (and cwd) to the repo root
_candidates = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
_candidates += [os.path.abspath(os.path.join(_candidates[1], *[".."] * n)) for n in (1, 2)]
for _root in _candidates:
    if os.path.isfile(os.path.join(_root, "verl", "__init__.py")):
        sys.path.insert(0, _root)
        break

from verl.utils.reward_score.math_dapo import normalize_final_answer

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem.\n\n{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)

PLAIN_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="/data2/datasets/acereason")
    parser.add_argument("--hf_dataset", default="nvidia/AceReason-Math")
    args = parser.parse_args()

    ds = datasets.load_dataset(args.hf_dataset, split="train")
    print(f"loaded {len(ds)} rows from {args.hf_dataset}")

    # the raw dataset contains exact duplicate rows and a few problems with
    # conflicting answers -- drop both (dedup keeps first occurrence)
    answers_per_problem = {}
    for example in ds:
        answers_per_problem.setdefault(example["problem"], set()).add(example["answer"])
    conflicting = {p for p, a in answers_per_problem.items() if len(a) > 1}

    rows = []
    seen = set()
    dropped = dup_dropped = conflict_dropped = 0
    for example in ds:
        problem = example["problem"]
        answer = example["answer"]
        if problem in conflicting:
            conflict_dropped += 1
            continue
        if problem in seen:
            dup_dropped += 1
            continue
        seen.add(problem)
        if not PLAIN_NUMBER_RE.match(normalize_final_answer(answer)):
            dropped += 1
            continue
        rows.append(
            {
                "source_prompt": problem,
                "solution": answer,
                "data_source": "math_dapo",
                "prompt": [{"content": PROMPT_TEMPLATE.format(problem=problem), "role": "user"}],
                "ability": "MATH",
                "reward_model": {"ground_truth": answer, "style": "rule-lighteval/MATH_v2"},
                # deterministic uuid so re-runs produce identical parquets
                "extra_info": {"index": str(uuid.uuid5(uuid.NAMESPACE_OID, problem))},
            }
        )

    print(
        f"kept {len(rows)} rows | dropped: {dropped} non-robust answers, "
        f"{dup_dropped} exact duplicates, {conflict_dropped} conflicting-answer rows"
    )

    df = pd.DataFrame(rows)
    os.makedirs(args.local_save_dir, exist_ok=True)
    out = os.path.join(args.local_save_dir, "acereason-math-filtered.parquet")
    df.to_parquet(out)
    print("saved", out)


if __name__ == "__main__":
    main()
