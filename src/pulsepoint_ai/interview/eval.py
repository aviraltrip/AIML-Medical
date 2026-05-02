"""Compare fine-tuned interviewer vs vanilla GPT-4 baseline.

For each held-out scenario, both models produce a next-question. Med-student rater
scores each on a 1-5 Likert across {relevance, specificity, OSCE-correctness}.
This script aggregates the ratings and prints the delta.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", type=Path, required=True, help="JSONL with one rating per row")
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.ratings.read_text().splitlines() if line.strip()]
    by_model: dict[str, list[float]] = {}
    for r in rows:
        per_model = by_model.setdefault(r["model"], [])
        per_model.append((r["relevance"] + r["specificity"] + r["osce_correctness"]) / 3)

    summary = {m: {"mean": float(np.mean(v)), "n": len(v)} for m, v in by_model.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
