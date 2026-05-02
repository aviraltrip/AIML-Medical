"""Build the OSCE-style Q&A dataset for LoRA fine-tuning.

Pipeline:
    1. Load 50 seed scenarios from data/interview_qa/seeds.jsonl (med-student-authored).
    2. Use Gemini to expand each seed into ~8 paraphrased Q/A variants.
    3. Save to data/interview_qa/raw_synthetic.jsonl for clinical review.
    4. After review, split into train/val/test (80/10/10) by chief complaint.

CLI:
    uv run python -m pulsepoint_ai.interview.dataset_builder \
        --seeds data/interview_qa/seeds.jsonl \
        --out   data/interview_qa/raw_synthetic.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_seeds(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def expand_seed(seed: dict, n_variants: int = 8) -> list[dict]:
    """Stub. Real implementation calls LLMClient with a paraphrase + diversify prompt."""
    raise NotImplementedError(
        "Wire to pulsepoint_ai.llm.client.LLMClient when GOOGLE_API_KEY is set."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", type=int, default=8)
    args = ap.parse_args()

    seeds = load_seeds(args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for seed in seeds:
            for v in expand_seed(seed, args.variants):
                fh.write(json.dumps(v) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
