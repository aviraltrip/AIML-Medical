"""Load chunks.jsonl + embeddings.npz into memory once at app startup."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

KB_DIR = Path("data/kb")


@dataclass(frozen=True)
class KB:
    chunks: list[dict]
    embeddings: np.ndarray


@lru_cache(maxsize=1)
def load_kb() -> KB:
    chunks_path = KB_DIR / "chunks.jsonl"
    emb_path = KB_DIR / "embeddings.npz"
    if not chunks_path.exists() or not emb_path.exists():
        return KB(chunks=[], embeddings=np.zeros((0, 0), dtype=np.float32))
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    embs = np.load(emb_path)["embeddings"].astype(np.float32)
    return KB(chunks=chunks, embeddings=embs)
