"""Heading-aware chunker that targets ~target_tokens per chunk with overlap."""
from __future__ import annotations

from dataclasses import dataclass

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class Chunk:
    text: str
    heading_path: list[str]


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in ".!?\n" and len("".join(buf).strip()) > 0:
            out.append("".join(buf).strip())
            buf = []
    if buf:
        out.append("".join(buf).strip())
    return [s for s in out if s]


def chunk_text(
    text: str,
    *,
    target_tokens: int = 400,
    overlap_tokens: int = 50,
    headings: list[str] | None = None,
) -> list[Chunk]:
    sentences = _split_sentences(text)
    chunks: list[Chunk] = []
    cur: list[str] = []
    cur_tokens = 0

    for s in sentences:
        n = len(_ENC.encode(s))
        if cur_tokens + n > target_tokens and cur:
            chunks.append(Chunk(text=" ".join(cur), heading_path=headings or []))
            # overlap: pop sentences from end until below overlap_tokens, then start new
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(cur):
                pt = len(_ENC.encode(prev))
                if tail_tokens + pt > overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_tokens += pt
            cur = tail
            cur_tokens = tail_tokens
        cur.append(s)
        cur_tokens += n

    if cur:
        chunks.append(Chunk(text=" ".join(cur), heading_path=headings or []))
    return chunks
