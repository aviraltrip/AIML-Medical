"""End-to-end RAG ingestion. Wired by scripts/build_kb.py.

Steps:
    1. Read configs/rag/sources.yaml
    2. For each source: download (httpx), parse (pdfplumber/trafilatura), clean
    3. Chunk (rag.chunker)
    4. Embed (rag.embed.Embedder)
    5. Write data/kb/chunks.jsonl + data/kb/embeddings.npz
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import numpy as np
import pdfplumber
import trafilatura

from pulsepoint_ai.core.config import get_rag_sources
from pulsepoint_ai.engines.triage.rag.chunker import chunk_text
from pulsepoint_ai.engines.triage.rag.embed import Embedder

KB_DIR = Path("data/kb")
RAW_DIR = Path("data/raw/kb")


def _hash_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


async def _fetch(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content


def _parse_pdf(raw: bytes) -> str:
    import io

    text = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text.append(t)
    return "\n".join(text)


def _parse_html(raw: bytes) -> str:
    return trafilatura.extract(raw.decode("utf-8", errors="ignore")) or ""


async def ingest_all() -> None:
    cfg = get_rag_sources()
    chunking = cfg["chunking"]
    sources = cfg["sources"]
    KB_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []


    for src in sources:
        if src["type"] not in {"pdf", "html"}:
            continue
        print(f"Ingesting remote source: {src['id']}...")
        try:
            raw = await _fetch(src["url"])
            text = _parse_pdf(raw) if src["type"] == "pdf" else _parse_html(raw)
        except Exception as e:
            print(f"[skip] {src['id']}: {e}")
            continue
        if not text:
            print(f"[skip] {src['id']}: empty parsed text")
            continue
        for chunk in chunk_text(
            text,
            target_tokens=chunking["target_tokens"],
            overlap_tokens=chunking["overlap_tokens"],
        ):
            cid = f"{src['id']}_{_hash_id(chunk.text)}"
            all_chunks.append(
                {
                    "id": cid,
                    "text": chunk.text,
                    "source": {"id": src["id"], "title": src["title"], "url": src["url"]},
                    "tags": src.get("tags", []),
                }
            )


    if RAW_DIR.exists():
        for file_path in RAW_DIR.glob("*"):
            if file_path.suffix.lower() not in {".md", ".txt"}:
                continue

            print(f"Ingesting local file: {file_path.name}...")
            text = file_path.read_text(encoding="utf-8")
            if not text:
                continue

            sid = f"local_{file_path.stem}"
            for chunk in chunk_text(
                text,
                target_tokens=chunking["target_tokens"],
                overlap_tokens=chunking["overlap_tokens"],
            ):
                cid = f"{sid}_{_hash_id(chunk.text)}"
                all_chunks.append(
                    {
                        "id": cid,
                        "text": chunk.text,
                        "source": {"id": sid, "title": f"Local: {file_path.name}", "url": str(file_path)},
                        "tags": ["local", "clinical_knowledge"],
                    }
                )

    if not all_chunks:
        print("No chunks found to embed.")
        return

    print(f"Built {len(all_chunks)} total chunks; embedding...")

    embedder = Embedder()
    embs = await embedder.embed_many([c["text"] for c in all_chunks])

    with (KB_DIR / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c) + "\n")
    np.savez(KB_DIR / "embeddings.npz", embeddings=embs)
    print(f"Wrote {KB_DIR}/chunks.jsonl and {KB_DIR}/embeddings.npz")


def main() -> None:
    asyncio.run(ingest_all())


if __name__ == "__main__":
    main()
