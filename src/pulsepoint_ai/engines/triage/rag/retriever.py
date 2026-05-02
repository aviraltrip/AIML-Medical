"""In-memory cosine retriever. Suitable for KB sizes up to ~50MB."""
from __future__ import annotations

import numpy as np

from pulsepoint_ai.core.schemas.common import Source
from pulsepoint_ai.core.schemas.rag import RetrievedChunk
from pulsepoint_ai.engines.triage.rag.embed import Embedder
from pulsepoint_ai.engines.triage.rag.kb_loader import KB, load_kb


class Retriever:
    def __init__(self, embedder: Embedder | None = None, kb: KB | None = None) -> None:
        self._kb = kb or load_kb()
        self._embedder = embedder or Embedder()
        if self._kb.embeddings.size > 0:
            norms = np.linalg.norm(self._kb.embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._normalized = self._kb.embeddings / norms
        else:
            self._normalized = self._kb.embeddings

    async def search(
        self, query: str, *, top_k: int = 5, filter_tags: list[str] | None = None
    ) -> list[RetrievedChunk]:
        if self._kb.embeddings.size == 0:
            return []
        q_vec = await self._embedder.embed_one(query)
        q_norm = q_vec / (np.linalg.norm(q_vec) or 1.0)
        scores = self._normalized @ q_norm

        order = np.argsort(scores)[::-1]
        results: list[RetrievedChunk] = []
        for i in order:
            chunk = self._kb.chunks[int(i)]
            if filter_tags and not set(filter_tags) & set(chunk.get("tags", [])):
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk["id"],
                    text=chunk["text"],
                    source=Source(**chunk["source"]),
                    score=float(scores[int(i)]),
                )
            )
            if len(results) >= top_k:
                break
        return results
