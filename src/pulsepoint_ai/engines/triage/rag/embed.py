"""Gemini Text Embedding wrapper (text-embedding-004)."""
from __future__ import annotations

import google.generativeai as genai
import numpy as np

from pulsepoint_ai.core.config import get_models_config, get_settings


class Embedder:
    def __init__(self) -> None:
        cfg = get_models_config()["embedding"]
        self.model = cfg["model"]
        self.dim = cfg["dim"]


        genai.configure(api_key=get_settings().google_api_key)

    async def embed_one(self, text: str) -> np.ndarray:
        """Turns a single string into a vector using Gemini."""
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            return np.array(result['embedding'], dtype=np.float32)
        except Exception as e:
            print(f"Gemini Embedding error: {e}")
            return np.zeros(self.dim, dtype=np.float32)

    async def embed_many(self, texts: list[str]) -> np.ndarray:
        """Batch processes multiple strings into a 2D numpy array."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        try:

            result = genai.embed_content(
                model=self.model,
                content=texts,
                task_type="retrieval_document"
            )
            return np.array(result['embedding'], dtype=np.float32)
        except Exception as e:
            print(f"Gemini Batch embedding error: {e}")
            return np.zeros((len(texts), self.dim), dtype=np.float32)
