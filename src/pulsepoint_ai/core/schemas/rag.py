from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pulsepoint_ai.core.schemas.common import Source


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    source: Source
    score: float = Field(..., ge=-1, le=1)


class RagSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=2)
    top_k: int = Field(5, ge=1, le=20)
    filter_tags: list[str] = Field(default_factory=list)


class RagSearchResponse(BaseModel):
    request_id: str
    results: list[RetrievedChunk]
