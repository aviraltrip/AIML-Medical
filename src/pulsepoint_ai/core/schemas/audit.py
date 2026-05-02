from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AuditEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    ts: datetime
    request_id: str
    user_id: str | None
    endpoint: str
    model: str
    prompt_version: str | None
    rag_chunk_ids: list[str]
    classifier_output: dict | None
    final_verdict: dict | None
    hallucinated: bool
    hallucinated_terms: list[str]
    latency_ms: int


class AuditExportRow(BaseModel):
    ts: datetime
    endpoint: str
    model: str
    prompt_version: str | None
    hallucinated: bool
    final_severity: str | None
    latency_ms: int
