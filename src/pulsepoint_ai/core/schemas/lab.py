from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from pulsepoint_ai.core.schemas.common import Gender
from pulsepoint_ai.core.schemas.triage import HallucinationCheck


class LabStatus(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    UNKNOWN = "unknown"


class LabFlag(BaseModel):
    name: str
    canonical: str
    value: float
    unit: str
    range_low: float | None
    range_high: float | None
    status: LabStatus
    explanation: str | None = None


class LabAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ocr_text: str = Field(..., min_length=10)
    age: int = Field(..., ge=0, le=120)
    gender: Gender


class LabAnalyzeResponse(BaseModel):
    request_id: str
    flags: list[LabFlag]
    unflagged_count: int
    hallucination_check: HallucinationCheck
