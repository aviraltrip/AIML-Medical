from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class TrendLabel(str, Enum):
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    WORSENING = "WORSENING"


class TrendPoint(BaseModel):
    date: date
    value: float


class TrendAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str
    metric: str
    series: list[TrendPoint] = Field(..., min_length=3)


class TrendAnalyzeResponse(BaseModel):
    request_id: str
    metric: str
    slope: float
    trend: TrendLabel
    predicted_3mo: float
    alert_caregiver: bool
    plain_text: str


class HealthRiskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str


class HealthRiskBreakdown(BaseModel):
    lab_trends: float
    triage_history: float
    vitals_pattern: float
    medication_adherence: float
    condition_flags: float


class HealthRiskResponse(BaseModel):
    request_id: str
    score: int = Field(..., ge=0, le=100)
    breakdown: HealthRiskBreakdown
    plain_text: str


class WeeklyDigestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str
    week_ending: date


class WeeklyDigestResponse(BaseModel):
    request_id: str
    summary: str
    highlights: list[str]
    questions_for_doctor: list[str]
