from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pulsepoint_ai.core.schemas.common import (
    PatientProfile,
    SeverityTier,
    Source,
    Vitals,
)


class FeatureImportance(BaseModel):
    feature: str
    shap: float


class TopCondition(BaseModel):
    name: str
    icd10: str
    prob: float = Field(..., ge=0, le=1)


class HallucinationCheck(BaseModel):
    blocked: list[str] = Field(default_factory=list)
    passed: bool = True


class TriageAssessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str
    symptoms: list[str] = Field(..., min_length=1)
    vitals: Vitals = Field(default_factory=Vitals)
    patient_profile: PatientProfile
    language: str = Field("en", pattern=r"^[a-z]{2}$")


class TriageAssessResponse(BaseModel):
    request_id: str
    severity: SeverityTier
    tier_probabilities: dict[str, float]
    rules_fired: list[str]
    top_conditions: list[TopCondition]
    reasoning_steps: list[str]
    red_flags: list[str]
    doctor_briefing: str
    feature_importance: list[FeatureImportance]
    sources: list[Source]
    confidence: float = Field(..., ge=0, le=1)
    hallucination_check: HallucinationCheck
    next_action: str
    model_versions: dict[str, str]
