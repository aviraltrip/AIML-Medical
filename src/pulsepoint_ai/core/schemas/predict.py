from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pulsepoint_ai.core.schemas.common import Gender, Source
from pulsepoint_ai.core.schemas.triage import FeatureImportance


class DiseasePrediction(BaseModel):
    icd10: str
    name: str
    prob: float = Field(..., ge=0, le=1)
    top_features: list[FeatureImportance]


class DiseasePredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symptoms: list[str] = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=120)
    gender: Gender
    top_k: int = Field(5, ge=1, le=10)


class DiseasePredictResponse(BaseModel):
    request_id: str
    predictions: list[DiseasePrediction]
    model_version: str


class ConditionCardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    icd10: str
    name: str | None = None
    language: str = Field("en", pattern=r"^[a-z]{2}$")
    reading_level: str = Field("grade6", pattern=r"^grade(4|6|8)$")


class ConditionCardResponse(BaseModel):
    request_id: str
    name: str
    plain_summary: str
    action_steps: list[str]
    sources: list[Source]
    unsupported: bool = False


class SymptomExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        description="Free-text from PDF/OCR/clinical notes to extract canonical symptoms from.",
    )


class SymptomMatchOut(BaseModel):
    canonical: str
    phrase: str
    negated: bool
    sentence: str


class SymptomExtractionResponse(BaseModel):
    request_id: str
    symptoms: list[str] = Field(
        ...,
        description="Canonical symptom names that are positively asserted in the text.",
    )
    negated: list[str] = Field(
        default_factory=list,
        description="Canonical symptoms that were detected but inside a negation scope (e.g. 'no history of X').",
    )
    matches: list[SymptomMatchOut] = Field(
        default_factory=list,
        description="Per-occurrence audit trail showing the matched phrase, its sentence, and whether it was negated.",
    )
