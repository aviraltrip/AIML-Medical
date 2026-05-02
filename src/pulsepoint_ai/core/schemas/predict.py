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
