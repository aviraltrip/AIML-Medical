from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from pulsepoint_ai.core.schemas.common import PatientProfile


class AnswerType(StrEnum):
    YES_NO = "yes_no"
    SCALE = "scale"
    FREE_TEXT = "free_text"


class AnsweredQuestion(BaseModel):
    question: str
    answer: str


class InterviewerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symptoms: list[str] = Field(..., min_length=1)
    answered: list[AnsweredQuestion] = Field(default_factory=list)
    patient_profile: PatientProfile
    relay_mode: bool = Field(
        False,
        description="If true, phrase the question for a caregiver to relay over phone.",
    )


class InterviewerResponse(BaseModel):
    request_id: str
    question: str
    rationale: str
    expected_answer_type: AnswerType
    model_version: str
