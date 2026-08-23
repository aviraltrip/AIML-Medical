from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SeverityTier(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"
    EMERGENCY = "EMERGENCY"


class Gender(StrEnum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class Vitals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spo2: float | None = Field(None, ge=0, le=100, description="Oxygen saturation %")
    bp_systolic: int | None = Field(None, ge=40, le=300)
    bp_diastolic: int | None = Field(None, ge=20, le=200)
    blood_sugar_mg_dl: float | None = Field(None, ge=10, le=900)
    pulse_bpm: int | None = Field(None, ge=20, le=300)
    temp_c: float | None = Field(None, ge=25, le=45)
    respiratory_rate: int | None = Field(None, ge=4, le=80)


class PatientProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=0, le=120)
    gender: Gender
    known_conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    url: str
