from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pulsepoint_ai.core.schemas.common import SeverityTier


class CareLocatorRequest(BaseModel):
    """Inputs come straight from the upstream pipeline — ICD-10 codes from
    /predict/disease, severity tier from /triage/assess, and the patient's
    geo-coordinates from the frontend."""

    model_config = ConfigDict(extra="forbid")

    icd10_codes: list[str] = Field(
        ..., min_length=1, description="ICD-10 codes returned by the disease classifier."
    )
    severity_tier: SeverityTier = Field(
        ..., description="Severity returned by the triage engine. Drives urgency-aware sort."
    )
    patient_lat: float = Field(..., ge=-90, le=90)
    patient_lon: float = Field(..., ge=-180, le=180)
    radius_km: int | None = Field(
        None, ge=1, le=500, description="Override the default search radius from config."
    )
    max_results: int | None = Field(
        None, ge=1, le=50, description="Override the per-tier max_results from config."
    )


class DoctorMatch(BaseModel):
    id: str
    name: str
    specialty: str
    clinic: str
    phone: str
    lat: float
    lon: float
    available_today: bool
    rating: float = Field(..., ge=0, le=5)
    languages: list[str] = Field(default_factory=list)
    fee_inr: int | None = None

    distance_km: float = Field(..., ge=0)
    score: float = Field(..., ge=0, le=1)
    relevance_tier: int = Field(
        ...,
        ge=0,
        le=2,
        description="2 = exact specialty match, 1 = generalist fallback, 0 = unrelated specialty.",
    )
    score_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Per-feature contributions (specialty_match, proximity, availability, rating).",
    )


class CareLocatorResponse(BaseModel):
    request_id: str
    required_specialties: list[str]
    severity_tier: SeverityTier
    sort_policy: str = Field(
        ..., description="Either 'distance' (urgent cases) or 'score' (routine)."
    )
    radius_km: int
    doctors: list[DoctorMatch]
    total_found: int
    total_in_catalog: int
    model_version: str
