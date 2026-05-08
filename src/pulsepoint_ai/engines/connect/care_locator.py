"""HyperLocal Care Matching.

End-to-end pipeline that takes the Predict/Triage engine outputs (ICD-10 codes
and severity tier) and produces a ranked list of nearby doctors. Same safety
philosophy as the rest of PulsePoint: every weight, prefix, and policy lives
in configs/care_locator.yaml — Python only orchestrates.

Stages:
    1. Multi-label specialty mapper (longest-prefix match over ICD-10 codes,
       deterministic fallback to the configured default specialty).
    2. Geo filter using the haversine formula in km.
    3. Weighted relevance scorer (specialty_match, proximity, availability,
       rating) — a learning-to-rank style linear combination.
    4. Urgency-conditioned sort: EMERGENCY/URGENT sort by raw distance and
       require same-day availability; routine cases sort by score.
"""
from __future__ import annotations

import math
import uuid
from typing import Any

from pulsepoint_ai.core.config import get_care_locator_config, get_doctors
from pulsepoint_ai.core.schemas.care import (
    CareLocatorRequest,
    CareLocatorResponse,
    DoctorMatch,
)
from pulsepoint_ai.core.schemas.common import SeverityTier

EARTH_RADIUS_KM = 6371.0


# ---------- 1. Multi-label specialty mapper ----------------------------------
def map_icd10_to_specialties(icd10_codes: list[str]) -> list[str]:
    """Multi-label mapping: each ICD-10 code resolves to its longest matching
    prefix in the config. Deduped while preserving first-seen order so the
    primary diagnosis stays first in the list."""
    cfg = get_care_locator_config()
    mappings = cfg.get("specialty_mapping", [])
    default = cfg.get("default_specialty", "General Physician")

    # Sort by prefix length DESC so longest-match wins (e.g., I21 beats I).
    sorted_map = sorted(mappings, key=lambda m: -len(m["prefix"]))

    seen: set[str] = set()
    ordered: list[str] = []
    for code in icd10_codes:
        norm = code.strip().upper()
        matched: str | None = None
        for m in sorted_map:
            if norm.startswith(m["prefix"].upper()):
                matched = m["specialty"]
                break
        spec = matched or default
        if spec not in seen:
            seen.add(spec)
            ordered.append(spec)

    return ordered or [default]


# ---------- 2. Geo distance ---------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two coordinates in kilometres."""
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


# ---------- 3. Per-feature scoring -------------------------------------------
def _relevance_tier(doctor_specialty: str, required: list[str], cfg: dict[str, Any]) -> int:
    """2 = exact match, 1 = generalist fallback, 0 = unrelated specialty.

    This tier is the primary sort key for urgency-driven distance sorts so that
    a closer-but-irrelevant specialist (e.g., orthopedist for an MI) never
    outranks a slightly farther cardiologist.
    """
    if doctor_specialty in required:
        return 2
    if doctor_specialty == cfg.get("default_specialty"):
        return 1
    return 0


def _specialty_match_score(doctor_specialty: str, required: list[str], cfg: dict[str, Any]) -> float:
    sm = cfg["specialty_match"]
    tier = _relevance_tier(doctor_specialty, required, cfg)
    if tier == 2:
        return float(sm["exact"])
    if tier == 1:
        return float(sm.get("generalist_partial", sm["partial"]))
    return float(sm["partial"])


def _proximity_score(distance_km: float, radius_km: float) -> float:
    """Continuous proximity feature in [0,1]. Linear decay to radius."""
    if radius_km <= 0:
        return 0.0
    return max(0.0, 1.0 - (distance_km / radius_km))


def _availability_score(
    available_today: bool, requires_same_day: bool, cfg: dict[str, Any]
) -> float:
    av = cfg["availability"]
    if requires_same_day:
        return float(av["same_day_match"] if available_today else av["same_day_miss"])
    return float(av["routine_match"] if available_today else av["routine_miss"])


def _rating_score(rating: float | None) -> float:
    if rating is None:
        return 0.0
    return max(0.0, min(1.0, float(rating) / 5.0))


def _score_doctor(
    doc: dict[str, Any],
    *,
    required_specialties: list[str],
    distance_km: float,
    radius_km: float,
    requires_same_day: bool,
    cfg: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    weights = cfg["scoring_weights"]
    spec = _specialty_match_score(doc["specialty"], required_specialties, cfg)
    prox = _proximity_score(distance_km, radius_km)
    avail = _availability_score(bool(doc.get("available_today", False)), requires_same_day, cfg)
    rating = _rating_score(doc.get("rating"))

    breakdown = {
        "specialty_match": round(spec, 3),
        "proximity": round(prox, 3),
        "availability": round(avail, 3),
        "rating": round(rating, 3),
    }
    score = (
        weights["specialty_match"] * spec
        + weights["proximity"] * prox
        + weights["availability"] * avail
        + weights["rating"] * rating
    )
    return float(score), breakdown


# ---------- 4. Pipeline entry-point ------------------------------------------
def find_care(req: CareLocatorRequest) -> CareLocatorResponse:
    cfg = get_care_locator_config()
    catalog = get_doctors()

    radius_km = int(req.radius_km or cfg.get("default_radius_km", 50))

    policy = cfg.get("urgency_policy", {}).get(req.severity_tier.value, {})
    sort_by = policy.get("sort_by", "score")
    requires_same_day = bool(policy.get("require_same_day", False))
    max_results = int(req.max_results or policy.get("max_results", 5))
    drop_irrelevant = bool(policy.get("drop_irrelevant_specialty", False))

    required_specialties = map_icd10_to_specialties(req.icd10_codes)

    matches: list[DoctorMatch] = []
    for doc in catalog:
        if requires_same_day and not doc.get("available_today", False):
            continue

        distance = haversine_km(req.patient_lat, req.patient_lon, doc["lat"], doc["lon"])
        if distance > radius_km:
            continue

        tier = _relevance_tier(doc["specialty"], required_specialties, cfg)
        score, breakdown = _score_doctor(
            doc,
            required_specialties=required_specialties,
            distance_km=distance,
            radius_km=float(radius_km),
            requires_same_day=requires_same_day,
            cfg=cfg,
        )

        matches.append(
            DoctorMatch(
                id=doc["id"],
                name=doc["name"],
                specialty=doc["specialty"],
                clinic=doc["clinic"],
                phone=doc["phone"],
                lat=float(doc["lat"]),
                lon=float(doc["lon"]),
                available_today=bool(doc.get("available_today", False)),
                rating=float(doc.get("rating", 0.0)),
                languages=list(doc.get("languages", [])),
                fee_inr=doc.get("fee_inr"),
                distance_km=round(distance, 2),
                score=round(score, 3),
                relevance_tier=tier,
                score_breakdown=breakdown,
            )
        )

    if drop_irrelevant and any(m.relevance_tier > 0 for m in matches):
        # Only suppress unrelated specialists when at least one qualified match exists.
        matches = [m for m in matches if m.relevance_tier > 0]

    if sort_by == "distance":
        # Specialty-first, then nearest. Prevents an unrelated specialist from
        # outranking a slightly farther cardiologist on an MI emergency.
        matches.sort(key=lambda m: (-m.relevance_tier, m.distance_km, -m.score))
    else:
        matches.sort(key=lambda m: (-m.relevance_tier, -m.score, m.distance_km))

    return CareLocatorResponse(
        request_id=str(uuid.uuid4()),
        required_specialties=required_specialties,
        severity_tier=req.severity_tier,
        sort_policy=sort_by,
        radius_km=radius_km,
        doctors=matches[:max_results],
        total_found=len(matches),
        total_in_catalog=len(catalog),
        model_version=str(cfg.get("version", "v0.1.0")),
    )


__all__ = [
    "find_care",
    "map_icd10_to_specialties",
    "haversine_km",
]
