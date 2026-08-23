"""Feature builders for the LightGBM triage classifier.

The symptom vector is a fixed-length multi-hot encoding aligned with configs/symptoms.yaml.
The vitals vector is a small dense numeric block.

Feature ORDER is locked to feature_names.json in models/triage_lgbm/ — never reorder
without retraining.
"""
from __future__ import annotations

import numpy as np

from pulsepoint_ai.core.config import get_symptoms
from pulsepoint_ai.core.schemas.common import Gender, PatientProfile, Vitals


def canonical_symptom_names() -> list[str]:
    cfg = get_symptoms()
    return [s["name"] for s in cfg["symptoms"]]


def symptom_to_index() -> dict[str, int]:
    return {name: i for i, name in enumerate(canonical_symptom_names())}


def build_symptom_vector(symptoms: list[str]) -> np.ndarray:  # type: ignore[type-arg]
    idx = symptom_to_index()
    vec = np.zeros(len(idx), dtype=np.float32)
    for s in symptoms:
        if s in idx:
            vec[idx[s]] = 1.0
    return vec


def build_vitals_vector(vitals: Vitals) -> np.ndarray:  # type: ignore[type-arg]
    """Order: [spo2, bp_sys, bp_dia, sugar, pulse, temp, resp_rate]. NaN for missing."""
    return np.array(
        [
            vitals.spo2 if vitals.spo2 is not None else np.nan,
            vitals.bp_systolic if vitals.bp_systolic is not None else np.nan,
            vitals.bp_diastolic if vitals.bp_diastolic is not None else np.nan,
            vitals.blood_sugar_mg_dl if vitals.blood_sugar_mg_dl is not None else np.nan,
            vitals.pulse_bpm if vitals.pulse_bpm is not None else np.nan,
            vitals.temp_c if vitals.temp_c is not None else np.nan,
            vitals.respiratory_rate if vitals.respiratory_rate is not None else np.nan,
        ],
        dtype=np.float32,
    )


def build_profile_vector(profile: PatientProfile) -> np.ndarray:  # type: ignore[type-arg]
    """Order: [age, gender_male, gender_female]."""
    return np.array(
        [
            float(profile.age),
            1.0 if profile.gender == Gender.MALE else 0.0,
            1.0 if profile.gender == Gender.FEMALE else 0.0,
        ],
        dtype=np.float32,
    )


def build_full_feature_vector(
    symptoms: list[str], vitals: Vitals, profile: PatientProfile
) -> np.ndarray:  # type: ignore[type-arg]
    return np.concatenate(
        [
            build_symptom_vector(symptoms),
            build_vitals_vector(vitals),
            build_profile_vector(profile),
        ]
    )


def feature_names() -> list[str]:
    sym = canonical_symptom_names()
    vit = [
        "v_spo2",
        "v_bp_systolic",
        "v_bp_diastolic",
        "v_blood_sugar",
        "v_pulse",
        "v_temp",
        "v_resp_rate",
    ]
    prof = ["p_age", "p_gender_male", "p_gender_female"]
    return sym + vit + prof
