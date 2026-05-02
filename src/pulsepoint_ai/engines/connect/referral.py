"""Referral Card generator - Compiles clinical data for doctor hand-off."""
from __future__ import annotations

from typing import Any, Dict
from pulsepoint_ai.core.schemas.common import PatientProfile, Vitals

def generate_referral_data(
    patient_profile: PatientProfile,
    vitals: Vitals,
    triage_result: Dict[str, Any],
    symptoms: list[str]
) -> Dict[str, Any]:
    """Compiles all AI and clinical data into a single referral structure."""
    
    return {
        "header": {
            "hospital_ready": True,
            "priority": triage_result["severity"].value.upper(),
            "timestamp": triage_result["request_id"] # Use ID as a placeholder for time
        },
        "patient": {
            "age": patient_profile.age,
            "gender": patient_profile.gender,
            "vitals": {
                "HR": vitals.heart_rate,
                "BP": f"{vitals.bp_systolic}/{vitals.bp_diastolic}",
                "Temp": f"{vitals.temperature}°C"
            }
        },
        "clinical": {
            "chief_complaint": ", ".join(symptoms),
            "triage_reasoning": triage_result.get("doctor_briefing", "No briefing provided."),
            "red_flags": triage_result.get("red_flags", []),
            "top_conditions": [c["name"] for c in triage_result.get("top_conditions", [])]
        },
        "disclaimer": "AI-generated referral summary for clinical guidance only."
    }
