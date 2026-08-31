import pytest
from pulsepoint_ai.core.schemas.common import Gender, PatientProfile, SeverityTier, Vitals
from pulsepoint_ai.core.schemas.triage import TriageAssessRequest
from pulsepoint_ai.engines.triage.pipeline import run_triage
from pulsepoint_ai.engines.triage.rag.retriever import Retriever
from pulsepoint_ai.llm.client import LLMClient
from pulsepoint_ai.engines.triage.classifier.infer_adherence import adherence_classifier, predict as predict_adherence
from pulsepoint_ai.engines.chronic_scoring import calculate_idrs, evaluate_chronic_risk


def test_idrs_boundary_conditions():
    # Female age 35, waist 85, moderate physical activity, no family history
    score, details = calculate_idrs(
        age=35,
        gender=Gender.FEMALE,
        waist_circumference=85,
        physical_activity="moderate",
        family_history="none",
    )
    assert score == 50  # age:20 + waist:10 + activity:20 + family:0 = 50
    assert details["age_score"] == 20
    assert details["waist_score"] == 10
    assert details["activity_score"] == 20


def test_adherence_risk_scoring_bounds():
    # High risk profile: laborer, distance in symptoms, previous non-adherence
    high_risk_profile = PatientProfile(
        age=60,
        gender=Gender.MALE,
        known_conditions=["laborer", "daily wage", "previous non-adherence"],
    )
    high_risk_symptoms = ["fever", "distance 25 km from PHC"]
    prob = predict_adherence(high_risk_profile, high_risk_symptoms, SeverityTier.URGENT)
    assert 0.0 <= prob <= 1.0

    # Low risk profile: nearby, routine
    low_risk_profile = PatientProfile(
        age=28,
        gender=Gender.FEMALE,
        known_conditions=[],
    )
    low_risk_symptoms = ["mild headache"]
    prob_low = predict_adherence(low_risk_profile, low_risk_symptoms, SeverityTier.LOW)
    assert 0.0 <= prob_low <= 1.0


def test_full_triage_pipeline_execution():
    import asyncio

    async def _test():
        req = TriageAssessRequest(
            patient_id="patient-unit-test-01",
            symptoms=["frequent urination", "excessive thirst", "fatigue"],
            vitals=Vitals(
                pulse_bpm=78,
                bp_systolic=145,
                bp_diastolic=92,
                blood_sugar_mg_dl=180.0,
                temp_c=36.8,
            ),
            patient_profile=PatientProfile(
                age=54,
                gender=Gender.MALE,
                known_conditions=["prehypertension"],
            ),
        )
        llm = LLMClient()
        retriever = Retriever()
        result = await run_triage(req, llm=llm, retriever=retriever)
        assert result.severity in {SeverityTier.HIGH, SeverityTier.URGENT, SeverityTier.EMERGENCY, SeverityTier.MEDIUM}
        assert len(result.top_conditions) > 0
        assert len(result.reasoning_steps) > 0

    asyncio.run(_test())
