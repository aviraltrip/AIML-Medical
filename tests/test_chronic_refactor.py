from fastapi.testclient import TestClient

from pulsepoint_ai.core.schemas.common import Gender, PatientProfile, Vitals, SeverityTier
from pulsepoint_ai.engines.chronic_scoring import (
    calculate_idrs,
    calculate_hypertension_risk,
    evaluate_chronic_risk,
    infer_physical_activity,
    infer_family_history,
    infer_waist_circumference,
)
from pulsepoint_ai.engines.predict.lab_detector import detect_labs
from pulsepoint_ai.core.schemas.lab import LabStatus
from pulsepoint_ai.api.main import app

client = TestClient(app)


def test_idrs_calculations():
    # 1. Standard Case: age=52, waist=102 (male), active=moderate, family history=one_parent
    # Age >= 50 -> 30 pts
    # Male Waist >= 100 -> 20 pts
    # Moderate activity -> 20 pts
    # One parent -> 10 pts
    # Total = 30 + 20 + 20 + 10 = 80
    score, details = calculate_idrs(
        age=52,
        gender=Gender.MALE,
        waist_circumference=102,
        physical_activity="moderate",
        family_history="one_parent"
    )
    assert score == 80
    assert details["age_score"] == 30
    assert details["waist_score"] == 20
    assert details["activity_score"] == 20
    assert details["family_score"] == 10

    # 2. Young female, active, no family history
    # Age 25 -> 0 pts
    # Female Waist 78 -> 0 pts
    # Vigorous activity -> 0 pts
    # Family history none -> 0 pts
    # Total = 0
    score2, details2 = calculate_idrs(
        age=25,
        gender=Gender.FEMALE,
        waist_circumference=78,
        physical_activity="vigorous",
        family_history="none"
    )
    assert score2 == 0


def test_parameter_inferences():
    # Test sedentary activity inference
    symptoms = ["frequent urination", "sedentary lifestyle"]
    known_conditions = ["obesity"]
    assert infer_physical_activity(symptoms, known_conditions) == "sedentary"

    # Test family history inference
    symptoms2 = ["headache", "mother diabetic"]
    assert infer_family_history(symptoms2, []) == "one_parent"

    # Test waist circumference inference
    symptoms3 = ["frequent urination", "waist: 96 cm"]
    assert infer_waist_circumference(symptoms3, Gender.MALE) == 96.0

    # Test obesity fallback for waist circumference
    symptoms4 = ["overweight", "polyuria"]
    assert infer_waist_circumference(symptoms4, Gender.FEMALE) == 95.0


def test_hypertension_staging():
    # Test Normal BP
    vitals = Vitals(bp_systolic=115, bp_diastolic=75)
    profile = PatientProfile(age=30, gender=Gender.MALE)
    prob, staging, details = calculate_hypertension_risk(vitals, profile, [])
    assert staging == "Normal"
    assert prob == 0.05

    # Test Stage 1 Hypertension
    vitals = Vitals(bp_systolic=145, bp_diastolic=95)
    prob, staging, details = calculate_hypertension_risk(vitals, profile, [])
    assert staging == "Stage 1 Hypertension"
    assert prob == 0.65

    # Test Hypertensive Crisis
    vitals = Vitals(bp_systolic=185, bp_diastolic=125)
    prob, staging, details = calculate_hypertension_risk(vitals, profile, [])
    assert staging == "Hypertensive Crisis"
    assert prob == 0.98

    # Test Missing vitals fallback
    vitals_missing = Vitals()
    symptoms = ["headache", "dizziness", "tobacco use"]
    prob, staging, details = calculate_hypertension_risk(vitals_missing, profile, symptoms)
    assert "Suspected" in staging
    assert not details["vitals_present"]


def test_chronic_risk_evaluation():
    profile = PatientProfile(age=55, gender=Gender.MALE)
    vitals = Vitals(bp_systolic=165, bp_diastolic=105, blood_sugar_mg_dl=135)
    symptoms = ["fatigue", "polyuria", "family history of diabetes"]

    results = evaluate_chronic_risk(profile, vitals, symptoms)
    assert results["tier"] == SeverityTier.URGENT  # Stage 2 HTN and FBS >= 126 triggers URGENT
    assert results["diabetes_idrs"] >= 50
    assert results["diabetes_prob"] >= 0.70
    assert results["hypertension_prob"] >= 0.80
    assert "bp_stage2_hypertension" in results["rules_fired"]
    assert "glucose_diabetic_range" in results["rules_fired"]


def test_fuzzy_ocr_lab_parser():
    # Test OCR parsing with spaces, noise, and different cases
    ocr_text = """
    Patient Report
    Fasting Blood Sugar: 128 mg/dL
    HB-A1C = 6.8 %
    microalbuminuria: 35 mg/L
    Random Glucose level is 145 mg/dL
    Total Cholesterol : 185
    Urine Protein - 25 mg/dL
    """
    
    flags, normal_count = detect_labs(ocr_text, 45, Gender.FEMALE)
    
    flagged_canonicals = {f.canonical for f in flags}
    assert "fasting_glucose" in flagged_canonicals
    assert "hba1c" in flagged_canonicals
    assert "microalbumin" in flagged_canonicals
    assert "random_glucose" in flagged_canonicals
    assert "total_cholesterol" in flagged_canonicals
    assert "urine_protein" in flagged_canonicals

    # Check status classification thresholds
    for f in flags:
        if f.canonical == "hba1c":
            assert f.value == 6.8
            assert f.status == LabStatus.HIGH  # HbA1c >= 6.5
        elif f.canonical == "fasting_glucose":
            assert f.value == 128.0
            assert f.status == LabStatus.HIGH  # FBS >= 126
        elif f.canonical == "microalbumin":
            assert f.value == 35.0
            assert f.status == LabStatus.HIGH  # microalbumin >= 30
        elif f.canonical == "urine_protein":
            assert f.value == 25.0
            assert f.status == LabStatus.HIGH  # urine protein >= 20


def test_adherence_classifier():
    from pulsepoint_ai.engines.triage.classifier.infer_adherence import AdherenceClassifier
    from pulsepoint_ai.core.schemas.common import SeverityTier
    
    clf = AdherenceClassifier()
    profile = PatientProfile(age=45, gender=Gender.MALE)
    
    # Test default parsing
    feats = clf.parse_rural_features(profile, ["fatigue"], SeverityTier.MEDIUM)
    assert feats["distance_to_phc"] == 5.0
    assert feats["occupation_farming"] == 0.0
    assert feats["occupation_laborer"] == 0.0
    assert feats["tobacco_alcohol_usage"] == 0.0
    
    # Test parsing context clues
    symptoms = ["fatigue", "12 km from clinic", "i am a farmer", "tobacco bidi chewing"]
    feats2 = clf.parse_rural_features(profile, symptoms, SeverityTier.URGENT)
    assert feats2["distance_to_phc"] == 12.0
    assert feats2["occupation_farming"] == 1.0
    assert feats2["tobacco_alcohol_usage"] == 1.0
    assert feats2["severity_tier_val"] == 3.0
    
    # Test predict output
    prob = clf.predict_default_risk(profile, symptoms, SeverityTier.URGENT)
    assert 0.0 <= prob <= 1.0


from unittest.mock import AsyncMock, patch
import numpy as np

def test_api_endpoints_contract():
    async def mock_complete_json(prompt, prompt_version="v1"):
        if "reasoner_v1" in prompt_version or "reasoner" in prompt:
            return {
                "severity": "URGENT",
                "top_conditions": [
                    {"name": "Type 2 Diabetes Risk", "icd10": "E11", "prob": 0.85},
                    {"name": "Hypertension Risk", "icd10": "I10", "prob": 0.90}
                ],
                "reasoning_steps": ["Deterministic scoring indicates high chronic risk.", "Vitals rules checked."],
                "red_flags": ["Elevated blood glucose (135 mg/dL)", "Stage 2 Hypertension (165/105 mmHg)"],
                "doctor_briefing": "Patient screened HIGH RISK for Type 2 Diabetes and Stage 2 Hypertension. IDRS score estimated at 80/100 driven by age, obesity, and family history. Blood pressure elevated at 165/105. Recommend PHC referral for fasting glucose and HbA1c confirmation.",
                "sources": [{"id": "who_guidelines", "title": "WHO Screening Guidelines", "url": "https://who.int"}],
                "confidence": 0.90
            }
        elif "lab_explainer_v1" in prompt_version:
            return {
                "explanations": [
                    {"id": "fasting_glucose", "text": "Fasting blood glucose is elevated, suggesting a risk of prediabetes or diabetes."},
                    {"id": "hba1c", "text": "HbA1c level is in the diabetic range, indicating high risk of Type 2 Diabetes."}
                ]
            }
        else: # interviewer
            return {
                "question": "Is there a history of sugar (diabetes) or high BP in your parents?",
                "rationale": "Determines genetic predisposition for metabolic screening.",
                "expected_answer_type": "yes_no"
            }

    async def mock_embed_one(text):
        return np.zeros(3072, dtype=np.float32)

    with patch("pulsepoint_ai.llm.client.LLMClient.complete_json", new_callable=AsyncMock) as mock_complete, \
         patch("pulsepoint_ai.engines.triage.rag.embed.Embedder.embed_one", new_callable=AsyncMock) as mock_embed:
        
        mock_complete.side_effect = mock_complete_json
        mock_embed.side_effect = mock_embed_one

        # 1. Test POST /api/v1/triage/assess
        assess_payload = {
            "patient_id": "test-patient-123",
            "symptoms": ["polyuria", "fatigue", "excessive_thirst", "family history of diabetes"],
            "vitals": {
                "bp_systolic": 145,
                "bp_diastolic": 92,
                "blood_sugar_mg_dl": 115
            },
            "patient_profile": {
                "age": 48,
                "gender": "male",
                "known_conditions": ["overweight"],
                "medications": [],
                "allergies": []
            },
            "language": "en"
        }
        
        response = client.post("/api/v1/triage/assess", json=assess_payload)
        if response.status_code != 200:
            print(f"Assess Endpoint failed: status={response.status_code}, body={response.text}")
        assert response.status_code == 200
        data = response.json()
        
        # Contract validation
        assert "request_id" in data
        assert "severity" in data
        assert "tier_probabilities" in data
        assert "top_conditions" in data
        assert "reasoning_steps" in data
        assert "doctor_briefing" in data
        assert "feature_importance" in data
        assert "next_action" in data
        
        assert data["severity"] in ["LOW", "MEDIUM", "HIGH", "URGENT", "EMERGENCY"]
        assert any(c["icd10"] == "E11" for c in data["top_conditions"])
        assert any(c["icd10"] == "I10" for c in data["top_conditions"])

        # 2. Test POST /api/v1/triage/interview
        interview_payload = {
            "symptoms": ["fatigue", "vision_blurred"],
            "answered": [
                {"question": "Do you have a family history of diabetes?", "answer": "yes"}
            ],
            "patient_profile": {
                "age": 52,
                "gender": "female",
                "known_conditions": [],
                "medications": [],
                "allergies": []
            },
            "relay_mode": False
        }
        
        response = client.post("/api/v1/triage/interview", json=interview_payload)
        if response.status_code != 200:
            print(f"Interview Endpoint failed: status={response.status_code}, body={response.text}")
        assert response.status_code == 200
        data = response.json()
        
        assert "request_id" in data
        assert "question" in data
        assert "rationale" in data
        assert "expected_answer_type" in data

        # 3. Test POST /api/v1/predict/disease
        disease_payload = {
            "symptoms": ["polyuria", "excessive_thirst", "headache"],
            "age": 50,
            "gender": "male",
            "top_k": 3
        }
        
        response = client.post("/api/v1/predict/disease", json=disease_payload)
        if response.status_code != 200:
            print(f"Disease Endpoint failed: status={response.status_code}, body={response.text}")
        assert response.status_code == 200
        data = response.json()
        
        assert "request_id" in data
        assert "predictions" in data
        assert len(data["predictions"]) > 0
        assert data["predictions"][0]["icd10"] in ["E11", "I10", "E66.9"]

        # 4. Test POST /api/v1/predict/labs
        labs_payload = {
            "ocr_text": "Fasting Blood Sugar 142 mg/dL. HbA1c is 7.2%. Creatinine: 1.2 mg/dL.",
            "age": 60,
            "gender": "male"
        }
        
        response = client.post("/api/v1/predict/labs", json=labs_payload)
        if response.status_code != 200:
            print(f"Labs Endpoint failed: status={response.status_code}, body={response.text}")
        assert response.status_code == 200
        data = response.json()
        
        assert "request_id" in data
        assert "flags" in data
        assert "unflagged_count" in data
        assert "hallucination_check" in data
