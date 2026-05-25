"""Deterministic chronic disease risk scoring engine.

Calculates explainable risk scores for Type 2 Diabetes (using IDRS) and
Hypertension (using JNC-8/WHO thresholds). Includes safe confidence-based fallbacks.
"""
from __future__ import annotations

import re
from typing import Any
from pulsepoint_ai.core.schemas.common import SeverityTier, Vitals, Gender, PatientProfile

def infer_physical_activity(symptoms: list[str], known_conditions: list[str]) -> str:
    """Infers physical activity level from keywords in symptoms or known conditions."""
    all_text = " ".join(symptoms + known_conditions).lower()
    if any(x in all_text for x in ["sedentary", "no exercise", "inactive", "desk job", "no physical activity", "lazy"]):
        return "sedentary"
    if any(x in all_text for x in ["vigorous exercise", "strenuous", "manual labor", "active", "athlete", "heavy work"]):
        return "vigorous"
    return "moderate"  # Safe default for rural screening

def infer_family_history(symptoms: list[str], known_conditions: list[str]) -> str:
    """Infers family history of diabetes/hypertension from symptoms or known conditions."""
    all_text = " ".join(symptoms + known_conditions).lower()
    if any(x in all_text for x in ["both parents diabetic", "both parents have diabetes", "both parents with diabetes"]):
        return "both_parents"
    if any(x in all_text for x in [
        "family history of diabetes", "mother diabetic", "father diabetic",
        "parent diabetic", "one parent diabetic", "family_diabetes",
        "family history of hypertension", "mother hypertensive", "father hypertensive"
    ]):
        return "one_parent"
    return "none"

def infer_waist_circumference(symptoms: list[str], gender: Gender) -> float | None:
    """Extracts waist circumference from text or infers it based on obesity/overweight indicators."""
    all_text = " ".join(symptoms).lower()
    
    # Try pattern matching waist: 95, waist 95 cm, waist_95, etc.
    match = re.search(r"\bwaist(?:_circumference)?[\s:=]*(\d+)", all_text)
    if match:
        return float(match.group(1))
        
    # Check for obesity/overweight clues
    if any(x in all_text for x in ["obesity", "obese", "overweight", "high bmi"]):
        return 105.0 if gender == Gender.MALE else 95.0
        
    return None

def calculate_idrs(
    age: int,
    gender: Gender,
    waist_circumference: float | None,
    physical_activity: str,
    family_history: str
) -> tuple[int, dict[str, Any]]:
    """Calculates the Indian Diabetes Risk Score (IDRS) and returns score and breakdown.
    
    Maximum score is 100.
    - Low Risk: < 30
    - Moderate Risk: 30 - 50
    - High Risk: >= 60
    """
    breakdown = {}
    
    # 1. Age Score
    if age < 35:
        age_score = 0
    elif age < 50:
        age_score = 20
    else:
        age_score = 30
    breakdown["age_score"] = age_score

    # 2. Waist Circumference Score
    # Male: <90cm (0), 90-99cm (10), >=100cm (20)
    # Female: <80cm (0), 80-89cm (10), >=90cm (20)
    if waist_circumference is None:
        # Default fallback to moderate if missing
        waist_score = 10
        breakdown["waist_inferred"] = True
    else:
        breakdown["waist_inferred"] = False
        if gender == Gender.MALE:
            if waist_circumference < 90:
                waist_score = 0
            elif waist_circumference < 100:
                waist_score = 10
            else:
                waist_score = 20
        else: # Female or other
            if waist_circumference < 80:
                waist_score = 0
            elif waist_circumference < 90:
                waist_score = 10
            else:
                waist_score = 20
    breakdown["waist_score"] = waist_score

    # 3. Physical Activity Score
    # Vigorous (0), Moderate (20), Sedentary/None (30)
    if physical_activity == "vigorous":
        activity_score = 0
    elif physical_activity == "sedentary":
        activity_score = 30
    else:
        activity_score = 20
    breakdown["activity_score"] = activity_score

    # 4. Family History Score
    # None (0), One parent (10), Both parents (20)
    if family_history == "both_parents":
        family_score = 20
    elif family_history == "one_parent":
        family_score = 10
    else:
        family_score = 0
    breakdown["family_score"] = family_score

    total_score = age_score + waist_score + activity_score + family_score
    return total_score, breakdown

def calculate_hypertension_risk(
    vitals: Vitals,
    profile: PatientProfile,
    symptoms: list[str]
) -> tuple[float, str, dict[str, Any]]:
    """Calculates Hypertension Risk and returns (probability, staging, details)."""
    sbp = vitals.bp_systolic
    dbp = vitals.bp_diastolic
    
    details = {}
    
    # 1. Deterministic Staging if Vitals are Present
    if sbp is not None and dbp is not None:
        details["vitals_present"] = True
        if sbp >= 180 or dbp >= 120:
            staging = "Hypertensive Crisis"
            prob = 0.98
        elif sbp >= 160 or dbp >= 100:
            staging = "Stage 2 Hypertension"
            prob = 0.85
        elif sbp >= 140 or dbp >= 90:
            staging = "Stage 1 Hypertension"
            prob = 0.65
        elif sbp >= 120 or dbp >= 80:
            staging = "Prehypertension"
            prob = 0.25
        else:
            staging = "Normal"
            prob = 0.05
    else:
        # 2. Cautious Fallback if Vitals are Missing
        details["vitals_present"] = False
        all_text = " ".join(symptoms + profile.known_conditions).lower()
        
        # Risk factors count
        risk_score = 0
        if profile.age >= 50:
            risk_score += 1
        if any(x in all_text for x in ["headache", "dizziness", "stress", "tension"]):
            risk_score += 2
        if any(x in all_text for x in ["tobacco", "smoking", "bidi", "khaini", "alcohol", "daroo"]):
            risk_score += 1.5
        if any(x in all_text for x in ["sedentary", "no exercise", "salt", "high salt"]):
            risk_score += 1.5
            
        if risk_score >= 4:
            staging = "Suspected High Risk"
            prob = 0.60
        elif risk_score >= 2:
            staging = "Suspected Moderate Risk"
            prob = 0.35
        else:
            staging = "Suspected Low Risk"
            prob = 0.15
            
    return prob, staging, details

def evaluate_chronic_risk(
    profile: PatientProfile,
    vitals: Vitals,
    symptoms: list[str]
) -> dict[str, Any]:
    """Orchestrates comprehensive Diabetes and Hypertension risk assessments."""
    # 1. Parse/Infer parameters for IDRS
    activity = infer_physical_activity(symptoms, profile.known_conditions)
    fam_hist = infer_family_history(symptoms, profile.known_conditions)
    waist = infer_waist_circumference(symptoms, profile.gender)
    
    # 2. Calculate IDRS
    idrs_score, idrs_breakdown = calculate_idrs(
        age=profile.age,
        gender=profile.gender,
        waist_circumference=waist,
        physical_activity=activity,
        family_history=fam_hist
    )
    
    # Map IDRS to a probability
    if idrs_score >= 60:
        diab_prob = 0.70 + (idrs_score - 60) / 40 * 0.25 # up to 0.95
    elif idrs_score >= 30:
        diab_prob = 0.30 + (idrs_score - 30) / 30 * 0.35 # up to 0.65
    else:
        diab_prob = 0.05 + (idrs_score) / 30 * 0.20 # up to 0.25
        
    # Add clinical values if glucose is measured
    glucose = vitals.blood_sugar_mg_dl
    if glucose is not None:
        idrs_breakdown["glucose_measured"] = True
        # Severe hyperglycemia increases probability significantly
        if glucose >= 200:
            diab_prob = max(diab_prob, 0.90)
        elif glucose >= 126:
            diab_prob = max(diab_prob, 0.75)
    else:
        idrs_breakdown["glucose_measured"] = False
        
    # 3. Calculate Hypertension Risk
    htn_prob, htn_staging, htn_details = calculate_hypertension_risk(vitals, profile, symptoms)
    
    # 4. Determine Severity Tier based on BP, Glucose, and IDRS
    # Rules matching:
    # EMERGENCY: BP >= 180/120 or Glucose >= 400 or Glucose < 50
    # URGENT: BP >= 160/100 or Glucose >= 126 or IDRS >= 80 or HbA1c >= 6.5
    # HIGH: BP >= 140/90 or Glucose >= 100 or IDRS >= 60
    # MEDIUM: BP >= 120/80 or IDRS 30-50
    # LOW: Else
    sbp = vitals.bp_systolic
    dbp = vitals.bp_diastolic
    
    tier = SeverityTier.LOW
    rules_fired = []
    
    def _rank(t):
        return [SeverityTier.LOW, SeverityTier.MEDIUM, SeverityTier.HIGH, SeverityTier.URGENT, SeverityTier.EMERGENCY].index(t)
    
    # 1. EMERGENCY checks
    if (sbp is not None and sbp >= 180) or (dbp is not None and dbp >= 120):
        tier = max(tier, SeverityTier.EMERGENCY, key=_rank)
        rules_fired.append("bp_hypertensive_crisis")
    if glucose is not None and (glucose >= 400 or glucose < 50):
        tier = max(tier, SeverityTier.EMERGENCY, key=_rank)
        rules_fired.append("glucose_crisis")
        
    # 2. URGENT checks
    if (sbp is not None and sbp >= 160) or (dbp is not None and dbp >= 100):
        if "bp_hypertensive_crisis" not in rules_fired:
            tier = max(tier, SeverityTier.URGENT, key=_rank)
            rules_fired.append("bp_stage2_hypertension")
    if glucose is not None and glucose >= 126:
        if "glucose_crisis" not in rules_fired:
            tier = max(tier, SeverityTier.URGENT, key=_rank)
            rules_fired.append("glucose_diabetic_range")
    if idrs_score >= 80:
        tier = max(tier, SeverityTier.URGENT, key=_rank)
        rules_fired.append("idrs_extremely_high")
        
    # 3. HIGH checks
    if (sbp is not None and sbp >= 140) or (dbp is not None and dbp >= 90):
        if not {"bp_hypertensive_crisis", "bp_stage2_hypertension"} & set(rules_fired):
            tier = max(tier, SeverityTier.HIGH, key=_rank)
            rules_fired.append("bp_stage1_hypertension")
    if glucose is not None and glucose >= 100:
        if not {"glucose_crisis", "glucose_diabetic_range"} & set(rules_fired):
            tier = max(tier, SeverityTier.HIGH, key=_rank)
            rules_fired.append("glucose_elevated")
    if idrs_score >= 60:
        if "idrs_extremely_high" not in rules_fired:
            tier = max(tier, SeverityTier.HIGH, key=_rank)
            rules_fired.append("idrs_high")
            
    # 4. MEDIUM checks
    if (sbp is not None and (sbp >= 120 or dbp >= 80)) or idrs_score >= 30:
        if not rules_fired:
            tier = max(tier, SeverityTier.MEDIUM, key=_rank)
            rules_fired.append("mild_elevated_risk")
            
    # 5. LOW fallback
    if not rules_fired:
        tier = SeverityTier.LOW
        rules_fired.append("low_risk_screening")
        
    # Calculate confidence score
    # High if vitals are completely filled, lower if missing
    confidence = 0.95
    if vitals.bp_systolic is None or vitals.bp_diastolic is None:
        confidence -= 0.25
    if vitals.blood_sugar_mg_dl is None:
        confidence -= 0.15
    if waist is None:
        confidence -= 0.10
    confidence = max(0.40, min(1.0, confidence))
    
    # Tier probabilities mock
    idx = [SeverityTier.LOW, SeverityTier.MEDIUM, SeverityTier.HIGH, SeverityTier.URGENT, SeverityTier.EMERGENCY].index(tier)
    probs = {}
    for i, t in enumerate([SeverityTier.LOW, SeverityTier.MEDIUM, SeverityTier.HIGH, SeverityTier.URGENT, SeverityTier.EMERGENCY]):
        if i == idx:
            probs[t.value] = 0.70 + (confidence * 0.25)
        else:
            # Distribute remaining
            probs[t.value] = (1.0 - (0.70 + (confidence * 0.25))) / 4.0
            
    return {
        "tier": tier,
        "probs": probs,
        "diabetes_prob": diab_prob,
        "diabetes_idrs": idrs_score,
        "diabetes_breakdown": idrs_breakdown,
        "hypertension_prob": htn_prob,
        "hypertension_staging": htn_staging,
        "hypertension_details": htn_details,
        "rules_fired": rules_fired,
        "confidence": confidence,
    }
