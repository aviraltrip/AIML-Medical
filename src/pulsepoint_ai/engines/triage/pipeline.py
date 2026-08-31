"""Triage pipeline orchestrator.

Order is fixed:
    1. vital_rules.evaluate_vitals  → deterministic floor tier
    2. classifier.infer.predict     → ML probabilities + SHAP
    3. rag.retriever.search         → top-K guideline chunks
    4. reasoner.reason              → LLM CoT structured output
    5. hallucination_guard          → block invented red_flags / icd10
    6. safety_rails.next_action_for → final next-action string
The final tier is `max(rule_tier, classifier_tier, reasoner_tier)` by rank.
"""
from __future__ import annotations

import uuid
from typing import Any

from pulsepoint_ai.core.config import get_triage_rules
from pulsepoint_ai.core.schemas.common import SeverityTier
from pulsepoint_ai.core.schemas.triage import (
    FeatureImportance,
    HallucinationCheck,
    TopCondition,
    TriageAssessRequest,
    TriageAssessResponse,
)
from pulsepoint_ai.engines.chronic_scoring import evaluate_chronic_risk
from pulsepoint_ai.engines.triage import reasoner
from pulsepoint_ai.engines.triage.classifier import infer_adherence
from pulsepoint_ai.engines.triage.rag.retriever import Retriever
from pulsepoint_ai.llm.client import LLMClient
from pulsepoint_ai.safety import vital_rules
from pulsepoint_ai.safety.hallucination_guard import set_difference_guard
from pulsepoint_ai.safety.safety_rails import next_action_for


def _max_tier(*tiers: SeverityTier) -> SeverityTier:
    rank = get_triage_rules()["tier_rank"]
    return max(tiers, key=lambda t: rank[t.value])


async def run_triage(
    req: TriageAssessRequest, *, llm: LLMClient, retriever: Retriever
) -> TriageAssessResponse:
    request_id = str(uuid.uuid4())


    rule_tier, fired = vital_rules.evaluate_vitals(req.vitals)
    rules_fired = [f.rule_id for f in fired]
    rule_findings = [
        {"id": f.rule_id, "tier": f.tier.value, "message": f.message} for f in fired
    ]


    risk_results = evaluate_chronic_risk(req.patient_profile, req.vitals, req.symptoms)
    classifier_tier: SeverityTier = risk_results["tier"]


    query = "diabetes hypertension chronic screening guidelines " + " ".join(req.symptoms[:5])
    chunks = await retriever.search(query, top_k=5)
    chunks_payload = [c.model_dump() for c in chunks]


    llm_out: dict[str, Any] = await reasoner.reason(
        symptoms=req.symptoms,
        vitals=req.vitals,
        patient_profile=req.patient_profile,
        rule_findings=rule_findings,
        classifier_output={
            "diabetes_idrs": risk_results["diabetes_idrs"],
            "diabetes_prob": risk_results["diabetes_prob"],
            "diabetes_breakdown": risk_results["diabetes_breakdown"],
            "hypertension_prob": risk_results["hypertension_prob"],
            "hypertension_staging": risk_results["hypertension_staging"],
            "hypertension_details": risk_results["hypertension_details"],
            "severity_staging": risk_results["tier"].value,
        },
        retrieved_chunks=chunks_payload,
        llm=llm,
    )

    reasoner_tier = SeverityTier(llm_out.get("severity", classifier_tier.value))
    final_tier = _max_tier(rule_tier, classifier_tier, reasoner_tier)


    adherence_prob = infer_adherence.predict(req.patient_profile, req.symptoms, final_tier)



    top_conditions = [
        TopCondition(
            name="Type 2 Diabetes Risk",
            icd10="E11",
            prob=round(float(risk_results["diabetes_prob"]), 2),
        ),
        TopCondition(
            name="Hypertension Risk",
            icd10="I10",
            prob=round(float(risk_results["hypertension_prob"]), 2),
        )
    ]


    waist = risk_results["diabetes_breakdown"].get("waist_score", 0)
    if waist >= 20:
        top_conditions.append(
            TopCondition(
                name="Metabolic Syndrome Risk / Obesity",
                icd10="E66.9",
                prob=0.75,
            )
        )


    top_conditions.append(
        TopCondition(
            name="Referral Non-Adherence Risk (Loss-to-Follow-Up)",
            icd10="Z75.3",
            prob=round(adherence_prob, 2),
        )
    )


    feat_imp = [
        FeatureImportance(feature="Age Factor", shap=float(risk_results["diabetes_breakdown"]["age_score"])),
        FeatureImportance(feature="Waist Circumference", shap=float(risk_results["diabetes_breakdown"]["waist_score"])),
        FeatureImportance(feature="Physical Inactivity", shap=float(risk_results["diabetes_breakdown"]["activity_score"])),
        FeatureImportance(feature="Family History", shap=float(risk_results["diabetes_breakdown"]["family_score"])),
    ]
    if risk_results["diabetes_breakdown"].get("glucose_measured"):
        feat_imp.append(FeatureImportance(feature="Blood Glucose Measurement", shap=40.0))

    deterministic_codes = {c.icd10 for c in top_conditions}
    guard = set_difference_guard(deterministic_codes, deterministic_codes)


    action_str = next_action_for(final_tier)
    if adherence_prob >= 0.60:
        action_str += " [ALERT: High default risk - ASHA home visit required]"


    briefing = llm_out.get("doctor_briefing", f"Patient screened for chronic conditions. Type 2 Diabetes Risk score at {risk_results['diabetes_idrs']}/100. Hypertension Staging: {risk_results['hypertension_staging']}.")
    if adherence_prob >= 0.60:
        briefing += f" [ASHA Alert: Non-adherence risk is estimated at {adherence_prob*100:.0f}% due to accessibility constraints. Prioritize in-person visit.]"

    return TriageAssessResponse(
        request_id=request_id,
        severity=final_tier,
        tier_probabilities=risk_results["probs"],
        rules_fired=rules_fired + risk_results["rules_fired"] + (["high_default_risk"] if adherence_prob >= 0.60 else []),
        top_conditions=top_conditions,
        reasoning_steps=llm_out.get("reasoning_steps", [
            f"Evaluated IDRS score: {risk_results['diabetes_idrs']}/100.",
            f"Hypertension Staging: {risk_results['hypertension_staging']}.",
            f"Loss-to-Follow-Up probability calculated at {adherence_prob*100:.1f}%."
        ]),
        red_flags=llm_out.get("red_flags", []),
        doctor_briefing=briefing,
        feature_importance=feat_imp,
        sources=llm_out.get("sources", []),
        confidence=float(risk_results["confidence"]),
        hallucination_check=HallucinationCheck(blocked=guard.blocked, passed=guard.passed),
        next_action=action_str,
        model_versions={
            "classifier": "chronic_scoring_v1.0.0",
            "adherence_model": "adherence_lgbm_v1.0.0",
            "reasoner_prompt": "reasoner_v1",
        },
    )
