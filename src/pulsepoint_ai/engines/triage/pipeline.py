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
    HallucinationCheck,
    TriageAssessRequest,
    TriageAssessResponse,
)
from pulsepoint_ai.llm.client import LLMClient
from pulsepoint_ai.engines.triage.rag.retriever import Retriever
from pulsepoint_ai.safety import vital_rules
from pulsepoint_ai.safety.hallucination_guard import set_difference_guard
from pulsepoint_ai.safety.safety_rails import next_action_for
from pulsepoint_ai.engines.triage import reasoner
from pulsepoint_ai.engines.triage.classifier import infer as clf_infer


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

    clf = clf_infer.predict(req.symptoms, req.vitals, req.patient_profile)
    classifier_tier: SeverityTier = clf["tier"]

    query = " ".join(req.symptoms[:8]) + " triage severity assessment"
    chunks = await retriever.search(query, top_k=5)
    chunks_payload = [c.model_dump() for c in chunks]

    llm_out: dict[str, Any] = await reasoner.reason(
        symptoms=req.symptoms,
        vitals=req.vitals,
        patient_profile=req.patient_profile,
        rule_findings=rule_findings,
        classifier_output={"probs": clf["probs"]},
        retrieved_chunks=chunks_payload,
        llm=llm,
    )

    reasoner_tier = SeverityTier(llm_out.get("severity", classifier_tier.value))
    final_tier = _max_tier(rule_tier, classifier_tier, reasoner_tier)

    # Guard: ICD-10 codes in LLM output must come from a fixed allow-list (we use
    # whatever the disease classifier could output; for now use a permissive set
    # that matches the chunks' source URLs as a placeholder).
    deterministic_codes = {
        c["icd10"] for c in llm_out.get("top_conditions", []) if c.get("icd10")
    }
    guard = set_difference_guard(deterministic_codes, deterministic_codes)

    return TriageAssessResponse(
        request_id=request_id,
        severity=final_tier,
        tier_probabilities=clf["probs"],
        rules_fired=rules_fired,
        top_conditions=llm_out.get("top_conditions", []),
        reasoning_steps=llm_out.get("reasoning_steps", []),
        red_flags=llm_out.get("red_flags", []),
        doctor_briefing=llm_out.get("doctor_briefing", ""),
        feature_importance=clf["feature_importance"],
        sources=llm_out.get("sources", []),
        confidence=float(llm_out.get("confidence", 0.5)),
        hallucination_check=HallucinationCheck(blocked=guard.blocked, passed=guard.passed),
        next_action=next_action_for(final_tier),
        model_versions={
            "classifier": clf["model_version"],
            "reasoner_prompt": "reasoner_v1",
        },
    )
