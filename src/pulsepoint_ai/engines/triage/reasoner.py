"""RAG-grounded LLM reasoner. Stub — wire the actual LLM client in src/llm/client.py.

Inputs: rule findings, classifier output, retrieved chunks.
Output: strict JSON validated against TriageAssessResponse fields.
"""
from __future__ import annotations

from typing import Any

from pulsepoint_ai.core.schemas.common import PatientProfile, Vitals
from pulsepoint_ai.llm.client import LLMClient
from pulsepoint_ai.llm.prompts import render_prompt


async def reason(
    *,
    symptoms: list[str],
    vitals: Vitals,
    patient_profile: PatientProfile,
    rule_findings: list[dict[str, Any]],
    classifier_output: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    llm: LLMClient,
) -> dict[str, Any]:
    prompt = render_prompt(
        "reasoner_v1",
        symptoms=symptoms,
        vitals=vitals.model_dump(exclude_none=True),
        patient_profile=patient_profile.model_dump(),
        rule_findings=rule_findings,
        classifier_output=classifier_output,
        retrieved_chunks=retrieved_chunks,
    )
    response = await llm.complete_json(prompt, prompt_version="reasoner_v1")
    return response
