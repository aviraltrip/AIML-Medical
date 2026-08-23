"""Condition Cards: RAG-grounded patient education for predicted diseases."""
from __future__ import annotations

import uuid
from typing import Any

from pulsepoint_ai.core.schemas.predict import ConditionCardRequest, ConditionCardResponse
from pulsepoint_ai.engines.triage.rag.retriever import Retriever
from pulsepoint_ai.llm.client import LLMClient

_retriever_singleton: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = Retriever()
    return _retriever_singleton


_ICD10_NAMES = {
    "J45.909": "Asthma",
    "J18.9": "Pneumonia",
    "I21.9": "Acute Myocardial Infarction",
    "J06.9": "Upper Respiratory Infection",
    "R51": "Headache",
    "K59.00": "Constipation",
    "E11.9": "Type 2 Diabetes",
    "I10": "Hypertension",
    "R50.9": "Fever",
    "J00": "Common Cold",
    "I26.9": "Pulmonary Embolism",
    "I50.9": "Acute Heart Failure",
}


async def generate_condition_card(condition_name: str, *, llm: LLMClient, retriever: Retriever) -> dict[str, Any]:
    query = f"What is {condition_name} and what are the first aid or action steps?"
    chunks = await retriever.search(query, top_k=2)
    context = "\n".join([c.text for c in chunks])

    prompt = (
        f"Generate a medical condition card for '{condition_name}'.\n"
        f"Context from medical guidelines: {context}\n"
        "Instructions:\n"
        "1. Reading level: Grade 6.\n"
        "2. No jargon.\n"
        "3. Provide exactly 3 clear action steps.\n"
        "4. Max 80 words.\n"
        'Output strict JSON: {"summary": str, "actions": [str, str, str]}'
    )
    return await llm.complete_json(prompt, prompt_version="condition_card_v1")


async def get_card(request: ConditionCardRequest, *, llm: LLMClient) -> ConditionCardResponse:
    """Router-facing entry point: takes a ConditionCardRequest and returns the response shape."""
    name = request.name or _ICD10_NAMES.get(request.icd10, request.icd10)
    retriever = _get_retriever()

    try:
        card = await generate_condition_card(name, llm=llm, retriever=retriever)
        actions = card.get("actions") or card.get("action_steps") or []
        if not isinstance(actions, list):
            actions = [str(actions)]
        summary = card.get("summary") or card.get("plain_summary") or ""
        unsupported = False
    except Exception as e:
        print(f"Condition card generation failed: {e}")
        summary = (
            f"{name} is a medical condition. We couldn't fetch a tailored summary right now — "
            "please consult a clinician for personal guidance."
        )
        actions = [
            "Talk to a doctor about your symptoms.",
            "Follow any medications already prescribed to you.",
            "Seek urgent care if symptoms get worse.",
        ]
        unsupported = True

    return ConditionCardResponse(
        request_id=str(uuid.uuid4()),
        name=name,
        plain_summary=str(summary),
        action_steps=[str(a) for a in actions][:5],
        sources=[],
        unsupported=unsupported,
    )
