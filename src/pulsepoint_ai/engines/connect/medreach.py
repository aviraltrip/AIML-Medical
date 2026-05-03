"""MedReach Assistant: Summarizes patient data for async doctor consultation."""
from __future__ import annotations

from typing import Any, Dict
from pulsepoint_ai.llm.client import LLMClient

async def summarize_for_doctor(
    patient_data: Dict[str, Any],
    triage_history: list[Dict[str, Any]],
    *,
    llm: LLMClient
) -> str:
    """Generates a concise briefing for the doctor's MedReach queue."""
    
    prompt = (
        "You are a Clinical Assistant. Summarize this patient's case for a doctor.\n"
        f"Patient Profile: {patient_data}\n"
        f"Recent Triage History: {triage_history[-1:] if triage_history else []}\n"
        "Instructions:\n"
        "1. Be extremely concise (under 100 words).\n"
        "2. Highlight the most critical symptom and severity tier.\n"
        "3. List any significant vital sign abnormalities.\n"
        "4. Write the summary as a single clinical hand-off note paragraph.\n"
        'Output strict JSON only: {"summary": "<the hand-off note as a single string>"}'
    )

    try:
        result = await llm.complete_json(prompt, prompt_version="medreach_v1")
        summary = result.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary
        # Tolerate alternate keys the LLM may pick
        for k in ("note", "briefing", "handoff", "text"):
            v = result.get(k)
            if isinstance(v, str) and v.strip():
                return v
    except Exception as e:
        print(f"MedReach summarize failed: {e}")

    # Last-resort deterministic fallback so the API never returns a misleading error string
    chief = patient_data.get("chief_complaint") or "unspecified complaint"
    age = patient_data.get("age", "unknown age")
    gender = patient_data.get("gender", "unknown gender")
    last = triage_history[-1] if triage_history else {}
    severity = last.get("severity", "unspecified severity")
    return (
        f"{age}-y/o {gender} presenting with {chief}. "
        f"Most recent triage severity: {severity}. "
        "Detailed AI summary unavailable; please review raw record."
    )
