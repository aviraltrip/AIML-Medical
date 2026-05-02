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
        f"Recent Triage History: {triage_history[-1:]}\n"
        "Instructions:\n"
        "1. Be extremely concise (under 100 words).\n"
        "2. Highlight the most critical symptom and severity tier.\n"
        "3. List any significant vital sign abnormalities.\n"
        "4. Format as a clinical hand-off note."
    )
    
    summary = await llm.complete_json(prompt, prompt_version="medreach_v1")
    return summary.get("summary", "Error generating briefing.")
