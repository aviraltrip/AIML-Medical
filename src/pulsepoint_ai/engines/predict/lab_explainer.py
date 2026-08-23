"""LLM-generated plain-language explanations for flagged labs.

The LLM receives the deterministic flag list grounded by clinical knowledge.
After generation, the hallucination guard verifies the LLM's flag IDs match
the deterministic set and a medical disclaimer is appended.
"""
from __future__ import annotations

import jinja2

from pulsepoint_ai.core.config import get_lab_knowledge, get_settings
from pulsepoint_ai.core.schemas.lab import LabFlag, LabStatus
from pulsepoint_ai.llm.client import LLMClient
from pulsepoint_ai.safety.hallucination_guard import set_difference_guard

MEDICAL_DISCLAIMER = (
    "\n\n[Disclaimer: This is an AI-generated explanation for informational purposes only. "
    "Consult a healthcare professional for clinical interpretation.]"
)

def _get_jinja_env() -> jinja2.Environment:
    prompt_path = get_settings().configs_dir / "prompts"
    return jinja2.Environment(loader=jinja2.FileSystemLoader(str(prompt_path)))

async def explain_flags(flags: list[LabFlag], *, llm: LLMClient) -> tuple[list[LabFlag], list[str]]:
    """Generates plain-language explanations for flagged (abnormal) lab results."""
    abnormal = [f for f in flags if f.status in {LabStatus.LOW, LabStatus.HIGH}]
    if not abnormal:
        return flags, []

    deterministic_ids = {f.canonical for f in abnormal}
    knowledge = get_lab_knowledge()


    env = _get_jinja_env()
    template = env.get_template("lab_explainer_v1.j2")
    prompt = template.render(flags=abnormal, knowledge=knowledge)


    out = await llm.complete_json(prompt, prompt_version="lab_explainer_v1")


    by_id = {e["id"]: e["text"] for e in out.get("explanations", []) if isinstance(e, dict)}
    guard = set_difference_guard(deterministic_ids, set(by_id.keys()))

    annotated: list[LabFlag] = []
    for f in flags:
        if f.canonical in by_id and f.canonical not in guard.blocked:
            explanation = by_id[f.canonical].strip()
            if not explanation.endswith(MEDICAL_DISCLAIMER):
                explanation += MEDICAL_DISCLAIMER

            annotated.append(f.model_copy(update={"explanation": explanation}))
        else:
            annotated.append(f)

    return annotated, guard.blocked
