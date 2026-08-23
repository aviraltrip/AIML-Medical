"""Hardcoded safety rails. These are not configurable at runtime."""
from __future__ import annotations

import re

from pulsepoint_ai.core.schemas.common import SeverityTier

DISCLAIMER = (
    "Not a diagnosis. PulsePoint provides AI-assisted triage and information; "
    "always consult a qualified clinician for medical decisions."
)

EMERGENCY_NUMBER_IN = "108"

NEXT_ACTION_BY_TIER: dict[SeverityTier, str] = {
    SeverityTier.LOW: "lifestyle counseling and monthly follow-up by CHW",
    SeverityTier.MEDIUM: "repeat BP after rest and lifestyle counseling",
    SeverityTier.HIGH: "HbA1c screening and fasting glucose test",
    SeverityTier.URGENT: "PHC referral for medical evaluation within 48h",
    SeverityTier.EMERGENCY: "immediate PHC referral required due to crisis-level screening values",
}


_BANNED_OUTPUT_PATTERNS = [
    re.compile(r"\btake\s+\d+\s*(mg|g|ml)\b", re.IGNORECASE),
    re.compile(r"\bprescribe\b", re.IGNORECASE),
    re.compile(r"\bdose:\s*\d+", re.IGNORECASE),
]


def scrub_medication_advice(text: str) -> tuple[str, list[str]]:
    """Returns (scrubbed_text, list_of_blocked_phrases). LLM is never allowed
    to suggest specific medication dosing."""
    blocked: list[str] = []
    out = text
    for pat in _BANNED_OUTPUT_PATTERNS:
        for m in pat.finditer(text):
            blocked.append(m.group(0))
        out = pat.sub("[redacted: consult your doctor]", out)
    return out, blocked


def next_action_for(tier: SeverityTier) -> str:
    return NEXT_ACTION_BY_TIER[tier]
