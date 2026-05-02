"""Hallucination guard: set difference between deterministic ground-truth and LLM output.

Used on every LLM response that asserts factual medical items (lab flags, condition codes,
red flags, ICD-10 codes). LLM-only items that aren't in the ground-truth set are blocked.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardResult:
    passed: bool
    blocked: list[str]
    cleaned: list[str]


def set_difference_guard(
    deterministic: set[str],
    llm_output: set[str],
    *,
    case_insensitive: bool = True,
) -> GuardResult:
    """Block any item produced by the LLM that wasn't in the deterministic set."""
    if case_insensitive:
        norm_det = {x.strip().lower() for x in deterministic}
        norm_out = {x.strip().lower() for x in llm_output}
    else:
        norm_det = set(deterministic)
        norm_out = set(llm_output)

    invented = norm_out - norm_det
    cleaned = norm_out & norm_det
    return GuardResult(
        passed=len(invented) == 0,
        blocked=sorted(invented),
        cleaned=sorted(cleaned),
    )
