"""Deterministic lab abnormal detector. Regex extraction + hardcoded ranges.

This module is the GROUND TRUTH for the hallucination guard around lab explanations.
It cannot be influenced by an LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pulsepoint_ai.core.config import get_lab_ranges
from pulsepoint_ai.core.schemas.common import Gender
from pulsepoint_ai.core.schemas.lab import LabFlag, LabStatus

_VALUE_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class _RangeKey:
    age: int
    gender: Gender


def _pick_range(test_cfg: dict, key: _RangeKey) -> tuple[float | None, float | None]:
    ranges = test_cfg.get("ranges", {})
    if key.age < 18 and "pediatric" in ranges:
        r = ranges["pediatric"]
    elif key.gender == Gender.MALE and "male_adult" in ranges:
        r = ranges["male_adult"]
    elif key.gender == Gender.FEMALE and "female_adult" in ranges:
        r = ranges["female_adult"]
    elif "adult" in ranges:
        r = ranges["adult"]
    else:
        return None, None
    return float(r["low"]), float(r["high"])


def _classify(value: float, low: float | None, high: float | None) -> LabStatus:
    if low is None or high is None:
        return LabStatus.UNKNOWN
    if value < low:
        return LabStatus.LOW
    if value > high:
        return LabStatus.HIGH
    return LabStatus.NORMAL


def detect_labs(ocr_text: str, age: int, gender: Gender) -> tuple[list[LabFlag], int]:
    """Returns (all_lab_results, count_normal). Caller filters to flagged subset
    if needed."""
    cfg = get_lab_ranges()["tests"]
    text = ocr_text.lower()
    key = _RangeKey(age=age, gender=gender)

    results: list[LabFlag] = []
    seen: set[str] = set()

    for canonical, test_cfg in cfg.items():
        if canonical in seen:
            continue
        for alias in test_cfg["aliases"]:
            # match alias followed by ":" or whitespace then a number
            pattern = re.compile(
                rf"\b{re.escape(alias)}\b[\s:=]*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE
            )
            m = pattern.search(text)
            if m:
                try:
                    value = float(m.group(1))
                except ValueError:
                    continue
                low, high = _pick_range(test_cfg, key)
                status = _classify(value, low, high)
                results.append(
                    LabFlag(
                        name=alias,
                        canonical=canonical,
                        value=value,
                        unit=test_cfg.get("unit", ""),
                        range_low=low,
                        range_high=high,
                        status=status,
                    )
                )
                seen.add(canonical)
                break

    normal = sum(1 for r in results if r.status == LabStatus.NORMAL)
    return results, normal
