"""Deterministic lab abnormal detector. Regex extraction + hardcoded ranges.

This module is the GROUND TRUTH for the hallucination guard around lab explanations.
It cannot be influenced by an LLM. It is optimized for low-resource rural OCR noise.
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
    """Returns (all_lab_results, count_normal). Matches fuzzy, noisy OCR text."""
    cfg = get_lab_ranges()["tests"]


    text = " ".join(ocr_text.split()).lower()
    key = _RangeKey(age=age, gender=gender)

    results: list[LabFlag] = []
    seen: set[str] = set()

    for canonical, test_cfg in cfg.items():
        if canonical in seen:
            continue
        for alias in test_cfg["aliases"]:

            cleaned_alias = alias.lower()
            cleaned_alias = re.sub(r"[\s\-_.]+", "SPACEPLACEHOLDER", cleaned_alias)


            alias_pattern = re.escape(cleaned_alias)
            alias_pattern = alias_pattern.replace("SPACEPLACEHOLDER", r"[\s\-_.]*")



            pattern = re.compile(
                rf"\b{alias_pattern}\b[^\d]{{0,25}}?([-+]?\d+(?:\.\d+)?)", re.IGNORECASE
            )
            m = pattern.search(text)
            if m:
                try:
                    value = float(m.group(1))
                except ValueError:
                    continue

                low, high = _pick_range(test_cfg, key)
                status = _classify(value, low, high)


                if canonical == "hba1c":

                    status = LabStatus.HIGH if value >= 5.7 else LabStatus.NORMAL
                elif canonical == "fasting_glucose":

                    status = LabStatus.HIGH if value >= 100 else LabStatus.NORMAL
                elif canonical == "random_glucose" or canonical == "postprandial_glucose":

                    status = LabStatus.HIGH if value >= 140 else LabStatus.NORMAL
                elif canonical == "microalbumin":

                    status = LabStatus.HIGH if value >= 30 else LabStatus.NORMAL
                elif canonical == "urine_protein":

                    status = LabStatus.HIGH if value >= 20 else LabStatus.NORMAL

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
