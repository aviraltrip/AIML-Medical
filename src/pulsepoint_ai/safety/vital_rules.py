"""Deterministic vital-signs rule engine.

This module is the SAFETY FLOOR. The LLM cannot override the tier produced here.
Rules are loaded from configs/triage_rules.yaml — never hardcoded in Python.
"""
from __future__ import annotations

from dataclasses import dataclass

from pulsepoint_ai.core.config import get_triage_rules
from pulsepoint_ai.core.schemas.common import SeverityTier, Vitals


@dataclass(frozen=True)
class FiredRule:
    rule_id: str
    tier: SeverityTier
    message: str


_OPS = {
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "eq": lambda a, b: a == b,
}


def _max_tier(tiers: list[SeverityTier], rank: dict[str, int]) -> SeverityTier:
    return max(tiers, key=lambda t: rank[t.value])


def evaluate_vitals(vitals: Vitals) -> tuple[SeverityTier, list[FiredRule]]:
    cfg = get_triage_rules()
    rank = cfg["tier_rank"]
    fired: list[FiredRule] = []

    for rule in cfg["rules"]:
        if "metric_pair" in rule:
            metrics = rule["metric_pair"]
            values = [getattr(vitals, m, None) for m in metrics]
            if any(v is None for v in values):
                continue
            thresholds = rule["value"]
            if rule["op"] == "any_gte":
                if any(v >= t for v, t in zip(values, thresholds, strict=True)):
                    fired.append(
                        FiredRule(
                            rule_id=rule["id"],
                            tier=SeverityTier(rule["tier"]),
                            message=rule["message"],
                        )
                    )
            continue

        metric = rule["metric"]
        value = getattr(vitals, metric, None)
        if value is None:
            continue
        op_fn = _OPS.get(rule["op"])
        if op_fn is None:
            continue
        if op_fn(value, rule["value"]):
            fired.append(
                FiredRule(
                    rule_id=rule["id"],
                    tier=SeverityTier(rule["tier"]),
                    message=rule["message"],
                )
            )

    if not fired:
        return SeverityTier.LOW, []

    tier = _max_tier([f.tier for f in fired], rank)
    return tier, fired
