"""Objective scoring functions for intervention plan ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights for plan ranking objective terms."""

    risk_reduction: float = 1.0
    cost: float = 0.08
    downtime: float = 0.12
    sla_compliance: float = 0.2


def estimate_sla_compliance(plan: dict[str, Any]) -> float:
    """Estimate SLA compliance likelihood from action profile."""
    downtime_minutes = max(0.0, float(plan.get("expected_downtime_minutes", 0.0)))
    action = str(plan.get("recommended_action", "")).lower()

    base = 1.0 - min(0.95, downtime_minutes / (24.0 * 60.0))
    if "defer" in action:
        base -= 0.2
    elif "replace" in action:
        base += 0.08
    elif "monitor" in action:
        base -= 0.05

    return float(max(0.05, min(1.0, base)))


def compute_score(plan: dict[str, Any], weights: ObjectiveWeights) -> float:
    """Compute scalar optimizer score using normalized objective terms."""
    risk_reduction = float(plan.get("predicted_risk_reduction", 0.0))
    estimated_cost = float(plan.get("estimated_cost", 0.0))
    downtime_minutes = float(plan.get("expected_downtime_minutes", 0.0))
    sla_compliance = estimate_sla_compliance(plan)

    score = (
        weights.risk_reduction * (risk_reduction * 100.0)
        - weights.cost * (estimated_cost / 1000.0)
        - weights.downtime * (downtime_minutes / 60.0)
        + weights.sla_compliance * (sla_compliance * 10.0)
    )
    return round(float(score), 4)
