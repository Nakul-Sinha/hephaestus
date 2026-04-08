"""Constraint utilities for plan feasibility validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConstraintSet:
    """Normalized optimization constraints used by planning modules."""

    budget_ceiling: float
    available_crew: dict[str, int]
    spare_parts_inventory: set[str]
    blackout_windows: set[str]


def build_constraint_set(constraints: dict[str, Any]) -> ConstraintSet:
    """Build normalized constraints from backend request payload."""
    return ConstraintSet(
        budget_ceiling=float(constraints.get("budget_ceiling", 0.0)),
        available_crew={
            str(role): int(count)
            for role, count in dict(constraints.get("available_crew", {})).items()
        },
        spare_parts_inventory={str(part) for part in list(constraints.get("spare_parts_inventory", []))},
        blackout_windows={str(window) for window in list(constraints.get("blackout_windows", []))},
    )


def evaluate_plan_feasibility(plan: dict[str, Any], constraints: ConstraintSet) -> list[str]:
    """Return infeasibility reasons for a plan under the given constraints."""
    reasons: list[str] = []

    estimated_cost = float(plan.get("estimated_cost", 0.0))
    if estimated_cost > constraints.budget_ceiling:
        reasons.append("budget ceiling exceeded")

    required_skills = [str(skill) for skill in list(plan.get("required_skills", []))]
    for skill in required_skills:
        if constraints.available_crew and constraints.available_crew.get(skill, 0) <= 0:
            reasons.append(f"missing skill: {skill}")

    required_parts = [str(part) for part in list(plan.get("required_parts", []))]
    for part in required_parts:
        if constraints.spare_parts_inventory and part not in constraints.spare_parts_inventory:
            reasons.append(f"missing part: {part}")

    maintenance_window = str(plan.get("maintenance_window", "any"))
    if maintenance_window in constraints.blackout_windows:
        reasons.append(f"maintenance window blocked: {maintenance_window}")

    return reasons
