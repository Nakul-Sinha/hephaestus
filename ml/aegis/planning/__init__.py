"""Planning module exports for constraints and objective scoring."""

from __future__ import annotations

from typing import Any

from ml.aegis.planning.constraints import ConstraintSet, build_constraint_set, evaluate_plan_feasibility
from ml.aegis.planning.objective import ObjectiveWeights, compute_score

# Backward-compatible aliases for older imports used in downstream modules.
PlanningConstraints = ConstraintSet
PolicyWeights = ObjectiveWeights


def check_plan_feasibility(plan: dict[str, Any], constraints: PlanningConstraints | dict[str, Any]) -> list[str]:
    """Compatibility wrapper for feasibility checks."""
    normalized = constraints if isinstance(constraints, ConstraintSet) else build_constraint_set(constraints)
    return evaluate_plan_feasibility(plan, normalized)


def score_plan(plan: dict[str, Any], weights: PolicyWeights | None = None) -> float:
    """Compatibility wrapper for objective score computation."""
    return compute_score(plan, weights or ObjectiveWeights())


def rank_plans(plans: list[dict[str, Any]], weights: PolicyWeights | None = None) -> list[dict[str, Any]]:
    """Compatibility helper that ranks plans by descending score."""
    policy = weights or ObjectiveWeights()
    return sorted(plans, key=lambda plan: score_plan(plan, policy), reverse=True)


POLICY_PROFILES: dict[str, PolicyWeights] = {
    "default": ObjectiveWeights(),
}


__all__ = [
    "ConstraintSet",
    "PlanningConstraints",
    "build_constraint_set",
    "evaluate_plan_feasibility",
    "check_plan_feasibility",
    "ObjectiveWeights",
    "PolicyWeights",
    "compute_score",
    "score_plan",
    "rank_plans",
    "POLICY_PROFILES",
]
