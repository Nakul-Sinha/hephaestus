"""Plan optimizer that ranks intervention plans under constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ml.aegis.planning.constraints import build_constraint_set, evaluate_plan_feasibility
from ml.aegis.planning.objective import ObjectiveWeights, compute_score


@dataclass(frozen=True)
class OptimizationResult:
	"""Structured optimization output used by adapter integration."""

	recommended_plan_id: str
	ranked_plans: list[dict[str, Any]]
	feasible_count: int
	confidence: float


class PlanOptimizer:
	"""Apply constraints and score/rank intervention plans."""

	def __init__(self, weights: ObjectiveWeights | None = None) -> None:
		self.weights = weights or ObjectiveWeights()

	def optimize(self, plans: list[dict[str, Any]], constraints: dict[str, Any]) -> OptimizationResult:
		"""Evaluate feasibility and rank plans by scalar objective score."""
		if not plans:
			raise ValueError("at least one plan is required for optimization")

		normalized_constraints = build_constraint_set(constraints)
		scored: list[dict[str, Any]] = []
		for plan in plans:
			candidate = dict(plan)
			reasons = evaluate_plan_feasibility(candidate, normalized_constraints)
			candidate["is_feasible"] = len(reasons) == 0
			candidate["infeasibility_reasons"] = reasons
			candidate["optimizer_score"] = compute_score(candidate, self.weights)
			scored.append(candidate)

		ranked = sorted(scored, key=lambda row: float(row["optimizer_score"]), reverse=True)
		feasible = [row for row in ranked if row["is_feasible"]]
		recommended = feasible[0] if feasible else ranked[0]

		confidence = 0.84 if feasible else 0.58
		if len(ranked) > 1:
			top = float(ranked[0]["optimizer_score"])
			second = float(ranked[1]["optimizer_score"])
			margin = max(0.0, min(1.0, (top - second) / 20.0))
			confidence = min(0.95, confidence + margin * 0.1)

		return OptimizationResult(
			recommended_plan_id=str(recommended["plan_id"]),
			ranked_plans=ranked,
			feasible_count=len(feasible),
			confidence=round(float(confidence), 4),
		)
