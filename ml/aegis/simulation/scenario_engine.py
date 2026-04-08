"""Scenario engine for what-if simulation comparisons."""

from __future__ import annotations

from typing import Any

from ml.aegis.simulation.monte_carlo import SimulationConfig, simulate_plan


def run_scenario_comparison(
	ranked_plans: list[dict[str, Any]],
	horizon_days: int,
	baseline_risk: float,
	n_iterations: int = 256,
) -> dict[str, Any]:
	"""Run simulations for top-ranked plans and compare outcomes."""
	if not ranked_plans:
		raise ValueError("at least one ranked plan is required for simulation")

	top_plans = ranked_plans[:3]
	simulations = [
		simulate_plan(
			plan=plan,
			horizon_days=horizon_days,
			baseline_risk=baseline_risk,
			config=SimulationConfig(n_iterations=n_iterations, random_seed=42 + idx * 7),
		)
		for idx, plan in enumerate(top_plans)
	]

	pairwise: dict[str, dict[str, float]] = {}
	for left in simulations:
		left_plan = left["plan_id"]
		pairwise[left_plan] = {}
		left_risk = left["daily_risk_mean"][-1]
		left_cost = left["total_expected_cost"]
		left_value = float(left_risk) * 0.7 + float(left_cost) / 100000.0

		for right in simulations:
			right_plan = right["plan_id"]
			if left_plan == right_plan:
				pairwise[left_plan][right_plan] = 0.5
				continue

			right_risk = right["daily_risk_mean"][-1]
			right_cost = right["total_expected_cost"]
			right_value = float(right_risk) * 0.7 + float(right_cost) / 100000.0

			margin = max(-0.4, min(0.4, right_value - left_value))
			probability = max(0.05, min(0.95, 0.5 + margin))
			pairwise[left_plan][right_plan] = round(float(probability), 4)

	return {
		"simulations": simulations,
		"pairwise_win_probabilities": pairwise,
	}
