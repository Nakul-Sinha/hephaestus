"""Monte Carlo simulator for intervention plan scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
	"""Runtime settings for Monte Carlo simulation."""

	n_iterations: int = 256
	random_seed: int = 42


def _build_mean_risk_curve(start_risk: float, reduction: float, horizon_days: int) -> np.ndarray:
	start = max(0.01, min(0.99, float(start_risk)))
	target = max(0.01, min(0.99, start * (1.0 - float(reduction))))
	drift = np.linspace(start, target, num=horizon_days)
	tail = np.linspace(target, min(0.99, target + 0.1), num=horizon_days)
	return np.minimum(drift * 0.6 + tail * 0.4, 0.99)


def simulate_plan(
	plan: dict[str, Any],
	horizon_days: int,
	baseline_risk: float,
	config: SimulationConfig | None = None,
) -> dict[str, Any]:
	"""Simulate one plan and return uncertainty-aware outputs."""
	runtime = config or SimulationConfig()
	plan_id = str(plan.get("plan_id", ""))
	plan_offset = sum(ord(char) for char in plan_id) % 10000
	rng = np.random.default_rng(runtime.random_seed + plan_offset)

	reduction = float(plan.get("predicted_risk_reduction", 0.0))
	mean_curve = _build_mean_risk_curve(start_risk=baseline_risk, reduction=reduction, horizon_days=horizon_days)

	noise = rng.normal(0.0, 0.035, size=(runtime.n_iterations, horizon_days))
	risk_samples = np.clip(mean_curve + noise, 0.0, 1.0)

	daily_mean = np.mean(risk_samples, axis=0)
	daily_p5 = np.quantile(risk_samples, 0.05, axis=0)
	daily_p95 = np.quantile(risk_samples, 0.95, axis=0)

	base_cost = float(plan.get("estimated_cost", 0.0))
	cost_samples = np.clip(rng.normal(loc=base_cost, scale=max(1.0, base_cost * 0.08), size=runtime.n_iterations), 0.0, None)

	base_downtime = float(plan.get("expected_downtime_minutes", 0.0)) / 60.0
	downtime_samples = np.clip(
		rng.normal(loc=base_downtime, scale=max(0.1, base_downtime * 0.2), size=runtime.n_iterations),
		0.0,
		None,
	)

	probability_of_failure = float(np.mean(np.max(risk_samples, axis=1) >= 0.8))

	return {
		"plan_id": str(plan.get("plan_id", "unknown")),
		"horizon_days": int(horizon_days),
		"n_iterations": int(runtime.n_iterations),
		"daily_risk_mean": [round(float(v), 4) for v in daily_mean.tolist()],
		"daily_risk_p5": [round(float(v), 4) for v in daily_p5.tolist()],
		"daily_risk_p95": [round(float(v), 4) for v in daily_p95.tolist()],
		"total_expected_cost": round(float(np.mean(cost_samples)), 2),
		"total_expected_downtime_hours": round(float(np.mean(downtime_samples)), 2),
		"probability_of_failure": round(probability_of_failure, 4),
	}
