"""Simulation module exports with backward-compatible helper aliases."""

from __future__ import annotations

from typing import Any

from ml.aegis.simulation.impact_estimator import estimate_impact, format_impact_for_report
from ml.aegis.simulation.monte_carlo import SimulationConfig, simulate_plan
from ml.aegis.simulation.scenario_engine import run_scenario_comparison


def simulate_do_nothing(horizon_days: int, baseline_risk: float, n_iterations: int = 256) -> dict[str, Any]:
    """Compatibility helper for baseline simulation without intervention."""
    baseline_plan = {
        "plan_id": "do_nothing",
        "predicted_risk_reduction": 0.0,
        "estimated_cost": 0.0,
        "expected_downtime_minutes": 0,
    }
    return simulate_plan(
        plan=baseline_plan,
        horizon_days=horizon_days,
        baseline_risk=baseline_risk,
        config=SimulationConfig(n_iterations=n_iterations, random_seed=42),
    )


def compare_plans(
    ranked_plans: list[dict[str, Any]],
    horizon_days: int,
    baseline_risk: float,
    n_iterations: int = 256,
) -> dict[str, Any]:
    """Compatibility alias for scenario comparison."""
    return run_scenario_comparison(
        ranked_plans=ranked_plans,
        horizon_days=horizon_days,
        baseline_risk=baseline_risk,
        n_iterations=n_iterations,
    )


def format_for_frontend(comparison: dict[str, Any]) -> dict[str, Any]:
    """Return scenario output in frontend-ready shape (currently passthrough)."""
    return comparison


__all__ = [
    "SimulationConfig",
    "simulate_plan",
    "simulate_do_nothing",
    "run_scenario_comparison",
    "compare_plans",
    "format_for_frontend",
    "estimate_impact",
    "format_impact_for_report",
]
