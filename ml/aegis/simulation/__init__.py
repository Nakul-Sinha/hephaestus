"""
Simulation module — Monte Carlo projections and scenario analysis.

Modules:
    monte_carlo:     Core simulation engine (vectorized NumPy)
    scenario_engine: Side-by-side plan comparison with pairwise stats
    impact_estimator: Business impact metrics (ROI, savings, downtime avoided)
"""

from ml.aegis.simulation.monte_carlo import simulate_plan, simulate_do_nothing, SimulationConfig
from ml.aegis.simulation.scenario_engine import compare_plans, format_for_frontend
from ml.aegis.simulation.impact_estimator import estimate_impact, format_impact_for_report

__all__ = [
    "simulate_plan", "simulate_do_nothing", "SimulationConfig",
    "compare_plans", "format_for_frontend",
    "estimate_impact", "format_impact_for_report",
]
