"""
Impact Estimator

Computes the business-impact delta between "do nothing" and the selected
intervention plan. Produces the dollar/hour savings numbers that go in
the Manager Summary report and the hackathon pitch slides.

Metrics:
    - downtime_avoided_hours
    - cost_saved_vs_reactive (what it would cost if we waited for failure)
    - cost_saved_vs_scheduled (what a fixed-schedule policy would cost)
    - risk_reduction_percentage
    - roi (return on investment of the intervention)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ml.aegis.data.schemas import ComparisonResult, SimulationResult


# ---------------------------------------------------------------------------
# Cost assumptions (configurable per deployment)
# ---------------------------------------------------------------------------

@dataclass
class CostAssumptions:
    """Industry-average cost parameters for impact calculation."""

    # Unplanned failure cost: emergency repair + production loss + penalties
    reactive_failure_cost: float = 25_000.0

    # Hourly cost of unplanned downtime (production loss)
    unplanned_downtime_cost_per_hour: float = 5_000.0

    # Cost of a fixed-schedule maintenance visit (regardless of condition)
    scheduled_maintenance_cost: float = 3_500.0

    # Average unplanned repair duration in hours
    reactive_repair_hours: float = 12.0

    # Scheduled maintenance frequency (days)
    scheduled_maintenance_interval_days: int = 90


# ---------------------------------------------------------------------------
# Impact Calculation
# ---------------------------------------------------------------------------

def estimate_impact(
    comparison: ComparisonResult,
    cost_assumptions: CostAssumptions | None = None,
) -> dict[str, Any]:
    """
    Compute the business impact of the best intervention plan vs alternatives.

    Args:
        comparison: ComparisonResult from the scenario engine.
        cost_assumptions: Cost parameters.

    Returns:
        {
            "vs_do_nothing": {
                "risk_reduction_pct": 72.3,
                "failure_prob_reduction_pct": 65.0,
                "downtime_avoided_hours": 10.5,
                "cost_of_intervention": 2200.0,
                "cost_of_failure_avoided": 25000.0,
                "net_savings": 22800.0,
                "roi_pct": 1036.4
            },
            "vs_scheduled": {
                "cost_difference": -1300.0,
                "precision_advantage": "Condition-based: intervene only when needed"
            },
            "30_day_projection": {
                "do_nothing_expected_cost": 18750.0,
                "intervention_expected_cost": 2200.0,
                "scheduled_expected_cost": 3500.0,
                "optimal_savings_vs_reactive": 16550.0
            }
        }
    """
    if cost_assumptions is None:
        cost_assumptions = CostAssumptions()

    baseline = comparison.baseline_trajectory
    best_plan_id = comparison.best_plan_id

    # Find best plan simulation
    best_sim = next(
        (s for s in comparison.plan_trajectories if s.plan_id == best_plan_id),
        None,
    )

    if best_sim is None:
        # No intervention plans — return empty impact
        return _empty_impact()

    # --- vs Do Nothing ---
    baseline_day30 = baseline.daily_risk_mean[-1] if baseline.daily_risk_mean else 0.0
    best_day30 = best_sim.daily_risk_mean[-1] if best_sim.daily_risk_mean else 0.0

    risk_reduction_pct = round(
        (baseline_day30 - best_day30) / max(baseline_day30, 1e-6) * 100, 1,
    )

    failure_prob_baseline = baseline.probability_of_failure
    failure_prob_best = best_sim.probability_of_failure
    failure_prob_reduction_pct = round(
        (failure_prob_baseline - failure_prob_best) * 100, 1,
    )

    # Downtime avoided: reactive repair hours * probability delta
    downtime_avoided_hours = round(
        cost_assumptions.reactive_repair_hours * (failure_prob_baseline - failure_prob_best),
        1,
    )

    # Cost of failure avoided (expected value)
    cost_of_failure_avoided = round(
        cost_assumptions.reactive_failure_cost * (failure_prob_baseline - failure_prob_best),
        2,
    )

    # Unplanned downtime cost avoided
    downtime_cost_avoided = round(
        downtime_avoided_hours * cost_assumptions.unplanned_downtime_cost_per_hour,
        2,
    )

    total_avoided = cost_of_failure_avoided + downtime_cost_avoided
    intervention_cost = best_sim.total_expected_cost
    net_savings = round(total_avoided - intervention_cost, 2)

    roi_pct = round(
        net_savings / max(intervention_cost, 1.0) * 100, 1,
    ) if intervention_cost > 0 else 0.0

    # --- vs Scheduled ---
    scheduled_cost = cost_assumptions.scheduled_maintenance_cost
    cost_difference = round(intervention_cost - scheduled_cost, 2)

    # --- 30-day projection ---
    do_nothing_expected = round(
        failure_prob_baseline * cost_assumptions.reactive_failure_cost
        + failure_prob_baseline * cost_assumptions.reactive_repair_hours
        * cost_assumptions.unplanned_downtime_cost_per_hour,
        2,
    )

    scheduled_expected = round(
        scheduled_cost + (1 - 0.7) * failure_prob_baseline  # Scheduled reduces ~70% of risk
        * cost_assumptions.reactive_failure_cost * 0.3,
        2,
    )

    return {
        "vs_do_nothing": {
            "risk_reduction_pct": risk_reduction_pct,
            "failure_prob_reduction_pct": failure_prob_reduction_pct,
            "downtime_avoided_hours": downtime_avoided_hours,
            "downtime_cost_avoided": downtime_cost_avoided,
            "cost_of_intervention": intervention_cost,
            "cost_of_failure_avoided": cost_of_failure_avoided,
            "net_savings": net_savings,
            "roi_pct": roi_pct,
        },
        "vs_scheduled": {
            "cost_difference": cost_difference,
            "precision_advantage": (
                "Condition-based maintenance intervenes only when degradation "
                "is detected, avoiding unnecessary scheduled stops"
            ),
        },
        "30_day_projection": {
            "do_nothing_expected_cost": do_nothing_expected,
            "intervention_expected_cost": intervention_cost,
            "scheduled_expected_cost": scheduled_expected,
            "optimal_savings_vs_reactive": round(do_nothing_expected - intervention_cost, 2),
        },
    }


def format_impact_for_report(impact: dict[str, Any]) -> str:
    """
    Format impact metrics into a human-readable summary for the
    Manager Summary report.
    """
    vdn = impact.get("vs_do_nothing", {})
    proj = impact.get("30_day_projection", {})

    lines = [
        "BUSINESS IMPACT SUMMARY",
        "=" * 40,
        "",
        f"Risk Reduction:          {vdn.get('risk_reduction_pct', 0):.1f}%",
        f"Failure Prob. Reduced:   {vdn.get('failure_prob_reduction_pct', 0):.1f}%",
        f"Downtime Avoided:        {vdn.get('downtime_avoided_hours', 0):.1f} hours",
        f"Intervention Cost:       ${vdn.get('cost_of_intervention', 0):,.0f}",
        f"Failure Cost Avoided:    ${vdn.get('cost_of_failure_avoided', 0):,.0f}",
        f"Net Savings:             ${vdn.get('net_savings', 0):,.0f}",
        f"ROI:                     {vdn.get('roi_pct', 0):.0f}%",
        "",
        "30-DAY COST PROJECTION",
        "-" * 40,
        f"Do Nothing:              ${proj.get('do_nothing_expected_cost', 0):,.0f}",
        f"This Intervention:       ${proj.get('intervention_expected_cost', 0):,.0f}",
        f"Scheduled Maintenance:   ${proj.get('scheduled_expected_cost', 0):,.0f}",
        f"Savings vs Reactive:     ${proj.get('optimal_savings_vs_reactive', 0):,.0f}",
    ]

    return "\n".join(lines)


def _empty_impact() -> dict[str, Any]:
    """Return an empty impact structure when no plans exist."""
    return {
        "vs_do_nothing": {
            "risk_reduction_pct": 0.0,
            "failure_prob_reduction_pct": 0.0,
            "downtime_avoided_hours": 0.0,
            "downtime_cost_avoided": 0.0,
            "cost_of_intervention": 0.0,
            "cost_of_failure_avoided": 0.0,
            "net_savings": 0.0,
            "roi_pct": 0.0,
        },
        "vs_scheduled": {
            "cost_difference": 0.0,
            "precision_advantage": "N/A — no intervention plans available",
        },
        "30_day_projection": {
            "do_nothing_expected_cost": 0.0,
            "intervention_expected_cost": 0.0,
            "scheduled_expected_cost": 0.0,
            "optimal_savings_vs_reactive": 0.0,
        },
    }
