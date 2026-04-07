"""
Simulation Agent (Agent 8)

Responsibilities:
    - Run Monte Carlo simulations for each feasible plan + "do nothing" baseline
    - Project risk, cost, and downtime over a 30-day horizon
    - Compute confidence intervals (5th to 95th percentiles)

Next agent: reporter_agent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml.aegis.agents import BaseAgent
from ml.aegis.data.schemas import AgentOutput, ComparisonResult, SimulationResult, InterventionPlan


# ---------------------------------------------------------------------------
# Basic Monte Carlo Simulation (inline until Phase 3 is implemented)
# ---------------------------------------------------------------------------

def simulate_plan(
    plan: InterventionPlan,
    current_risk: float,
    degradation_rate: float = 0.05,
    n_iterations: int = 1000,
    horizon_days: int = 30,
) -> SimulationResult:
    """Run Monte Carlo simulation for a single plan."""
    # Base arrays for risk trajectories
    np.random.seed(42)  # For reproducible demo
    
    # Uncertainty in degradation (more uncertain further into future)
    daily_noise = np.random.normal(0, 0.02, size=(n_iterations, horizon_days))
    
    # Calculate baseline risk trajectory
    risk_increment = np.clip(np.random.normal(degradation_rate, 0.01, size=(n_iterations, 1)), 0.01, 0.2)
    daily_increases = risk_increment * np.ones((n_iterations, horizon_days)) + daily_noise
    cumulative_increase = np.cumsum(daily_increases, axis=1)
    
    # Base risk trajectory without intervention
    base_trajectory = cumulative_increase + current_risk
    
    # Apply intervention effects
    if plan.recommended_action == "DO_NOTHING":
        impacted_trajectory = base_trajectory
        cost_dist = np.zeros(n_iterations)
        downtime_dist = np.zeros(n_iterations)
    else:
        # Determine day of intervention based on maintenance window
        intervention_day = 1 if plan.maintenance_window == "immediate" else (
            2 if plan.maintenance_window == "next_business_day" else 7
        )
        
        impacted_trajectory = base_trajectory.copy()
        
        # Risk drops after intervention
        risk_reduction = max(0.0, min(1.0, np.random.normal(plan.predicted_risk_reduction, 0.05)))
        for i in range(horizon_days):
            if i >= intervention_day:
                impacted_trajectory[:, i] = impacted_trajectory[:, i] * (1.0 - risk_reduction)
                
        # Sample cost and downtime
        cost_variance = plan.estimated_cost * 0.15
        cost_dist = np.random.normal(plan.estimated_cost, cost_variance, n_iterations)
        cost_dist = np.clip(cost_dist, plan.estimated_cost * 0.5, None)
        
        downtime_variance = plan.expected_downtime_minutes * 0.2
        downtime_dist = np.random.normal(plan.expected_downtime_minutes, downtime_variance, n_iterations)
        downtime_dist = np.clip(downtime_dist, plan.expected_downtime_minutes * 0.5, None)

    # Ensure risk stays between 0 and 1
    impacted_trajectory = np.clip(impacted_trajectory, 0.0, 1.0)
    
    # Compute percentiles
    mean_trajectory = np.mean(impacted_trajectory, axis=0).tolist()
    p5_trajectory = np.percentile(impacted_trajectory, 5, axis=0).tolist()
    p95_trajectory = np.percentile(impacted_trajectory, 95, axis=0).tolist()
    
    # Calculate probability of failure (risk > 0.9)
    failure_mask = impacted_trajectory > 0.90
    failure_prob = float(np.mean(np.any(failure_mask, axis=1)))
    
    return SimulationResult(
        plan_id=plan.plan_id,
        daily_risk_mean=mean_trajectory,
        daily_risk_p5=p5_trajectory,
        daily_risk_p95=p95_trajectory,
        total_expected_cost=float(np.mean(cost_dist)),
        expected_downtime_minutes=float(np.mean(downtime_dist)),
        probability_of_failure=failure_prob,
    )


# ---------------------------------------------------------------------------
# Simulation Agent
# ---------------------------------------------------------------------------

class SimulationAgent(BaseAgent):

    name = "simulation_agent"
    description = "Runs Monte Carlo projections of risk under various plans"

    def run(self, context: dict[str, Any]) -> AgentOutput:
        optimized_plans = context.get("optimized_plans", [])
        prediction_summaries = context.get("prediction_summaries", [])
        
        if not optimized_plans or not prediction_summaries:
            return self._build_output(
                payload={"status": "no_data_to_simulate"},
                confidence=0.0,
                errors=["Missing plans or predictions for simulation"],
                next_agent="reporter_agent"
            )

        evidence: list[str] = []
        errors: list[str] = []
        comparisons: list[ComparisonResult] = []

        # Find the max probability from prediction_summaries as starting risk
        current_risk_max = max((p["failure_probability"] for p in prediction_summaries), default=0.5)

        # Only simulate feasible plans + a "do nothing" baseline
        feasible_plans = [p for p in optimized_plans if p.is_feasible]
        
        if not feasible_plans:
            # Nothing to simulate if no plans are feasible
            return self._build_output(
                payload={"status": "no_feasible_plans"},
                confidence=0.0,
                errors=["No feasible plans found for simulation"],
                next_agent="reporter_agent"
            )

        # Asset IDs being targeted
        asset_ids = list(set(p.plan_id.split("-")[0] for p in feasible_plans))
        
        for asset_id in asset_ids:
            # Find plans for this asset
            asset_plans = [p for p in feasible_plans if p.plan_id.startswith(asset_id)]
            
            # Find current risk for this asset
            asset_pred = next((p for p in prediction_summaries if p["asset_id"] == asset_id), None)
            current_risk = asset_pred["failure_probability"] if asset_pred else 0.5
            
            # 1. Simulate "Do Nothing" baseline
            do_nothing_plan = InterventionPlan(
                plan_id=f"{asset_id}-DO_NOTHING",
                recommended_action="DO_NOTHING",
                required_parts=[],
                required_skills=[],
                estimated_duration_minutes=0,
                maintenance_window="immediate",
                predicted_risk_reduction=0.0,
                estimated_cost=0.0,
                expected_downtime_minutes=0,
                confidence=1.0,
                assumptions=["Baseline scenario"],
                rollback_plan="N/A",
                is_feasible=True
            )
            
            do_nothing_sim = simulate_plan(do_nothing_plan, current_risk)
            
            # 2. Simulate feasible plans
            sim_results = {"DO_NOTHING": do_nothing_sim}
            for plan in asset_plans:
                sim = simulate_plan(plan, current_risk)
                sim_results[plan.plan_id] = sim
                
            evidence.append(f"Simulated {len(asset_plans)} plans + baseline for {asset_id}")
            
            # 3. Create ComparisonResult
            comparisons.append(ComparisonResult(
                asset_id=asset_id,
                baseline_trajectory=do_nothing_sim,
                plan_trajectories=[sim for pid, sim in sim_results.items() if pid != "DO_NOTHING"],
                best_plan_id=asset_plans[0].plan_id if asset_plans else "DO_NOTHING",
                impact_summary={
                    "downtime_avoided_hours": 0.0, # Computed fully in Phase 3
                    "cost_saved": 0.0,
                    "risk_reduction": float(do_nothing_sim.probability_of_failure - min(s.probability_of_failure for s in sim_results.values()))
                }
            ))

        context["simulations"] = comparisons
        
        quality_modifier = context.get("quality_confidence_modifier", 1.0)
        confidence = 0.90 * quality_modifier

        return self._build_output(
            payload={
                "status": "completed",
                "assets_simulated": len(comparisons),
                "total_simulations": sum(1 + len(c.plan_trajectories) for c in comparisons),
                "comparisons": [c.model_dump() for c in comparisons]
            },
            confidence=round(confidence, 4),
            assumptions=[
                "Monte Carlo uses 1000 iterations over 30 days",
                "Degradation rate assumes constant mean increment with normal noise",
                "Repair effectiveness has 5% standard deviation variance"
            ],
            evidence=evidence,
            errors=errors if errors else None,
            next_agent="reporter_agent",
        )
