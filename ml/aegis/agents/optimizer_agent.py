"""
Optimizer Agent (Agent 7)

Responsibilities:
    - Score each plan using multi-objective function
    - Check hard constraints (budget, crew, parts, blackout windows)
    - Eliminate infeasible plans with reasons
    - Rank remaining plans and mark top as RECOMMENDED

Does NOT use an LLM. Uses mathematical optimization only.
Next agent: simulation_agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ml.aegis.agents import BaseAgent
from ml.aegis.data.schemas import AgentOutput, InterventionPlan


# ---------------------------------------------------------------------------
# Constraints & Objective (inline — Phase 3 modules will formalize these)
# ---------------------------------------------------------------------------

@dataclass
class PlanningConstraints:
    """Operational constraints for plan feasibility checks."""
    budget_ceiling: float = 50000.0
    available_crew: dict[str, int] = field(default_factory=lambda: {
        "mechanical_technician": 3,
        "electrician": 2,
        "vibration_analyst": 1,
        "thermal_engineer": 1,
        "pressure_specialist": 1,
        "process_engineer": 2,
        "motor_specialist": 1,
        "bearing_specialist": 1,
    })
    spare_parts_inventory: list[str] = field(default_factory=lambda: [
        "SKF_bearing_6205", "alignment_shims", "lubricant_kit",
        "high_grade_lubricant", "portable_vibration_sensor",
        "mechanical_seal_kit", "gasket_set", "pressure_test_kit",
        "industrial_sealant", "pressure_gauge",
        "coolant_fluid", "thermal_paste", "heat_exchanger_gasket", "flush_kit",
        "circuit_breaker", "fuse_kit", "motor_winding_kit",
        "general_repair_kit", "minor_repair_kit",
        "calibration_tool", "wiring_harness",
    ])
    blackout_windows: list[tuple[datetime, datetime]] = field(default_factory=list)
    max_concurrent_interventions: int = 5
    auto_approval_cost_limit: float = 10000.0


@dataclass
class PolicyWeights:
    """Weights for the multi-objective scoring function."""
    w_risk_reduction: float = 0.40
    w_cost: float = 0.25
    w_downtime: float = 0.20
    w_sla_compliance: float = 0.15


def check_plan_feasibility(
    plan: InterventionPlan,
    constraints: PlanningConstraints,
) -> tuple[bool, list[str]]:
    """Check if a plan is feasible under constraints."""
    violations: list[str] = []

    # Budget check
    if plan.estimated_cost > constraints.budget_ceiling:
        violations.append(
            f"Cost ${plan.estimated_cost:,.0f} exceeds budget ceiling "
            f"${constraints.budget_ceiling:,.0f}"
        )

    # Crew skill check
    for skill in plan.required_skills:
        available = constraints.available_crew.get(skill, 0)
        if available <= 0:
            violations.append(f"Required skill '{skill}' not available in crew roster")

    # Parts inventory check
    for part in plan.required_parts:
        if part not in constraints.spare_parts_inventory:
            violations.append(f"Required part '{part}' not in spare parts inventory")

    return len(violations) == 0, violations


def score_plan(
    plan: InterventionPlan,
    weights: PolicyWeights,
    max_cost: float = 10000.0,
    max_downtime: float = 480.0,
) -> float:
    """
    Score a plan using multi-objective weighted function.
    Higher score = better plan. Returns value in 0-1 range.
    """
    risk_score = plan.predicted_risk_reduction  # Already 0-1

    cost_normalized = 1.0 - min(plan.estimated_cost / max_cost, 1.0)
    downtime_normalized = 1.0 - min(plan.expected_downtime_minutes / max_downtime, 1.0)

    # SLA: plans with "immediate" window get higher SLA score
    sla_map = {"immediate": 1.0, "next_business_day": 0.7, "next_planned_shutdown": 0.4}
    sla_score = sla_map.get(plan.maintenance_window, 0.5)

    score = (
        weights.w_risk_reduction * risk_score
        + weights.w_cost * cost_normalized
        + weights.w_downtime * downtime_normalized
        + weights.w_sla_compliance * sla_score
    )

    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Optimizer Agent
# ---------------------------------------------------------------------------

class OptimizerAgent(BaseAgent):

    name = "optimizer_agent"
    description = "Scores, constrains, and ranks intervention plans"

    def __init__(
        self,
        constraints: PlanningConstraints | None = None,
        weights: PolicyWeights | None = None,
    ):
        self._constraints = constraints or PlanningConstraints()
        self._weights = weights or PolicyWeights()

    def run(self, context: dict[str, Any]) -> AgentOutput:
        plans: list[InterventionPlan] = context.get("plans", [])

        if not plans:
            return self._build_output(
                payload={"status": "no_plans"},
                confidence=0.0,
                errors=["No intervention plans to optimize"],
                next_agent="reporter_agent",
            )

        evidence: list[str] = []
        errors: list[str] = []

        constraints = context.get("constraints", self._constraints)
        weights = context.get("weights", self._weights)

        # Score and check feasibility for each plan
        feasible_plans: list[InterventionPlan] = []
        infeasible_plans: list[InterventionPlan] = []

        for plan in plans:
            is_feasible, violations = check_plan_feasibility(plan, constraints)

            if not is_feasible:
                plan.is_feasible = False
                plan.infeasibility_reasons = violations
                plan.optimizer_score = 0.0
                infeasible_plans.append(plan)
                evidence.append(f"Plan {plan.plan_id}: INFEASIBLE — {'; '.join(violations)}")
            else:
                plan.is_feasible = True
                plan.optimizer_score = score_plan(plan, weights)
                feasible_plans.append(plan)

        # Rank feasible plans by score (descending)
        feasible_plans.sort(key=lambda p: p.optimizer_score or 0, reverse=True)

        # Mark the top plan as RECOMMENDED
        recommended_plan_id = None
        if feasible_plans:
            recommended_plan_id = feasible_plans[0].plan_id
            evidence.append(
                f"RECOMMENDED: {feasible_plans[0].plan_id} "
                f"(score={feasible_plans[0].optimizer_score:.3f}, "
                f"cost=${feasible_plans[0].estimated_cost:,.0f}, "
                f"risk_reduction={feasible_plans[0].predicted_risk_reduction:.0%})"
            )

        evidence.append(f"Feasible: {len(feasible_plans)}, Infeasible: {len(infeasible_plans)}")

        # Confidence based on score dominance
        if len(feasible_plans) >= 2:
            score_gap = (feasible_plans[0].optimizer_score or 0) - (feasible_plans[1].optimizer_score or 0)
            if score_gap > 0.15:
                confidence = 0.90
            elif score_gap > 0.05:
                confidence = 0.75
            else:
                confidence = 0.60
                errors.append("Top plans are closely scored — either is reasonable")
        elif len(feasible_plans) == 1:
            confidence = 0.85
        else:
            confidence = 0.20
            errors.append("No feasible plans found — all violate constraints")

        quality_modifier = context.get("quality_confidence_modifier", 1.0)
        confidence *= quality_modifier

        # Combine and store
        optimized_plans = feasible_plans + infeasible_plans
        context["optimized_plans"] = optimized_plans
        context["recommended_plan_id"] = recommended_plan_id

        plans_payload = [
            {
                "plan_id": p.plan_id,
                "recommended_action": p.recommended_action,
                "estimated_cost": p.estimated_cost,
                "predicted_risk_reduction": p.predicted_risk_reduction,
                "optimizer_score": p.optimizer_score,
                "is_feasible": p.is_feasible,
                "infeasibility_reasons": p.infeasibility_reasons,
                "is_recommended": p.plan_id == recommended_plan_id,
            }
            for p in optimized_plans
        ]

        return self._build_output(
            payload={
                "status": "completed",
                "total_plans": len(plans),
                "feasible_count": len(feasible_plans),
                "infeasible_count": len(infeasible_plans),
                "recommended_plan_id": recommended_plan_id,
                "plans": plans_payload,
                "weights_used": {
                    "risk_reduction": weights.w_risk_reduction,
                    "cost": weights.w_cost,
                    "downtime": weights.w_downtime,
                    "sla_compliance": weights.w_sla_compliance,
                },
            },
            confidence=round(confidence, 4),
            assumptions=[
                f"Budget ceiling: ${constraints.budget_ceiling:,.0f}",
                f"Max concurrent interventions: {constraints.max_concurrent_interventions}",
                "Cost/downtime normalization based on $10K / 8hr baselines",
            ],
            evidence=evidence,
            errors=errors if errors else None,
            next_agent="simulation_agent",
        )
