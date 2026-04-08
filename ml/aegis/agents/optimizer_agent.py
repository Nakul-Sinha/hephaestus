"""Optimizer agent runtime implementation."""

from __future__ import annotations

from ml.aegis.data.schemas import AgentOutput


class OptimizerAgent:
    """Emit optimization-stage diagnostics for orchestration traces."""

    def run(self, trace_id: str, optimize_payload: dict) -> AgentOutput:
        ranked = list(optimize_payload.get("ranked_plans", []))
        feasible_count = sum(1 for plan in ranked if bool(plan.get("is_feasible", False)))
        confidence = 0.82 if feasible_count > 0 else 0.58

        return AgentOutput(
            input_context_id=trace_id,
            agent_name="OptimizerAgent",
            output_payload={
                "recommended_plan_id": optimize_payload.get("recommended_plan_id"),
                "candidate_count": len(ranked),
                "feasible_count": feasible_count,
            },
            confidence_score=confidence,
            assumptions=["optimizer-stage ranking remains the source of truth for feasible recommendations"],
            evidence_refs=["optimize-stage", "ml.aegis.planning.optimizer.PlanOptimizer"],
            errors=[],
            next_recommended_agent="SimulationAgent",
            execution_time_ms=0.0,
        )
