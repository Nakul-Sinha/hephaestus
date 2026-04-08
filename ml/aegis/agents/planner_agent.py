"""Planner agent runtime implementation."""

from __future__ import annotations

from ml.aegis.data.schemas import AgentOutput


class PlannerAgent:
  """Summarize planning-stage candidates for downstream optimization."""

  def run(self, trace_id: str, plan_payload: dict) -> AgentOutput:
    plans = list(plan_payload.get("plans", []))
    if plans:
      best = max(plans, key=lambda item: float(item.get("predicted_risk_reduction", 0.0)))
      best_plan_id = str(best.get("plan_id", "unknown"))
    else:
      best_plan_id = "unknown"

    confidence = 0.78 if plans else 0.5
    return AgentOutput(
      input_context_id=trace_id,
      agent_name="PlannerAgent",
      output_payload={
        "candidate_count": len(plans),
        "best_risk_reduction_plan_id": best_plan_id,
      },
      confidence_score=confidence,
      assumptions=["candidate plans use consistent schema across planner implementations"],
      evidence_refs=["plan-stage"],
      errors=[],
      next_recommended_agent="OptimizerAgent",
      execution_time_ms=0.0,
    )
