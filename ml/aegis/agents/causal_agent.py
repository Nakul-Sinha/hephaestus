"""Causal agent runtime implementation."""

from __future__ import annotations

from ml.aegis.data.schemas import AgentOutput


class CausalAgent:
    """Build a small root-cause hypothesis output from plan-stage context."""

    def run(self, trace_id: str, plan_payload: dict) -> AgentOutput:
        root_cause = str(plan_payload.get("root_cause", "unknown"))
        root_cause_confidence = float(plan_payload.get("root_cause_confidence", 0.0))
        contributors = [
            entry.get("feature")
            for entry in plan_payload.get("explainability", {}).get("top_contributors", [])
            if isinstance(entry, dict) and entry.get("feature")
        ]

        payload = {
            "primary_hypothesis": root_cause,
            "confidence": round(root_cause_confidence, 4),
            "supporting_features": contributors[:5],
        }
        confidence = max(0.45, min(0.95, root_cause_confidence))

        return AgentOutput(
            input_context_id=trace_id,
            agent_name="CausalAgent",
            output_payload=payload,
            confidence_score=confidence,
            assumptions=["top explainability contributors are valid evidence for causal hypothesis ranking"],
            evidence_refs=["plan-stage", "ml.aegis.models.explainability.ModelExplainer"],
            errors=[],
            next_recommended_agent="PlannerAgent",
            execution_time_ms=0.0,
        )
