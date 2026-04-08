"""Prognostics agent runtime implementation."""

from __future__ import annotations

from ml.aegis.data.schemas import AgentOutput


class PrognosticsAgent:
    """Normalize risk predictions into prognostics handoff output."""

    def run(self, trace_id: str, risk_payload: dict) -> AgentOutput:
        probability = float(risk_payload.get("failure_probability", 0.0))
        horizon = int(risk_payload.get("failure_horizon_hours", risk_payload.get("lookahead_hours", 0)))
        confidence = max(0.4, min(0.95, 0.55 + probability * 0.35))

        return AgentOutput(
            input_context_id=trace_id,
            agent_name="PrognosticsAgent",
            output_payload={
                "asset_id": risk_payload.get("asset_id"),
                "failure_probability": round(probability, 4),
                "failure_horizon_hours": horizon,
                "risk_band": risk_payload.get("risk_band", "unknown"),
            },
            confidence_score=confidence,
            assumptions=["current feature window is representative for short-horizon forecasting"],
            evidence_refs=["risk-stage", "ml.aegis.models.failure_risk.FailureRiskModel"],
            errors=[],
            next_recommended_agent="CausalAgent",
            execution_time_ms=0.0,
        )
