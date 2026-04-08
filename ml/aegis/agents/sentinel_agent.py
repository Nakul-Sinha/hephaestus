"""Sentinel agent runtime implementation."""

from __future__ import annotations

from ml.aegis.data.schemas import AgentOutput


class SentinelAgent:
    """Emit anomaly interpretation from risk-stage payload."""

    def run(self, trace_id: str, risk_payload: dict) -> AgentOutput:
        anomaly_score = float(risk_payload.get("anomaly_score", 0.0))
        signal = "normal"
        if anomaly_score >= 0.8:
            signal = "critical"
        elif anomaly_score >= 0.5:
            signal = "elevated"

        confidence = max(0.5, min(0.95, 0.65 + anomaly_score * 0.3))
        return AgentOutput(
            input_context_id=trace_id,
            agent_name="SentinelAgent",
            output_payload={
                "asset_id": risk_payload.get("asset_id"),
                "anomaly_score": round(anomaly_score, 4),
                "signal": signal,
            },
            confidence_score=confidence,
            assumptions=["anomaly score bands map to operational severity tiers"],
            evidence_refs=["risk-stage", "ml.aegis.models.anomaly.AnomalyDetector"],
            errors=[],
            next_recommended_agent="PrognosticsAgent",
            execution_time_ms=0.0,
        )
