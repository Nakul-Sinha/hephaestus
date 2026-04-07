"""
Sentinel Agent (Agent 3)

Responsibilities:
    - Run anomaly detection on quality-checked telemetry
    - Produce ranked list of at-risk assets
    - Filter: only assets above threshold pass downstream

Confidence logic: Based on anomaly score distance from threshold.
Next agent: prognostics_agent (if anomalies found) or reporter_agent (all clear).
"""

from __future__ import annotations

from typing import Any

from ml.aegis.agents import BaseAgent
from ml.aegis.data.schemas import AgentOutput
from ml.aegis.models.anomaly import AnomalyDetector, compute_rolling_features


class SentinelAgent(BaseAgent):

    name = "sentinel_agent"
    description = "Anomaly detection and regime change detection"

    def __init__(self, detector: AnomalyDetector | None = None):
        """
        Args:
            detector: Pre-trained AnomalyDetector instance.
                     If None, one must be provided in context['anomaly_detector'].
        """
        self._detector = detector

    def run(self, context: dict[str, Any]) -> AgentOutput:
        telemetry_df = context.get("telemetry_df")
        if telemetry_df is None or len(telemetry_df) == 0:
            return self._build_output(
                payload={"status": "no_data"},
                confidence=0.0,
                errors=["No telemetry data for anomaly detection"],
                next_agent="reporter_agent",
            )

        detector = self._detector or context.get("anomaly_detector")
        if detector is None:
            return self._build_output(
                payload={"status": "no_model"},
                confidence=0.0,
                errors=["No trained AnomalyDetector available"],
                next_agent="reporter_agent",
            )

        evidence: list[str] = []
        errors: list[str] = []

        # --- Score all assets ---
        try:
            scores_df = detector.score(telemetry_df)
        except Exception as e:
            return self._build_output(
                payload={"status": "scoring_failed"},
                confidence=0.0,
                errors=[f"Anomaly scoring failed: {e}"],
                next_agent="reporter_agent",
            )

        total_assets = len(scores_df)
        at_risk = scores_df[scores_df["is_anomalous"]].copy()
        at_risk = at_risk.sort_values("anomaly_score", ascending=False)
        n_at_risk = len(at_risk)

        evidence.append(f"Scored {total_assets} assets")
        evidence.append(f"{n_at_risk} assets flagged as anomalous")

        if n_at_risk > 0:
            top_asset = at_risk.iloc[0]
            evidence.append(
                f"Highest risk: {top_asset['asset_id']} "
                f"(score={top_asset['anomaly_score']:.3f})"
            )

        # Apply quality confidence modifier
        quality_modifier = context.get("quality_confidence_modifier", 1.0)

        # --- Confidence ---
        if n_at_risk == 0:
            confidence = 0.90 * quality_modifier
            next_agent = "reporter_agent"
        else:
            avg_score = float(at_risk["anomaly_score"].mean())
            confidence = min(0.95, 0.5 + avg_score * 0.4) * quality_modifier
            next_agent = "prognostics_agent"

        # Store in context
        context["anomaly_scores"] = scores_df
        context["at_risk_assets"] = at_risk
        context["at_risk_asset_ids"] = set(at_risk["asset_id"].tolist())

        # Convert for payload serialization
        at_risk_summary = []
        for _, row in at_risk.head(20).iterrows():
            at_risk_summary.append({
                "asset_id": row["asset_id"],
                "asset_type": row.get("asset_type", ""),
                "anomaly_score": round(float(row["anomaly_score"]), 4),
                "has_point_anomaly": bool(row.get("has_point_anomaly", False)),
                "top_sensors": row.get("top_contributing_sensors", []),
            })

        return self._build_output(
            payload={
                "status": "completed",
                "total_assets_scored": total_assets,
                "assets_at_risk": n_at_risk,
                "at_risk_summary": at_risk_summary,
                "mean_anomaly_score": round(float(scores_df["anomaly_score"].mean()), 4),
                "max_anomaly_score": round(float(scores_df["anomaly_score"].max()), 4),
            },
            confidence=round(confidence, 4),
            assumptions=[
                "Anomaly threshold calibrated on healthy historical data",
                "Z-score guard threshold set at 4.0 standard deviations",
            ],
            evidence=evidence,
            errors=errors if errors else None,
            next_agent=next_agent,
        )
