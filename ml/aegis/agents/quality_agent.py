"""
Quality Agent (Agent 2)

Responsibilities:
    - Run all data quality validators on ingested telemetry
    - Flag unreliable sensor channels
    - Produce quality summary with confidence modifier
    - Route to reporter_agent if data is too poor for prediction

Confidence logic: Based on overall data quality verdict.
Next agent: sentinel_agent (normal) or reporter_agent (data crisis).
"""

from __future__ import annotations

from typing import Any

from ml.aegis.agents import BaseAgent
from ml.aegis.data.schemas import AgentOutput
from ml.aegis.data.validators import (
    check_missingness,
    check_outlier_bursts,
    check_sensor_freeze,
    check_timestamp_integrity,
    compute_quality_summary,
)


class QualityAgent(BaseAgent):

    name = "quality_agent"
    description = "Runs data quality checks and drift diagnostics"

    def run(self, context: dict[str, Any]) -> AgentOutput:
        """
        Run all quality validators on the ingested telemetry.

        Expected context keys:
            - telemetry_df: pd.DataFrame (from Intake Agent)
        """
        telemetry_df = context.get("telemetry_df")

        if telemetry_df is None or len(telemetry_df) == 0:
            return self._build_output(
                payload={"status": "no_data"},
                confidence=0.0,
                errors=["No telemetry data available for quality checks"],
                next_agent="reporter_agent",
            )

        errors: list[str] = []
        evidence: list[str] = []

        # --- Run all validators ---

        # 1. Missingness
        missingness = check_missingness(telemetry_df)
        evidence.append(
            f"Missingness: {missingness['overall_missing_pct']}% overall → {missingness['verdict']}"
        )

        # 2. Sensor freeze
        freeze_incidents = check_sensor_freeze(telemetry_df)
        evidence.append(f"Frozen sensors: {len(freeze_incidents)} incidents detected")

        # 3. Outlier bursts
        outlier_results = check_outlier_bursts(telemetry_df)
        suspect_outliers = sum(1 for o in outlier_results if o["verdict"] != "ok")
        evidence.append(f"Outlier check: {suspect_outliers} suspect/unreliable channels")

        # 4. Timestamp integrity
        timestamp_check = check_timestamp_integrity(telemetry_df)
        evidence.append(
            f"Timestamps: {timestamp_check['duplicates']} duplicates, "
            f"{len(timestamp_check['gap_incidents'])} gaps → {timestamp_check['verdict']}"
        )

        # --- Aggregate summary ---
        quality_summary = compute_quality_summary(
            missingness=missingness,
            freeze_incidents=freeze_incidents,
            outlier_results=outlier_results,
            timestamp_check=timestamp_check,
        )

        overall_verdict = quality_summary["overall_verdict"]
        confidence_modifier = quality_summary["confidence_modifier"]

        # --- Decide routing ---
        if overall_verdict == "unreliable":
            next_agent = "reporter_agent"
            confidence = 0.3
            errors.append(
                "Data quality is UNRELIABLE — skipping prediction pipeline. "
                "Routing directly to Reporter Agent with quality crisis report."
            )
        elif overall_verdict == "suspect":
            next_agent = "sentinel_agent"
            confidence = 0.65
            errors.append(
                "Data quality is SUSPECT — predictions will have reduced confidence."
            )
        else:
            next_agent = "sentinel_agent"
            confidence = 0.90

        # Store quality results in context for downstream agents
        context["quality_summary"] = quality_summary
        context["quality_confidence_modifier"] = confidence_modifier
        context["freeze_incidents"] = freeze_incidents
        context["outlier_results"] = outlier_results

        return self._build_output(
            payload={
                "status": "quality_checked",
                "overall_verdict": overall_verdict,
                "confidence_modifier": confidence_modifier,
                "missingness_verdict": missingness["verdict"],
                "overall_missing_pct": missingness["overall_missing_pct"],
                "frozen_sensor_count": quality_summary["frozen_sensor_count"],
                "suspect_sensor_count": quality_summary["suspect_sensor_count"],
                "unreliable_sensor_count": quality_summary["unreliable_sensor_count"],
                "timestamp_duplicates": timestamp_check["duplicates"],
                "timestamp_gaps": len(timestamp_check["gap_incidents"]),
                "checks": quality_summary["checks"],
            },
            confidence=confidence,
            assumptions=[
                "Quality thresholds are calibrated for industrial sensor data",
                "Sensor freeze detection threshold is 2 hours",
                "Outlier z-score threshold is 4.0 (robust MAD-based)",
            ],
            evidence=evidence,
            errors=errors if errors else None,
            next_agent=next_agent,
        )
