"""Intake agent runtime implementation."""

from __future__ import annotations

from ml.aegis.data.schemas import AgentOutput


class IntakeAgent:
    """Summarize ingestion context and emit standardized handoff payload."""

    def run(self, trace_id: str, ingest_payload: dict) -> AgentOutput:
        telemetry_rows = int(ingest_payload.get("telemetry_rows", 0))
        confidence = 1.0 if telemetry_rows > 0 else 0.7

        return AgentOutput(
            input_context_id=trace_id,
            agent_name="IntakeAgent",
            output_payload={
                "source": ingest_payload.get("source", "unknown"),
                "telemetry_rows": telemetry_rows,
                "dataset_ready": bool(ingest_payload.get("dataset_ready", False)),
                "data_mode": ingest_payload.get("data_mode", "unknown"),
            },
            confidence_score=confidence,
            assumptions=["ingest payload has already passed endpoint schema validation"],
            evidence_refs=["ingest-stage"],
            errors=[],
            next_recommended_agent="QualityAgent",
            execution_time_ms=0.0,
        )
