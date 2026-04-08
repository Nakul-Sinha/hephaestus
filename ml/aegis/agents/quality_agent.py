"""Quality agent runtime implementation."""

from __future__ import annotations

from ml.aegis.data.schemas import AgentOutput


class QualityAgent:
    """Run lightweight quality checks on ingest metadata."""

    def run(self, trace_id: str, ingest_payload: dict) -> AgentOutput:
        telemetry_rows = int(ingest_payload.get("telemetry_rows", 0))
        event_rows = int(ingest_payload.get("event_rows", 0))
        maintenance_rows = int(ingest_payload.get("maintenance_rows", 0))

        flags: list[str] = []
        if telemetry_rows <= 0:
            flags.append("telemetry_missing")
        if event_rows == 0:
            flags.append("event_context_sparse")
        if maintenance_rows == 0:
            flags.append("maintenance_history_sparse")

        confidence = 0.9 if not flags else max(0.5, 0.9 - 0.1 * len(flags))
        return AgentOutput(
            input_context_id=trace_id,
            agent_name="QualityAgent",
            output_payload={
                "quality_flags": flags,
                "is_quality_ok": len(flags) == 0,
            },
            confidence_score=confidence,
            assumptions=["row-count sparsity is a proxy for data quality in MVP runtime"],
            evidence_refs=["ingest-stage"],
            errors=[],
            next_recommended_agent="SentinelAgent",
            execution_time_ms=0.0,
        )
