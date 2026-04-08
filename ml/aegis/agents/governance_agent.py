"""Governance agent runtime implementation."""

from __future__ import annotations

from ml.aegis.data.schemas import AgentOutput


class GovernanceAgent:
    """Apply final policy gate to orchestrator outputs."""

    def __init__(self, confidence_floor: float = 0.6) -> None:
        self.confidence_floor = confidence_floor

    def run(self, trace_id: str, stage_confidences: list[float], warnings: list[str]) -> AgentOutput:
        min_confidence = min(stage_confidences) if stage_confidences else 0.0
        verdict = "approved"
        reasons: list[str] = []
        if min_confidence < self.confidence_floor:
            verdict = "needs_human_review"
            reasons.append("confidence below floor")
        if len(warnings) >= 3:
            verdict = "needs_human_review"
            reasons.append("warning volume exceeds threshold")

        confidence = max(0.35, min(0.95, min_confidence if stage_confidences else 0.35))
        return AgentOutput(
            input_context_id=trace_id,
            agent_name="GovernanceAgent",
            output_payload={
                "verdict": verdict,
                "reasons": reasons,
                "confidence_floor": self.confidence_floor,
            },
            confidence_score=confidence,
            assumptions=["minimum stage confidence bounds end-to-end reliability"],
            evidence_refs=["governance-policy"],
            errors=[],
            next_recommended_agent=None,
            execution_time_ms=0.0,
        )
