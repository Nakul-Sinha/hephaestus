"""Governance checks for backend stage outputs."""

from __future__ import annotations

from datetime import datetime, timezone

from backend.config import BackendSettings


class GovernanceService:
    """Evaluates confidence and warning signals for each stage result."""

    def __init__(self, settings: BackendSettings) -> None:
        self.settings = settings

    def evaluate(
        self,
        incident_id: str,
        stage: str,
        confidence: float,
        warnings: list[str],
    ) -> dict:
        """Return governance verdict for a stage."""
        verdict = "approved"
        reasons: list[str] = []

        if confidence < self.settings.confidence_floor:
            verdict = "needs_human_review"
            reasons.append("confidence below configured floor")

        if len(warnings) >= 3:
            verdict = "needs_human_review"
            reasons.append("warning count exceeds auto-approval threshold")

        return {
            "incident_id": incident_id,
            "stage": stage,
            "verdict": verdict,
            "confidence": confidence,
            "reasons": reasons,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }