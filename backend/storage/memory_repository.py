"""In-memory repository implementation for backend incident workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache

from backend.contracts import IncidentRecord


class IncidentNotFoundError(KeyError):
    """Raised when an incident id does not exist in repository."""


class InMemoryIncidentRepository:
    """Simple in-memory incident store for MVP workflow execution."""

    def __init__(self) -> None:
        self._incidents: dict[str, IncidentRecord] = {}

    def create(self, incident_id: str, source: str, initial_stage: dict) -> IncidentRecord:
        now = datetime.now(timezone.utc)
        record = IncidentRecord(
            incident_id=incident_id,
            created_at=now,
            updated_at=now,
            source=source,
            stages={"ingest": initial_stage},
            timeline=[
                {
                    "stage": "ingest",
                    "timestamp": now.isoformat(),
                    "summary": "batch ingested",
                }
            ],
            confidence=initial_stage.get("confidence", 1.0),
            warnings=initial_stage.get("warnings", []),
        )
        self._incidents[incident_id] = record
        return record

    def get(self, incident_id: str) -> IncidentRecord:
        try:
            return self._incidents[incident_id]
        except KeyError as exc:
            raise IncidentNotFoundError(incident_id) from exc

    def save_stage(
        self,
        incident_id: str,
        stage_name: str,
        stage_payload: dict,
        confidence: float,
        warnings: list[str] | None = None,
    ) -> IncidentRecord:
        record = self.get(incident_id)
        now = datetime.now(timezone.utc)
        record.stages[stage_name] = stage_payload
        record.updated_at = now
        record.timeline.append(
            {
                "stage": stage_name,
                "timestamp": now.isoformat(),
                "summary": f"{stage_name} completed",
            }
        )
        if warnings:
            record.warnings.extend(warnings)
        record.confidence = min(record.confidence, confidence)
        self._incidents[incident_id] = record
        return record


@lru_cache(maxsize=1)
def get_incident_repository() -> InMemoryIncidentRepository:
    """Return singleton in-memory repository for app lifetime."""
    return InMemoryIncidentRepository()