"""Repository implementations and contracts for incident persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from backend.contracts import IncidentRecord
from backend.storage.db import ensure_incident_schema, postgres_connection
from backend.storage.memory_repository import IncidentNotFoundError


class IncidentRepository(Protocol):
    """Persistence contract required by service and adapter layers."""

    def clear(self) -> None:
        ...

    def create(self, incident_id: str, source: str, initial_stage: dict) -> IncidentRecord:
        ...

    def get(self, incident_id: str) -> IncidentRecord:
        ...

    def save_stage(
        self,
        incident_id: str,
        stage_name: str,
        stage_payload: dict,
        confidence: float,
        warnings: list[str] | None = None,
    ) -> IncidentRecord:
        ...

    def add_governance_event(self, incident_id: str, governance_event: dict) -> IncidentRecord:
        ...


class PostgresIncidentRepository:
    """Postgres-backed incident repository storing one JSON record per incident."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        ensure_incident_schema(dsn)

    def clear(self) -> None:
        with postgres_connection(self._dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM incident_records")
            connection.commit()

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
            confidence_trail=[
                {
                    "stage": "ingest",
                    "confidence": initial_stage.get("confidence", 1.0),
                    "timestamp": now.isoformat(),
                }
            ],
            confidence=initial_stage.get("confidence", 1.0),
            warnings=list(initial_stage.get("warnings", [])),
        )
        self._upsert(record)
        return record

    def get(self, incident_id: str) -> IncidentRecord:
        query = "SELECT payload FROM incident_records WHERE incident_id = %s"
        with postgres_connection(self._dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (incident_id,))
                row = cursor.fetchone()

        if not row:
            raise IncidentNotFoundError(incident_id)

        payload = row[0]
        return IncidentRecord.model_validate(payload)

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
        record.confidence_trail.append(
            {
                "stage": stage_name,
                "confidence": confidence,
                "timestamp": now.isoformat(),
            }
        )
        if warnings:
            record.warnings.extend(warnings)
        record.confidence = min(record.confidence, confidence)
        self._upsert(record)
        return record

    def add_governance_event(self, incident_id: str, governance_event: dict) -> IncidentRecord:
        record = self.get(incident_id)
        record.governance_trail.append(governance_event)
        self._upsert(record)
        return record

    def _upsert(self, record: IncidentRecord) -> None:
        payload = record.model_dump(mode="json")
        sql = """
        INSERT INTO incident_records (incident_id, payload, updated_at)
        VALUES (%s, %s::jsonb, NOW())
        ON CONFLICT (incident_id)
        DO UPDATE SET payload = EXCLUDED.payload, updated_at = NOW()
        """

        with postgres_connection(self._dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, (record.incident_id, payload))
            connection.commit()
