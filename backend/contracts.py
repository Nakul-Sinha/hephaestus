"""Request and internal state contracts for backend workflow endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class IngestBatchRequest(BaseModel):
    """Payload accepted by ingest endpoint."""

    source: str = Field(default="synthetic")
    telemetry_rows: int = Field(default=0, ge=0)
    event_rows: int = Field(default=0, ge=0)
    maintenance_rows: int = Field(default=0, ge=0)
    notes: str = Field(default="")
    asset_path: str | None = Field(default=None)
    telemetry_path: str | None = Field(default=None)
    event_path: str | None = Field(default=None)
    maintenance_path: str | None = Field(default=None)
    failure_path: str | None = Field(default=None)


class RiskAnalyzeRequest(BaseModel):
    """Payload accepted by risk analyze endpoint."""

    incident_id: str
    lookahead_hours: int = Field(default=48, ge=1, le=168)


class IncidentPlanRequest(BaseModel):
    """Payload accepted by incident planning endpoint."""

    incident_id: str


class OptimizationConstraints(BaseModel):
    """Constraint payload for optimization endpoint."""

    budget_ceiling: float = Field(default=45000.0, ge=0.0)
    available_crew: dict[str, int] = Field(default_factory=dict)
    spare_parts_inventory: list[str] = Field(default_factory=list)
    blackout_windows: list[str] = Field(default_factory=list)


class IncidentOptimizeRequest(BaseModel):
    """Payload accepted by optimization endpoint."""

    incident_id: str
    constraints: OptimizationConstraints = Field(default_factory=OptimizationConstraints)


class IncidentSimulateRequest(BaseModel):
    """Payload accepted by simulation endpoint."""

    incident_id: str
    horizon_days: int = Field(default=30, ge=1, le=90)


class RunIncidentRequest(BaseModel):
    """Payload for running the entire incident pipeline in one call."""

    ingest: IngestBatchRequest
    lookahead_hours: int = Field(default=48, ge=1, le=168)
    constraints: OptimizationConstraints = Field(default_factory=OptimizationConstraints)
    horizon_days: int = Field(default=30, ge=1, le=90)


class MLAdapterHealth(BaseModel):
    """Runtime health snapshot for ML adapter dependencies."""

    ml_models_importable: bool
    orchestrator_importable: bool
    details: list[str] = Field(default_factory=list)


class MLAdapterStageResult(BaseModel):
    """Typed stage output contract from adapter to pipeline service."""

    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class IncidentRecord(BaseModel):
    """In-memory representation of one incident lifecycle."""

    incident_id: str
    created_at: datetime
    updated_at: datetime
    source: str
    stages: dict[str, dict[str, Any]] = Field(default_factory=dict)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    confidence_trail: list[dict[str, Any]] = Field(default_factory=list)
    governance_trail: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)