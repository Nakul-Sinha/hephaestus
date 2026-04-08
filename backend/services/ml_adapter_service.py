"""ML adapter boundary between backend services and ml/aegis runtime."""

from __future__ import annotations

from importlib import import_module

from backend.contracts import (
    IncidentOptimizeRequest,
    IncidentPlanRequest,
    IncidentSimulateRequest,
    IngestBatchRequest,
    MLAdapterHealth,
    MLAdapterStageResult,
    RiskAnalyzeRequest,
)
from backend.services.incident_service import IncidentService


class MLAdapterService:
    """Adapter facade used by pipeline service to call ML-backed stages."""

    def __init__(self, incident_service: IncidentService) -> None:
        self.incident_service = incident_service

    @property
    def repository(self):
        """Expose underlying repository for governance trail updates."""
        return self.incident_service.repository

    def health(self) -> MLAdapterHealth:
        """Return import health of key ML modules used by integration path."""
        details: list[str] = []

        def _importable(path: str) -> bool:
            try:
                import_module(path)
                return True
            except Exception as exc:
                details.append(f"{path} import failed: {exc}")
                return False

        models_ok = _importable("ml.aegis.models.anomaly") and _importable("ml.aegis.models.failure_risk")
        orchestrator_ok = _importable("ml.aegis.agents.orchestrator")

        return MLAdapterHealth(
            ml_models_importable=models_ok,
            orchestrator_importable=orchestrator_ok,
            details=details,
        )

    def run_ingest(self, request: IngestBatchRequest) -> MLAdapterStageResult:
        payload = self.incident_service.ingest_batch(request)
        ingest = payload.get("ingest", {})
        return MLAdapterStageResult(
            payload=payload,
            confidence=float(ingest.get("confidence", 1.0)),
            warnings=list(ingest.get("warnings", [])),
        )

    def run_risk(self, request: RiskAnalyzeRequest) -> MLAdapterStageResult:
        payload, confidence, warnings = self.incident_service.analyze_risk(request)
        return MLAdapterStageResult(payload=payload, confidence=confidence, warnings=warnings)

    def run_plan(self, request: IncidentPlanRequest) -> MLAdapterStageResult:
        payload, confidence, warnings = self.incident_service.plan_incident(request)
        return MLAdapterStageResult(payload=payload, confidence=confidence, warnings=warnings)

    def run_optimize(self, request: IncidentOptimizeRequest) -> MLAdapterStageResult:
        payload, confidence, warnings = self.incident_service.optimize_incident(request)
        return MLAdapterStageResult(payload=payload, confidence=confidence, warnings=warnings)

    def run_simulate(self, request: IncidentSimulateRequest) -> MLAdapterStageResult:
        payload, confidence, warnings = self.incident_service.simulate_incident(request)
        return MLAdapterStageResult(payload=payload, confidence=confidence, warnings=warnings)

    def run_report(self, incident_id: str) -> MLAdapterStageResult:
        payload, confidence, warnings = self.incident_service.generate_report(incident_id)
        return MLAdapterStageResult(payload=payload, confidence=confidence, warnings=warnings)
