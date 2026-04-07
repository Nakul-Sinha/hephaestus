"""Pipeline orchestration service for staged backend workflows."""

from __future__ import annotations

from importlib import import_module

from backend.config import BackendSettings, get_settings
from backend.contracts import (
    IncidentOptimizeRequest,
    IncidentPlanRequest,
    IncidentSimulateRequest,
    IngestBatchRequest,
    RunIncidentRequest,
    RiskAnalyzeRequest,
)
from backend.services.governance_service import GovernanceService
from backend.services.incident_service import IncidentService


class PipelineService:
    """Centralizes stage execution and governance checks for routes."""

    def __init__(
        self,
        incident_service: IncidentService,
        governance_service: GovernanceService,
        settings: BackendSettings,
    ) -> None:
        self.incident_service = incident_service
        self.governance_service = governance_service
        self.settings = settings

    def _ml_agents_available(self) -> bool:
        """Detect whether ML agent orchestrator implementation is callable."""
        try:
            module = import_module("ml.aegis.agents.orchestrator")
            return hasattr(module, "Orchestrator")
        except Exception:
            return False

    def _enrich_with_governance(
        self,
        incident_id: str,
        stage: str,
        payload: dict,
        confidence: float,
        warnings: list[str],
    ) -> tuple[dict, float, list[str]]:
        availability = self._ml_agents_available()
        enriched_warnings = list(warnings)
        if not availability:
            enriched_warnings.append("ml agents unavailable; deterministic fallback path used")

        governance = self.governance_service.evaluate(
            incident_id=incident_id,
            stage=stage,
            confidence=confidence,
            warnings=enriched_warnings,
        )
        self.incident_service.repository.add_governance_event(incident_id, governance)

        enriched_payload = {
            **payload,
            "execution_mode": "ml" if availability else "fallback",
            "governance": governance,
        }
        return enriched_payload, confidence, enriched_warnings

    def ingest_batch(self, request: IngestBatchRequest) -> tuple[dict, float, list[str]]:
        payload = self.incident_service.ingest_batch(request)
        confidence = float(payload.get("ingest", {}).get("confidence", 1.0))
        warnings = payload.get("ingest", {}).get("warnings", [])
        return self._enrich_with_governance(payload["incident_id"], "ingest", payload, confidence, warnings)

    def analyze_risk(self, request: RiskAnalyzeRequest) -> tuple[dict, float, list[str]]:
        payload, confidence, warnings = self.incident_service.analyze_risk(request)
        return self._enrich_with_governance(request.incident_id, "risk", payload, confidence, warnings)

    def plan_incident(self, request: IncidentPlanRequest) -> tuple[dict, float, list[str]]:
        payload, confidence, warnings = self.incident_service.plan_incident(request)
        return self._enrich_with_governance(request.incident_id, "plan", payload, confidence, warnings)

    def optimize_incident(self, request: IncidentOptimizeRequest) -> tuple[dict, float, list[str]]:
        payload, confidence, warnings = self.incident_service.optimize_incident(request)
        return self._enrich_with_governance(request.incident_id, "optimize", payload, confidence, warnings)

    def simulate_incident(self, request: IncidentSimulateRequest) -> tuple[dict, float, list[str]]:
        payload, confidence, warnings = self.incident_service.simulate_incident(request)
        return self._enrich_with_governance(request.incident_id, "simulate", payload, confidence, warnings)

    def generate_report(self, incident_id: str) -> tuple[dict, float, list[str]]:
        payload, confidence, warnings = self.incident_service.generate_report(incident_id)
        return self._enrich_with_governance(incident_id, "report", payload, confidence, warnings)

    def run_full_pipeline(self, request: RunIncidentRequest) -> tuple[dict, float, list[str]]:
        """Execute ingest -> risk -> plan -> optimize -> simulate -> report in one call."""
        ingest_payload, ingest_conf, ingest_warn = self.ingest_batch(request.ingest)
        incident_id = ingest_payload["incident_id"]

        risk_payload, risk_conf, risk_warn = self.analyze_risk(
            RiskAnalyzeRequest(incident_id=incident_id, lookahead_hours=request.lookahead_hours)
        )
        plan_payload, plan_conf, plan_warn = self.plan_incident(IncidentPlanRequest(incident_id=incident_id))
        optimize_payload, optimize_conf, optimize_warn = self.optimize_incident(
            IncidentOptimizeRequest(incident_id=incident_id, constraints=request.constraints)
        )
        simulate_payload, simulate_conf, simulate_warn = self.simulate_incident(
            IncidentSimulateRequest(incident_id=incident_id, horizon_days=request.horizon_days)
        )
        report_payload, report_conf, report_warn = self.generate_report(incident_id)

        warnings = ingest_warn + risk_warn + plan_warn + optimize_warn + simulate_warn + report_warn
        payload = {
            "incident_id": incident_id,
            "ingest": ingest_payload,
            "risk": risk_payload,
            "plan": plan_payload,
            "optimize": optimize_payload,
            "simulate": simulate_payload,
            "report": report_payload,
        }
        confidence = min(ingest_conf, risk_conf, plan_conf, optimize_conf, simulate_conf, report_conf)
        return payload, confidence, warnings


def build_pipeline_service(incident_service: IncidentService) -> PipelineService:
    """Construct pipeline service with governance dependencies."""
    settings = get_settings()
    governance_service = GovernanceService(settings=settings)
    return PipelineService(
        incident_service=incident_service,
        governance_service=governance_service,
        settings=settings,
    )