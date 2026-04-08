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
from backend.services.ml_adapter_service import MLAdapterService


class PipelineService:
    """Centralizes stage execution and governance checks for routes."""

    def __init__(
        self,
        ml_adapter: MLAdapterService,
        governance_service: GovernanceService,
        settings: BackendSettings,
    ) -> None:
        self.ml_adapter = ml_adapter
        self.governance_service = governance_service
        self.settings = settings

    def _ml_agents_available(self, stage: str) -> bool:
        """Detect whether required ML runtime is available for a given stage."""
        adapter_health = self.ml_adapter.health()
        # Stage-2 integration requires model modules for risk path.
        if stage in {"ingest", "risk", "plan", "optimize", "simulate", "report"}:
            return adapter_health.ml_models_importable

        if not adapter_health.orchestrator_importable:
            return False
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
        availability = self._ml_agents_available(stage)
        enriched_warnings = list(warnings)
        if not availability:
            enriched_warnings.append("ml agents unavailable; deterministic fallback path used")

        governance = self.governance_service.evaluate(
            incident_id=incident_id,
            stage=stage,
            confidence=confidence,
            warnings=enriched_warnings,
        )
        self.ml_adapter.repository.add_governance_event(incident_id, governance)

        enriched_payload = {
            **payload,
            "execution_mode": "ml" if availability else "fallback",
            "governance": governance,
        }
        return enriched_payload, confidence, enriched_warnings

    def ingest_batch(self, request: IngestBatchRequest) -> tuple[dict, float, list[str]]:
        stage_result = self.ml_adapter.run_ingest(request)
        payload = stage_result.payload
        confidence = stage_result.confidence
        warnings = stage_result.warnings
        return self._enrich_with_governance(payload["incident_id"], "ingest", payload, confidence, warnings)

    def analyze_risk(self, request: RiskAnalyzeRequest) -> tuple[dict, float, list[str]]:
        stage_result = self.ml_adapter.run_risk(request)
        payload, confidence, warnings = stage_result.payload, stage_result.confidence, stage_result.warnings
        return self._enrich_with_governance(request.incident_id, "risk", payload, confidence, warnings)

    def plan_incident(self, request: IncidentPlanRequest) -> tuple[dict, float, list[str]]:
        stage_result = self.ml_adapter.run_plan(request)
        payload, confidence, warnings = stage_result.payload, stage_result.confidence, stage_result.warnings
        return self._enrich_with_governance(request.incident_id, "plan", payload, confidence, warnings)

    def optimize_incident(self, request: IncidentOptimizeRequest) -> tuple[dict, float, list[str]]:
        stage_result = self.ml_adapter.run_optimize(request)
        payload, confidence, warnings = stage_result.payload, stage_result.confidence, stage_result.warnings
        return self._enrich_with_governance(request.incident_id, "optimize", payload, confidence, warnings)

    def simulate_incident(self, request: IncidentSimulateRequest) -> tuple[dict, float, list[str]]:
        stage_result = self.ml_adapter.run_simulate(request)
        payload, confidence, warnings = stage_result.payload, stage_result.confidence, stage_result.warnings
        return self._enrich_with_governance(request.incident_id, "simulate", payload, confidence, warnings)

    def generate_report(self, incident_id: str) -> tuple[dict, float, list[str]]:
        stage_result = self.ml_adapter.run_report(incident_id)
        payload, confidence, warnings = stage_result.payload, stage_result.confidence, stage_result.warnings
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


def build_pipeline_service(ml_adapter: MLAdapterService) -> PipelineService:
    """Construct pipeline service with governance dependencies."""
    settings = get_settings()
    governance_service = GovernanceService(settings=settings)
    return PipelineService(
        ml_adapter=ml_adapter,
        governance_service=governance_service,
        settings=settings,
    )