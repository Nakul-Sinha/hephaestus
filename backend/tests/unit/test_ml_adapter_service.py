from backend.contracts import (
    IncidentOptimizeRequest,
    IncidentPlanRequest,
    IncidentSimulateRequest,
    IngestBatchRequest,
    RiskAnalyzeRequest,
)
from backend.services.incident_service import IncidentService
from backend.services.ml_adapter_service import MLAdapterService
from backend.storage.memory_repository import InMemoryIncidentRepository


def test_ml_adapter_health_returns_flags() -> None:
    repo = InMemoryIncidentRepository()
    incident_service = IncidentService(repository=repo)
    adapter = MLAdapterService(incident_service=incident_service)

    health = adapter.health()

    assert isinstance(health.ml_models_importable, bool)
    assert isinstance(health.orchestrator_importable, bool)


def test_ml_adapter_ingest_and_risk_smoke() -> None:
    repo = InMemoryIncidentRepository()
    incident_service = IncidentService(repository=repo)
    adapter = MLAdapterService(incident_service=incident_service)

    ingest_result = adapter.run_ingest(
        IngestBatchRequest(source="synthetic", telemetry_rows=5000, event_rows=50, maintenance_rows=20)
    )
    incident_id = ingest_result.payload["incident_id"]

    risk_result = adapter.run_risk(RiskAnalyzeRequest(incident_id=incident_id, lookahead_hours=48))

    assert risk_result.payload["asset_id"]
    assert 0.0 <= risk_result.payload["failure_probability"] <= 1.0
    assert 0.0 <= risk_result.confidence <= 1.0
    assert risk_result.payload["model_source"] == "ml"

    plan_result = adapter.run_plan(IncidentPlanRequest(incident_id=incident_id))
    assert plan_result.payload["root_cause"]
    assert "explainability" in plan_result.payload
    assert plan_result.payload["evidence_refs"]

    optimize_result = adapter.run_optimize(
        IncidentOptimizeRequest(
            incident_id=incident_id,
            constraints={
                "budget_ceiling": 10000,
                "available_crew": {"mechanic": 2, "bearing_specialist": 1},
                "spare_parts_inventory": ["SKF_bearing", "lubricant"],
            },
        )
    )
    assert optimize_result.payload["model_source"] == "ml"
    assert optimize_result.payload["recommended_plan_id"]
    assert optimize_result.payload["ranked_plans"]

    simulate_result = adapter.run_simulate(IncidentSimulateRequest(incident_id=incident_id, horizon_days=21))
    assert simulate_result.payload["model_source"] == "ml"
    assert simulate_result.payload["simulations"]
    assert "pairwise_win_probabilities" in simulate_result.payload
