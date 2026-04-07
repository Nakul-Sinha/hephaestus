from backend.contracts import (
    IncidentOptimizeRequest,
    IncidentPlanRequest,
    IncidentSimulateRequest,
    IngestBatchRequest,
    OptimizationConstraints,
    RiskAnalyzeRequest,
)
from backend.services.incident_service import IncidentService
from backend.storage.memory_repository import InMemoryIncidentRepository


def test_incident_service_full_lifecycle_state_is_recorded() -> None:
    repo = InMemoryIncidentRepository()
    service = IncidentService(repository=repo)

    ingest_payload = service.ingest_batch(
        IngestBatchRequest(source="synthetic", telemetry_rows=12000, event_rows=120, maintenance_rows=40)
    )
    incident_id = ingest_payload["incident_id"]

    risk_payload, risk_conf, _ = service.analyze_risk(RiskAnalyzeRequest(incident_id=incident_id))
    assert risk_payload["risk_band"] in {"medium", "high"}
    assert risk_conf > 0.0

    plan_payload, _, _ = service.plan_incident(IncidentPlanRequest(incident_id=incident_id))
    assert len(plan_payload["plans"]) == 3

    optimize_payload, optimize_conf, _ = service.optimize_incident(
        IncidentOptimizeRequest(
            incident_id=incident_id,
            constraints=OptimizationConstraints(
                budget_ceiling=5000.0,
                available_crew={"mechanic": 1},
                spare_parts_inventory=["lubricant"],
            ),
        )
    )
    assert optimize_payload["recommended_plan_id"]
    assert optimize_conf > 0.0

    simulate_payload, _, _ = service.simulate_incident(
        IncidentSimulateRequest(incident_id=incident_id, horizon_days=14)
    )
    assert simulate_payload["horizon_days"] == 14
    assert len(simulate_payload["simulations"]) >= 1

    report_payload, report_conf, _ = service.generate_report(incident_id)
    assert report_payload["incident_id"] == incident_id
    assert len(report_payload["audit_trace"]) >= 5
    assert report_conf <= 1.0
