from fastapi.testclient import TestClient

from backend.app import app
from backend.storage import get_incident_repository


client = TestClient(app)
API_HEADERS = {"x-api-key": "hephaestus-dev-key"}


def _extract_incident_id(ingest_response: dict) -> str:
    return ingest_response["payload"]["incident_id"]


def test_write_endpoint_requires_api_key() -> None:
    response = client.post("/ingest/batch", json={"source": "synthetic", "telemetry_rows": 1})
    assert response.status_code == 401
    payload = response.json()
    assert payload["status"] == "error"
    assert "error" in payload["payload"]


def test_full_api_incident_flow() -> None:
    get_incident_repository().clear()

    ingest = client.post(
        "/ingest/batch",
        json={"source": "synthetic", "telemetry_rows": 15000, "event_rows": 200, "maintenance_rows": 80},
        headers=API_HEADERS,
    )
    assert ingest.status_code == 200
    assert "x-request-id" in ingest.headers
    ingest_json = ingest.json()
    incident_id = _extract_incident_id(ingest_json)

    risk = client.post(
        "/risk/analyze",
        json={"incident_id": incident_id, "lookahead_hours": 48},
        headers=API_HEADERS,
    )
    assert risk.status_code == 200
    assert risk.json()["status"] == "success"

    plan = client.post("/incident/plan", json={"incident_id": incident_id}, headers=API_HEADERS)
    assert plan.status_code == 200

    optimize = client.post(
        "/incident/optimize",
        json={
            "incident_id": incident_id,
            "constraints": {
                "budget_ceiling": 10000,
                "available_crew": {"mechanic": 2, "bearing_specialist": 1},
                "spare_parts_inventory": ["SKF_bearing", "lubricant"],
            },
        },
        headers=API_HEADERS,
    )
    assert optimize.status_code == 200

    simulate = client.post(
        "/incident/simulate",
        json={"incident_id": incident_id, "horizon_days": 30},
        headers=API_HEADERS,
    )
    assert simulate.status_code == 200

    report = client.get(f"/incident/{incident_id}/report")
    assert report.status_code == 200
    report_json = report.json()
    assert report_json["status"] == "success"
    assert "audit_trace" in report_json["payload"]


def test_single_call_pipeline_endpoint() -> None:
    get_incident_repository().clear()

    response = client.post(
        "/incident/run",
        json={
            "ingest": {
                "source": "synthetic",
                "telemetry_rows": 9000,
                "event_rows": 90,
                "maintenance_rows": 40,
            },
            "lookahead_hours": 36,
            "constraints": {
                "budget_ceiling": 15000,
                "available_crew": {"mechanic": 2},
                "spare_parts_inventory": ["lubricant"],
            },
            "horizon_days": 21,
        },
        headers=API_HEADERS,
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "success"
    assert "report" in response_json["payload"]
