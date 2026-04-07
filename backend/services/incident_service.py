"""Service layer for incident lifecycle workflow endpoints."""

from __future__ import annotations

from uuid import uuid4

from backend.contracts import (
    IncidentOptimizeRequest,
    IncidentPlanRequest,
    IncidentSimulateRequest,
    IngestBatchRequest,
    RiskAnalyzeRequest,
)
from backend.storage import InMemoryIncidentRepository


class IncidentService:
    """Coordinates stage computations and persists stage outputs."""

    def __init__(self, repository: InMemoryIncidentRepository) -> None:
        self.repository = repository

    def ingest_batch(self, request: IngestBatchRequest) -> dict:
        incident_id = f"inc-{uuid4().hex[:10]}"
        warnings = []
        if request.telemetry_rows == 0:
            warnings.append("telemetry_rows is zero; downstream confidence may degrade")
        payload = {
            "source": request.source,
            "telemetry_rows": request.telemetry_rows,
            "event_rows": request.event_rows,
            "maintenance_rows": request.maintenance_rows,
            "notes": request.notes,
            "confidence": 1.0 if request.telemetry_rows > 0 else 0.7,
            "warnings": warnings,
        }
        record = self.repository.create(incident_id=incident_id, source=request.source, initial_stage=payload)
        return {
            "incident_id": record.incident_id,
            "ingest": record.stages["ingest"],
            "timeline": record.timeline,
        }

    def analyze_risk(self, request: RiskAnalyzeRequest) -> tuple[dict, float, list[str]]:
        record = self.repository.get(request.incident_id)
        telemetry_rows = record.stages.get("ingest", {}).get("telemetry_rows", 0)
        base_probability = 0.35 if telemetry_rows < 5000 else 0.62
        risk_probability = min(0.95, round(base_probability + (request.lookahead_hours / 200.0), 2))
        risk_payload = {
            "lookahead_hours": request.lookahead_hours,
            "asset_id": "PUMP-0042",
            "failure_probability": risk_probability,
            "anomaly_score": round(min(0.99, risk_probability + 0.08), 2),
            "risk_band": "high" if risk_probability >= 0.7 else "medium",
        }
        warnings = [] if telemetry_rows > 0 else ["risk computed from sparse telemetry"]
        confidence = 0.84 if telemetry_rows >= 5000 else 0.68
        self.repository.save_stage(request.incident_id, "risk", risk_payload, confidence, warnings)
        return risk_payload, confidence, warnings

    def plan_incident(self, request: IncidentPlanRequest) -> tuple[dict, float, list[str]]:
        record = self.repository.get(request.incident_id)
        risk_payload = record.stages.get("risk")
        if not risk_payload:
            raise ValueError("risk stage must be completed before planning")

        plans = [
            {
                "plan_id": "plan-a",
                "recommended_action": "replace bearing assembly",
                "predicted_risk_reduction": 0.82,
                "estimated_cost": 7800.0,
                "expected_downtime_minutes": 240,
                "required_skills": ["mechanic", "bearing_specialist"],
                "required_parts": ["SKF_bearing"],
            },
            {
                "plan_id": "plan-b",
                "recommended_action": "emergency lubrication and condition monitoring",
                "predicted_risk_reduction": 0.58,
                "estimated_cost": 1200.0,
                "expected_downtime_minutes": 30,
                "required_skills": ["mechanic"],
                "required_parts": ["lubricant"],
            },
            {
                "plan_id": "plan-c",
                "recommended_action": "defer maintenance and increase monitoring frequency",
                "predicted_risk_reduction": 0.10,
                "estimated_cost": 100.0,
                "expected_downtime_minutes": 0,
                "required_skills": ["operator"],
                "required_parts": [],
            },
        ]

        payload = {
            "root_cause": "bearing_degradation",
            "root_cause_confidence": 0.78,
            "plans": plans,
        }
        confidence = 0.79
        warnings = []
        self.repository.save_stage(request.incident_id, "plan", payload, confidence, warnings)
        return payload, confidence, warnings

    def optimize_incident(self, request: IncidentOptimizeRequest) -> tuple[dict, float, list[str]]:
        record = self.repository.get(request.incident_id)
        plans = record.stages.get("plan", {}).get("plans", [])
        if not plans:
            raise ValueError("plan stage must be completed before optimization")

        scored_plans = []
        for plan in plans:
            blocked = []
            if plan["estimated_cost"] > request.constraints.budget_ceiling:
                blocked.append("budget ceiling exceeded")
            for skill in plan["required_skills"]:
                if request.constraints.available_crew and request.constraints.available_crew.get(skill, 0) <= 0:
                    blocked.append(f"missing skill: {skill}")
            for part in plan["required_parts"]:
                if request.constraints.spare_parts_inventory and part not in request.constraints.spare_parts_inventory:
                    blocked.append(f"missing part: {part}")

            score = round(
                (plan["predicted_risk_reduction"] * 100)
                - (plan["estimated_cost"] / 1000)
                - (plan["expected_downtime_minutes"] / 60),
                2,
            )
            scored_plans.append(
                {
                    **plan,
                    "is_feasible": len(blocked) == 0,
                    "infeasibility_reasons": blocked,
                    "optimizer_score": score,
                }
            )

        feasible = [plan for plan in scored_plans if plan["is_feasible"]]
        recommended = max(feasible, key=lambda plan: plan["optimizer_score"]) if feasible else scored_plans[0]

        payload = {
            "recommended_plan_id": recommended["plan_id"],
            "ranked_plans": sorted(scored_plans, key=lambda plan: plan["optimizer_score"], reverse=True),
            "constraints": request.constraints.model_dump(),
        }
        warnings = [] if feasible else ["no feasible plans found; fallback recommendation selected"]
        confidence = 0.83 if feasible else 0.55
        self.repository.save_stage(request.incident_id, "optimize", payload, confidence, warnings)
        return payload, confidence, warnings

    def simulate_incident(self, request: IncidentSimulateRequest) -> tuple[dict, float, list[str]]:
        record = self.repository.get(request.incident_id)
        optimization = record.stages.get("optimize")
        if not optimization:
            raise ValueError("optimize stage must be completed before simulation")

        ranked = optimization["ranked_plans"]
        simulations = []
        for plan in ranked[:3]:
            start_risk = 0.87
            drop = plan["predicted_risk_reduction"]
            day_one = round(max(0.02, start_risk * (1 - drop)), 2)
            day_end = round(min(0.98, day_one + (0.02 if plan["plan_id"] == "plan-c" else 0.25)), 2)
            simulations.append(
                {
                    "plan_id": plan["plan_id"],
                    "risk_curve": [start_risk, day_one, day_end],
                    "cost_curve": [0.0, plan["estimated_cost"], round(plan["estimated_cost"] * 1.1, 2)],
                    "downtime_hours": round(plan["expected_downtime_minutes"] / 60.0, 2),
                }
            )

        payload = {
            "horizon_days": request.horizon_days,
            "simulations": simulations,
        }
        confidence = 0.77
        warnings = []
        self.repository.save_stage(request.incident_id, "simulate", payload, confidence, warnings)
        return payload, confidence, warnings

    def generate_report(self, incident_id: str) -> tuple[dict, float, list[str]]:
        record = self.repository.get(incident_id)
        optimize = record.stages.get("optimize", {})
        recommended_plan_id = optimize.get("recommended_plan_id", "unknown")

        payload = {
            "incident_id": incident_id,
            "operator_playbook": {
                "recommended_plan_id": recommended_plan_id,
                "steps": [
                    "isolate the asset",
                    "execute intervention per plan requirements",
                    "run post-maintenance sensor verification",
                ],
            },
            "manager_summary": {
                "recommended_plan_id": recommended_plan_id,
                "expected_risk_reduction": "derived from optimized plan",
                "confidence": round(record.confidence, 2),
            },
            "audit_trace": record.timeline,
        }
        confidence = record.confidence
        warnings = record.warnings
        self.repository.save_stage(incident_id, "report", payload, confidence, warnings)
        return payload, confidence, warnings