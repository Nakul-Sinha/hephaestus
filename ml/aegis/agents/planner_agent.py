"""
Planner Agent (Agent 6)

Responsibilities:
    - Generate 3+ candidate intervention plans per incident
    - Each plan: recommended action, required parts, crew skills,
      estimated duration, maintenance window, rollback plan
    - Validate LLM output against InterventionPlan Pydantic schema

LLM Strategy:
    - Primary: Ollama local inference (Mistral/Llama) with JSON output enforcement
    - Fallback: Google Gemini API with response_schema validation
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from ml.aegis.agents import BaseAgent
from ml.aegis.data.schemas import AgentOutput, InterventionPlan


# ---------------------------------------------------------------------------
# LLM Caller
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, max_retries: int = 2) -> str | None:
    """Call LLM with Ollama primary, Gemini fallback."""
    for attempt in range(max_retries):
        try:
            import ollama
            response = ollama.generate(
                model="mistral",
                prompt=prompt,
                format="json",
                options={"temperature": 0.4},
            )
            return response.get("response", "")
        except Exception:
            if attempt == max_retries - 1:
                break

    try:
        import google.generativeai as genai
        import os
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.4,
            ),
        )
        return response.text
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Heuristic plan templates (used when LLM is unavailable)
# ---------------------------------------------------------------------------

PLAN_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "bearing_degradation": [
        {
            "recommended_action": "Full bearing replacement with shaft realignment",
            "required_parts": ["SKF_bearing_6205", "alignment_shims", "lubricant_kit"],
            "required_skills": ["mechanical_technician", "vibration_analyst"],
            "estimated_duration_minutes": 240,
            "maintenance_window": "next_planned_shutdown",
            "predicted_risk_reduction": 0.90,
            "estimated_cost": 2200.0,
            "expected_downtime_minutes": 300,
            "rollback_plan": "Reinstall original bearing if replacement is defective; monitor vibration at 1hr intervals",
        },
        {
            "recommended_action": "Enhanced lubrication and vibration monitoring",
            "required_parts": ["high_grade_lubricant", "portable_vibration_sensor"],
            "required_skills": ["mechanical_technician"],
            "estimated_duration_minutes": 60,
            "maintenance_window": "next_business_day",
            "predicted_risk_reduction": 0.40,
            "estimated_cost": 350.0,
            "expected_downtime_minutes": 90,
            "rollback_plan": "Revert to standard lubricant; escalate to full replacement if vibration increases",
        },
        {
            "recommended_action": "Increase monitoring frequency; defer intervention",
            "required_parts": [],
            "required_skills": ["vibration_analyst"],
            "estimated_duration_minutes": 30,
            "maintenance_window": "immediate",
            "predicted_risk_reduction": 0.10,
            "estimated_cost": 100.0,
            "expected_downtime_minutes": 0,
            "rollback_plan": "Escalate to active intervention if degradation rate increases",
        },
    ],
    "seal_leak": [
        {
            "recommended_action": "Complete seal replacement and pressure test",
            "required_parts": ["mechanical_seal_kit", "gasket_set", "pressure_test_kit"],
            "required_skills": ["mechanical_technician", "pressure_specialist"],
            "estimated_duration_minutes": 180,
            "maintenance_window": "next_planned_shutdown",
            "predicted_risk_reduction": 0.85,
            "estimated_cost": 1800.0,
            "expected_downtime_minutes": 240,
            "rollback_plan": "Temporary sealant application while sourcing replacement parts",
        },
        {
            "recommended_action": "Apply temporary sealant compound and monitor pressure",
            "required_parts": ["industrial_sealant", "pressure_gauge"],
            "required_skills": ["mechanical_technician"],
            "estimated_duration_minutes": 90,
            "maintenance_window": "next_business_day",
            "predicted_risk_reduction": 0.45,
            "estimated_cost": 400.0,
            "expected_downtime_minutes": 120,
            "rollback_plan": "Schedule full seal replacement within 7 days",
        },
        {
            "recommended_action": "Monitor pressure trends; schedule replacement in next maintenance window",
            "required_parts": [],
            "required_skills": ["process_engineer"],
            "estimated_duration_minutes": 15,
            "maintenance_window": "immediate",
            "predicted_risk_reduction": 0.05,
            "estimated_cost": 50.0,
            "expected_downtime_minutes": 0,
            "rollback_plan": "Immediate shutdown if pressure drops below safety threshold",
        },
    ],
    "overheating": [
        {
            "recommended_action": "Cooling system overhaul and thermal insulation repair",
            "required_parts": ["coolant_fluid", "thermal_paste", "heat_exchanger_gasket"],
            "required_skills": ["thermal_engineer", "mechanical_technician"],
            "estimated_duration_minutes": 300,
            "maintenance_window": "next_planned_shutdown",
            "predicted_risk_reduction": 0.88,
            "estimated_cost": 3500.0,
            "expected_downtime_minutes": 360,
            "rollback_plan": "Temporary external cooling; reduce operating load by 30%",
        },
        {
            "recommended_action": "Reduce operating load to 70% and flush cooling system",
            "required_parts": ["coolant_fluid", "flush_kit"],
            "required_skills": ["process_engineer"],
            "estimated_duration_minutes": 120,
            "maintenance_window": "next_business_day",
            "predicted_risk_reduction": 0.50,
            "estimated_cost": 600.0,
            "expected_downtime_minutes": 150,
            "rollback_plan": "Return to normal load if temperatures stabilize within 24 hours",
        },
        {
            "recommended_action": "Increase temperature monitoring frequency to every 5 minutes",
            "required_parts": [],
            "required_skills": ["process_engineer"],
            "estimated_duration_minutes": 15,
            "maintenance_window": "immediate",
            "predicted_risk_reduction": 0.08,
            "estimated_cost": 75.0,
            "expected_downtime_minutes": 0,
            "rollback_plan": "Automatic shutdown if temperature exceeds critical limit",
        },
    ],
    "electrical_fault": [
        {
            "recommended_action": "Full electrical diagnostic and component replacement",
            "required_parts": ["motor_winding_kit", "capacitor_bank", "wiring_harness"],
            "required_skills": ["electrician", "motor_specialist"],
            "estimated_duration_minutes": 360,
            "maintenance_window": "next_planned_shutdown",
            "predicted_risk_reduction": 0.92,
            "estimated_cost": 4200.0,
            "expected_downtime_minutes": 420,
            "rollback_plan": "Switch to backup drive unit if available; isolate faulty circuit",
        },
        {
            "recommended_action": "Isolate suspect circuit and run on partial capacity",
            "required_parts": ["circuit_breaker", "fuse_kit"],
            "required_skills": ["electrician"],
            "estimated_duration_minutes": 90,
            "maintenance_window": "next_business_day",
            "predicted_risk_reduction": 0.55,
            "estimated_cost": 500.0,
            "expected_downtime_minutes": 120,
            "rollback_plan": "Full shutdown if isolation fails; schedule complete rewiring",
        },
        {
            "recommended_action": "Monitor electrical parameters; schedule diagnostic",
            "required_parts": [],
            "required_skills": ["electrician"],
            "estimated_duration_minutes": 30,
            "maintenance_window": "immediate",
            "predicted_risk_reduction": 0.10,
            "estimated_cost": 100.0,
            "expected_downtime_minutes": 0,
            "rollback_plan": "Emergency shutdown procedure if anomalous current detected",
        },
    ],
}

# Default template for unknown causes
DEFAULT_TEMPLATES = [
    {
        "recommended_action": "Comprehensive inspection and targeted repair",
        "required_parts": ["general_repair_kit"],
        "required_skills": ["mechanical_technician"],
        "estimated_duration_minutes": 180,
        "maintenance_window": "next_planned_shutdown",
        "predicted_risk_reduction": 0.70,
        "estimated_cost": 1500.0,
        "expected_downtime_minutes": 240,
        "rollback_plan": "Revert any changes and escalate to specialist",
    },
    {
        "recommended_action": "Targeted minor repair based on primary symptom",
        "required_parts": ["minor_repair_kit"],
        "required_skills": ["mechanical_technician"],
        "estimated_duration_minutes": 90,
        "maintenance_window": "next_business_day",
        "predicted_risk_reduction": 0.40,
        "estimated_cost": 500.0,
        "expected_downtime_minutes": 120,
        "rollback_plan": "Monitor for 24 hours post-repair; escalate if symptoms persist",
    },
    {
        "recommended_action": "Enhanced monitoring and deferred maintenance",
        "required_parts": [],
        "required_skills": ["process_engineer"],
        "estimated_duration_minutes": 15,
        "maintenance_window": "immediate",
        "predicted_risk_reduction": 0.10,
        "estimated_cost": 75.0,
        "expected_downtime_minutes": 0,
        "rollback_plan": "Escalate to active intervention if condition worsens",
    },
]


class PlannerAgent(BaseAgent):

    name = "planner_agent"
    description = "Generates candidate intervention plans from causal analysis"

    def run(self, context: dict[str, Any]) -> AgentOutput:
        hypotheses_by_asset = context.get("hypotheses_by_asset", {})
        prediction_summaries = context.get("prediction_summaries", [])
        assets = context.get("assets", [])

        if not hypotheses_by_asset:
            return self._build_output(
                payload={"status": "no_hypotheses"},
                confidence=0.0,
                errors=["No causal hypotheses available for planning"],
                next_agent="reporter_agent",
            )

        asset_map = {a.asset_id: a for a in assets}
        all_plans: list[InterventionPlan] = []
        evidence: list[str] = []
        errors: list[str] = []

        for asset_id, hypotheses in hypotheses_by_asset.items():
            if not hypotheses:
                continue

            top_cause = hypotheses[0].get("cause", "unknown")
            cause_confidence = hypotheses[0].get("confidence", 0.5)

            # Try LLM first
            llm_plans = self._generate_plans_llm(
                asset_id=asset_id,
                asset_type=asset_map[asset_id].asset_type.value if asset_id in asset_map else "unknown",
                top_cause=top_cause,
                hypotheses=hypotheses,
            )

            if llm_plans:
                for plan_data in llm_plans:
                    plan = self._make_plan(asset_id, plan_data, cause_confidence)
                    if plan:
                        all_plans.append(plan)
                evidence.append(f"LLM generated {len(llm_plans)} plans for {asset_id}")
            else:
                # Heuristic fallback
                templates = PLAN_TEMPLATES.get(top_cause, DEFAULT_TEMPLATES)
                for tmpl in templates:
                    plan = self._make_plan(asset_id, tmpl, cause_confidence)
                    if plan:
                        all_plans.append(plan)
                errors.append(f"LLM unavailable — used heuristic plans for {asset_id} ({top_cause})")
                evidence.append(f"Heuristic plans generated for {asset_id}")

        # Confidence
        quality_modifier = context.get("quality_confidence_modifier", 1.0)
        if all_plans:
            avg_conf = sum(p.confidence for p in all_plans) / len(all_plans)
            confidence = min(0.85, avg_conf) * quality_modifier
        else:
            confidence = 0.0

        context["plans"] = all_plans

        plans_payload = [
            {
                "plan_id": p.plan_id,
                "asset_id": p.plan_id.split("-")[0] if "-" in p.plan_id else "",
                "recommended_action": p.recommended_action,
                "estimated_cost": p.estimated_cost,
                "predicted_risk_reduction": p.predicted_risk_reduction,
                "estimated_duration_minutes": p.estimated_duration_minutes,
                "confidence": p.confidence,
            }
            for p in all_plans
        ]

        return self._build_output(
            payload={
                "status": "completed",
                "total_plans": len(all_plans),
                "assets_planned": len(hypotheses_by_asset),
                "plans": plans_payload,
            },
            confidence=round(confidence, 4),
            assumptions=[
                "Plan costs are estimates based on typical maintenance operations",
                "Crew availability and parts inventory not yet validated (done by Optimizer)",
            ],
            evidence=evidence,
            errors=errors if errors else None,
            next_agent="optimizer_agent",
        )

    def _generate_plans_llm(
        self, asset_id: str, asset_type: str,
        top_cause: str, hypotheses: list[dict],
    ) -> list[dict] | None:
        prompt = f"""You are an industrial maintenance planner AI.

ASSET: {asset_id} (type: {asset_type})
ROOT CAUSE ANALYSIS:
{json.dumps(hypotheses[:3], indent=2)}

Generate exactly 3 maintenance plans as a JSON object with a "plans" array.

Plan 1: AGGRESSIVE — highest cost, maximum risk reduction, full fix
Plan 2: MODERATE — balanced cost/risk, partial fix
Plan 3: CONSERVATIVE — minimal cost, monitor/defer, lowest risk reduction

Each plan must have these fields:
- "recommended_action": string
- "required_parts": list of strings
- "required_skills": list of strings
- "estimated_duration_minutes": integer
- "maintenance_window": "immediate" | "next_business_day" | "next_planned_shutdown"
- "predicted_risk_reduction": float 0.0-1.0
- "estimated_cost": float
- "expected_downtime_minutes": integer
- "rollback_plan": string

Return ONLY valid JSON."""

        raw = _call_llm(prompt)
        if not raw:
            return None
        try:
            data = json.loads(raw)
            plans = data.get("plans", data) if isinstance(data, dict) else data
            return plans if isinstance(plans, list) else None
        except (json.JSONDecodeError, TypeError):
            return None

    def _make_plan(
        self, asset_id: str, data: dict, cause_confidence: float,
    ) -> InterventionPlan | None:
        try:
            plan_id = f"{asset_id}-{uuid.uuid4().hex[:6].upper()}"
            return InterventionPlan(
                plan_id=plan_id,
                recommended_action=data.get("recommended_action", "Unknown action"),
                required_parts=data.get("required_parts", []),
                required_skills=data.get("required_skills", []),
                estimated_duration_minutes=int(data.get("estimated_duration_minutes", 120)),
                maintenance_window=data.get("maintenance_window", "next_business_day"),
                predicted_risk_reduction=min(1.0, max(0.0, float(data.get("predicted_risk_reduction", 0.5)))),
                estimated_cost=max(0.0, float(data.get("estimated_cost", 500))),
                expected_downtime_minutes=int(data.get("expected_downtime_minutes", 60)),
                confidence=min(1.0, max(0.0, cause_confidence * 0.9)),
                assumptions=[f"Based on root cause: {data.get('recommended_action', 'N/A')[:40]}"],
                rollback_plan=data.get("rollback_plan", "Revert changes and escalate to specialist"),
            )
        except Exception:
            return None
