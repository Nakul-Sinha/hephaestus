"""
Reporter Agent (Agent 9)

Responsibilities:
    - Compile data from the entire pipeline into human-readable reports
    - Generate 3 formats: Operator Playbook, Manager Summary, Audit Trace
    - Use LLM for plain-text narrative generation

Next agent: governance_agent.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ml.aegis.agents import BaseAgent
from ml.aegis.data.schemas import AgentOutput


def _call_llm(prompt: str, max_retries: int = 2) -> str | None:
    for attempt in range(max_retries):
        try:
            import ollama
            response = ollama.generate(
                model="mistral",
                prompt=prompt,
                options={"temperature": 0.2},
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
            generation_config=genai.GenerationConfig(temperature=0.2),
        )
        return response.text
    except Exception:
        return None


class ReporterAgent(BaseAgent):

    name = "reporter_agent"
    description = "Composes human-readable summaries and audit logs"

    def run(self, context: dict[str, Any]) -> AgentOutput:
        quality_summary = context.get("quality_summary", {})
        prediction_summaries = context.get("prediction_summaries", [])
        causal_hypotheses = context.get("causal_hypotheses", [])
        optimized_plans = context.get("optimized_plans", [])
        recommended_plan_id = context.get("recommended_plan_id")
        simulations = context.get("simulations", [])
        
        evidence: list[str] = []
        errors: list[str] = []

        # 1. Audit Trace (always generated entirely from context)
        audit_trace = {
            "timestamp": str(datetime.now()),
            "pipeline_run_id": context.get("pipeline_run_id", "unknown"),
            "data_quality_verdict": quality_summary.get("overall_verdict", "unknown"),
            "assets_analyzed": len(prediction_summaries),
            "critical_assets": sum(1 for p in prediction_summaries if p["risk_level"] == "critical"),
            "hypotheses_generated": len(causal_hypotheses),
            "plans_generated": len(optimized_plans),
            "recommended_plan": recommended_plan_id,
        }
        
        # 2. Extract best plan payload
        best_plan = next((p for p in optimized_plans if p.plan_id == recommended_plan_id), None)
        target_asset = best_plan.plan_id.split("-")[0] if best_plan and "-" in best_plan.plan_id else "Fleet"
        
        # 3. Use LLM to write narratives
        if best_plan:
            # Manager Summary
            manager_prompt = f"""Write a 1-paragraph business summary for an industrial plant manager.
            Asset: {target_asset}
            Status: Critical failure predicted
            Recommended Action: {best_plan.recommended_action}
            Cost: ${best_plan.estimated_cost}
            Downtime: {best_plan.expected_downtime_minutes} minutes
            Risk Reduction: {best_plan.predicted_risk_reduction * 100}%
            Be concise, professional, and focus on business impact."""
            
            manager_summary = _call_llm(manager_prompt) or "LLM generation failed. Check audit trace."
            
            # Operator Playbook
            operator_prompt = f"""Write a step-by-step checklist for a maintenance technician.
            Asset: {target_asset}
            Action: {best_plan.recommended_action}
            Parts needed: {', '.join(best_plan.required_parts)}
            Skills needed: {', '.join(best_plan.required_skills)}
            Time: {best_plan.estimated_duration_minutes} minutes
            Format as a clear, bulleted list."""
            
            operator_playbook = _call_llm(operator_prompt) or "LLM generation failed. Check audit trace."
            
        else:
            if quality_summary.get("overall_verdict") == "unreliable":
                manager_summary = "CRITICAL: Telemetry data quality is too poor for predictive modeling. Sensor diagnostic required."
                operator_playbook = "1. Inspect field sensors for disconnects or physical damage.\n2. Verify network pathways to historian."
            else:
                manager_summary = "All monitored assets are operating within normal parameters. No intervention recommended."
                operator_playbook = "No pending maintenance actions."

        evidence.append("Audit trace generated")
        evidence.append("Manager summary generated")
        evidence.append("Operator playbook generated")

        return self._build_output(
            payload={
                "status": "completed",
                "reports": {
                    "manager_summary": manager_summary,
                    "operator_playbook": operator_playbook,
                    "audit_trace": audit_trace,
                }
            },
            confidence=1.0,
            assumptions=["Report reflects the outputs of all preceding agents"],
            evidence=evidence,
            errors=errors if errors else None,
            next_agent="governance_agent",
        )
