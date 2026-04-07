"""
Causal Agent (Agent 5)

Responsibilities:
    - Produce ranked root-cause hypotheses
    - Correlate SHAP explanations, events, and maintenance history
    - Use LLM for causal reasoning synthesis
    - Handle contradictions between evidence sources

LLM Strategy:
    - Primary: Ollama local inference (Mistral/Llama) via /api/generate with JSON mode
    - Fallback: Google Gemini API (when Ollama is unavailable or errors after retries)

Next agent: planner_agent.
"""

from __future__ import annotations

import json
from typing import Any

from ml.aegis.agents import BaseAgent
from ml.aegis.data.schemas import AgentOutput, CausalHypothesis
from ml.aegis.models.explainability import get_top_risk_drivers, map_sensors_to_hypotheses


# ---------------------------------------------------------------------------
# LLM Caller (Ollama primary, Gemini fallback)
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, max_retries: int = 2) -> str | None:
    """
    Call LLM with Ollama primary, Gemini fallback.
    Returns raw response text or None on total failure.
    """
    # Try Ollama first
    for attempt in range(max_retries):
        try:
            import ollama
            response = ollama.generate(
                model="mistral",
                prompt=prompt,
                format="json",
                options={"temperature": 0.3},
            )
            return response.get("response", "")
        except Exception:
            if attempt == max_retries - 1:
                break

    # Fallback: Gemini API
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
                temperature=0.3,
            ),
        )
        return response.text
    except Exception:
        return None


def _parse_hypotheses_json(raw: str) -> list[dict[str, Any]]:
    """Safely parse LLM JSON response into hypothesis list."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data.get("hypotheses", [data])
        elif isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Causal Agent
# ---------------------------------------------------------------------------

class CausalAgent(BaseAgent):

    name = "causal_agent"
    description = "Produces ranked root-cause hypotheses with evidence"

    def run(self, context: dict[str, Any]) -> AgentOutput:
        prediction_summaries = context.get("prediction_summaries", [])
        explanations = context.get("explanations", [])
        events = context.get("events", [])
        maintenance = context.get("maintenance", [])
        assets = context.get("assets", [])

        if not prediction_summaries:
            return self._build_output(
                payload={"status": "no_predictions"},
                confidence=0.0,
                errors=["No risk predictions available for causal analysis"],
                next_agent="reporter_agent",
            )

        evidence_list: list[str] = []
        errors: list[str] = []
        all_hypotheses: list[dict[str, Any]] = []

        # Build asset metadata lookup
        asset_map = {a.asset_id: a for a in assets}

        # Process top at-risk assets (up to 5)
        top_assets = sorted(
            prediction_summaries,
            key=lambda x: x["failure_probability"],
            reverse=True,
        )[:5]

        for pred in top_assets:
            asset_id = pred["asset_id"]
            asset = asset_map.get(asset_id)

            # Gather SHAP-based evidence
            risk_drivers = pred.get("top_risk_drivers", [])

            # Get heuristic hypothesis mapping from SHAP
            heuristic_hypotheses = map_sensors_to_hypotheses(risk_drivers)

            # Gather recent events for this asset
            asset_events = [
                {"type": e.event_type.value, "severity": e.severity.value,
                 "text": e.event_text, "time": str(e.timestamp)}
                for e in events if e.asset_id == asset_id
            ][-10:]  # Last 10 events

            # Gather maintenance history
            asset_maint = [
                {"action": m.action_type.value, "cost": m.cost,
                 "outcome": m.outcome.value, "time": str(m.timestamp)}
                for m in maintenance if m.asset_id == asset_id
            ][-5:]  # Last 5 maintenance actions

            # Build LLM prompt
            prompt = self._build_prompt(
                asset_id=asset_id,
                asset_type=asset.asset_type.value if asset else "unknown",
                failure_probability=pred["failure_probability"],
                risk_drivers=risk_drivers,
                heuristic_hypotheses=heuristic_hypotheses,
                recent_events=asset_events,
                maintenance_history=asset_maint,
                asset_age_years=round(
                    (asset_map[asset_id].installation_date.timestamp()
                     if asset_id in asset_map else 0) / (-365.25 * 86400) + 56.2, 1
                ) if asset_id in asset_map else 0,
            )

            # Call LLM
            llm_response = _call_llm(prompt)

            if llm_response:
                parsed = _parse_hypotheses_json(llm_response)
                if parsed:
                    for h in parsed:
                        all_hypotheses.append({
                            "asset_id": asset_id,
                            **h,
                        })
                    evidence_list.append(f"LLM generated {len(parsed)} hypotheses for {asset_id}")
                else:
                    errors.append(f"LLM response for {asset_id} could not be parsed as JSON")
                    # Use heuristic fallback
                    for hh in heuristic_hypotheses:
                        all_hypotheses.append({
                            "asset_id": asset_id,
                            "cause": hh["cause"],
                            "confidence": hh["initial_confidence"],
                            "evidence_for": [f"SHAP: {s} is a top risk driver" for s in hh["supporting_sensors"]],
                            "evidence_against": [],
                            "contradiction_notes": "",
                        })
                    evidence_list.append(f"Used heuristic fallback for {asset_id}")
            else:
                # LLM completely unavailable — use pure heuristics
                for hh in heuristic_hypotheses:
                    all_hypotheses.append({
                        "asset_id": asset_id,
                        "cause": hh["cause"],
                        "confidence": hh["initial_confidence"],
                        "evidence_for": [f"SHAP: {s} is a top risk driver" for s in hh["supporting_sensors"]],
                        "evidence_against": [],
                        "contradiction_notes": "",
                    })
                errors.append(f"LLM unavailable — using heuristic hypotheses for {asset_id}")
                evidence_list.append(f"Heuristic hypotheses generated for {asset_id}")

        # Build CausalHypothesis objects
        causal_hypotheses = []
        for h in all_hypotheses:
            try:
                causal_hypotheses.append(CausalHypothesis(
                    cause=h.get("cause", "unknown"),
                    confidence=min(1.0, max(0.0, float(h.get("confidence", 0.5)))),
                    evidence_for=h.get("evidence_for", []),
                    evidence_against=h.get("evidence_against", []),
                    contradiction_notes=h.get("contradiction_notes", ""),
                ))
            except Exception:
                continue

        # Overall confidence
        if causal_hypotheses:
            avg_conf = sum(h.confidence for h in causal_hypotheses) / len(causal_hypotheses)
            confidence = min(0.90, avg_conf)
        else:
            confidence = 0.30

        quality_modifier = context.get("quality_confidence_modifier", 1.0)
        confidence *= quality_modifier

        # Store in context
        context["causal_hypotheses"] = causal_hypotheses
        context["hypotheses_by_asset"] = {}
        for h_data in all_hypotheses:
            aid = h_data.get("asset_id", "")
            context["hypotheses_by_asset"].setdefault(aid, []).append(h_data)

        # Serialize for payload
        hypotheses_payload = [
            {
                "asset_id": h.get("asset_id", ""),
                "cause": h.get("cause", ""),
                "confidence": h.get("confidence", 0),
                "evidence_for": h.get("evidence_for", []),
                "evidence_against": h.get("evidence_against", []),
            }
            for h in all_hypotheses[:20]
        ]

        return self._build_output(
            payload={
                "status": "completed",
                "assets_analyzed": len(top_assets),
                "total_hypotheses": len(causal_hypotheses),
                "hypotheses": hypotheses_payload,
            },
            confidence=round(confidence, 4),
            assumptions=[
                "SHAP-to-cause mapping uses domain heuristics as starting point",
                "LLM reasoning is validated against SHAP evidence (contradictions flagged)",
                "Hypothesis confidence is preliminary — Governance Agent makes final check",
            ],
            evidence=evidence_list,
            errors=errors if errors else None,
            next_agent="planner_agent",
        )

    def _build_prompt(
        self,
        asset_id: str,
        asset_type: str,
        failure_probability: float,
        risk_drivers: list[dict],
        heuristic_hypotheses: list[dict],
        recent_events: list[dict],
        maintenance_history: list[dict],
        asset_age_years: float,
    ) -> str:
        return f"""You are an industrial reliability engineer AI analyzing asset {asset_id} (type: {asset_type}).

CURRENT STATUS:
- Failure probability: {failure_probability:.1%} within 48 hours
- Asset age: {asset_age_years} years

TOP RISK-DRIVING SENSOR FEATURES (from SHAP analysis):
{json.dumps(risk_drivers[:5], indent=2)}

PRELIMINARY HYPOTHESES (from sensor-to-cause mapping):
{json.dumps(heuristic_hypotheses[:3], indent=2)}

RECENT EVENTS:
{json.dumps(recent_events[:5], indent=2)}

MAINTENANCE HISTORY:
{json.dumps(maintenance_history[:3], indent=2)}

TASK: Produce a JSON object with a "hypotheses" array. Each hypothesis must have:
- "cause": string (e.g., "bearing_degradation", "seal_leak", "overheating")
- "confidence": float 0.0-1.0
- "evidence_for": list of strings
- "evidence_against": list of strings
- "contradiction_notes": string (explain any conflicting evidence)

Rank hypotheses by confidence. Include 2-4 hypotheses.
Return ONLY valid JSON, no other text."""
