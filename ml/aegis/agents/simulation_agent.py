"""Simulation agent runtime implementation."""

from __future__ import annotations

from ml.aegis.data.schemas import AgentOutput


class SimulationAgent:
    """Summarize simulation-stage outcomes for downstream reporting."""

    def run(self, trace_id: str, simulate_payload: dict) -> AgentOutput:
        simulations = list(simulate_payload.get("simulations", []))
        best_plan_id = "unknown"
        if simulations:
            best = min(
                simulations,
                key=lambda row: float(row.get("uncertainty", {}).get("risk_p95_end", 1.0)),
            )
            best_plan_id = str(best.get("plan_id", "unknown"))

        confidence = 0.8 if simulations else 0.55
        return AgentOutput(
            input_context_id=trace_id,
            agent_name="SimulationAgent",
            output_payload={
                "scenario_count": len(simulations),
                "best_plan_id": best_plan_id,
                "horizon_days": int(simulate_payload.get("horizon_days", 0)),
            },
            confidence_score=confidence,
            assumptions=["simulation uncertainty bands are sufficient for comparative plan selection"],
            evidence_refs=["simulate-stage", "ml.aegis.simulation.scenario_engine"],
            errors=[],
            next_recommended_agent="ReporterAgent",
            execution_time_ms=0.0,
        )
