"""Runtime orchestrator for executing the MVP agent graph."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Callable
from uuid import uuid4

from ml.aegis.agents.causal_agent import CausalAgent
from ml.aegis.agents.governance_agent import GovernanceAgent
from ml.aegis.agents.intake_agent import IntakeAgent
from ml.aegis.agents.optimizer_agent import OptimizerAgent
from ml.aegis.agents.planner_agent import PlannerAgent
from ml.aegis.agents.prognostics_agent import PrognosticsAgent
from ml.aegis.agents.quality_agent import QualityAgent
from ml.aegis.agents.reporter_agent import ReporterAgent
from ml.aegis.agents.sentinel_agent import SentinelAgent
from ml.aegis.agents.simulation_agent import SimulationAgent
from ml.aegis.data.schemas import AgentOutput, PipelineResult


@dataclass(frozen=True)
class StageExecution:
    """A single stage execution contract used by the orchestrator runtime."""

    stage_name: str
    runner: Callable[[], tuple[dict, float, list[str]]]
    retries: int = 1


class Orchestrator:
    """Execute staged workflow with retry handling and agent trace outputs."""

    def __init__(self, confidence_floor: float = 0.6) -> None:
        self.confidence_floor = confidence_floor
        self.intake_agent = IntakeAgent()
        self.quality_agent = QualityAgent()
        self.sentinel_agent = SentinelAgent()
        self.prognostics_agent = PrognosticsAgent()
        self.causal_agent = CausalAgent()
        self.planner_agent = PlannerAgent()
        self.optimizer_agent = OptimizerAgent()
        self.simulation_agent = SimulationAgent()
        self.reporter_agent = ReporterAgent()
        self.governance_agent = GovernanceAgent(confidence_floor=confidence_floor)

    def run(self, stage_executions: list[StageExecution], incident_id: str | None = None) -> dict:
        """Run stage executions and return stage outputs plus orchestration trace."""
        trace_id = incident_id or f"run-{uuid4().hex[:12]}"
        started = perf_counter()

        stage_payloads: dict[str, dict] = {}
        stage_confidences: list[float] = []
        all_warnings: list[str] = []
        agent_outputs: list[AgentOutput] = []

        for execution in stage_executions:
            payload: dict
            confidence: float
            warnings: list[str]
            last_error: Exception | None = None

            for _ in range(execution.retries + 1):
                try:
                    payload, confidence, warnings = execution.runner()
                    break
                except Exception as exc:
                    last_error = exc
            else:
                raise RuntimeError(f"stage {execution.stage_name} failed after retries: {last_error}") from last_error

            stage_payloads[execution.stage_name] = payload
            stage_confidences.append(confidence)
            all_warnings.extend(warnings)

            agent_outputs.extend(self._agent_outputs_for_stage(trace_id, execution.stage_name, payload))

        governance_output = self.governance_agent.run(trace_id, stage_confidences=stage_confidences, warnings=all_warnings)
        agent_outputs.append(governance_output)

        now = datetime.now(timezone.utc)
        final_plans = self._normalize_final_plans(
            plans=list(stage_payloads.get("plan", {}).get("plans", [])),
            default_confidence=min(stage_confidences) if stage_confidences else 0.6,
        )
        pipeline_result = PipelineResult(
            pipeline_run_id=trace_id,
            started_at=now,
            completed_at=now,
            status="completed",
            agent_outputs=agent_outputs,
            final_plans=final_plans,
            recommended_plan_id=stage_payloads.get("optimize", {}).get("recommended_plan_id"),
            overall_confidence=min(stage_confidences) if stage_confidences else 0.0,
            governance_verdict=governance_output.output_payload.get("verdict"),
            warnings=all_warnings,
            total_duration_ms=round((perf_counter() - started) * 1000.0, 3),
        )

        return {
            "trace_id": trace_id,
            "pipeline_result": pipeline_result.model_dump(mode="json"),
            "stage_payloads": stage_payloads,
            "confidence": min(stage_confidences) if stage_confidences else 0.0,
            "warnings": all_warnings,
        }

    def _agent_outputs_for_stage(self, trace_id: str, stage_name: str, payload: dict) -> list[AgentOutput]:
        if stage_name == "ingest":
            return [
                self.intake_agent.run(trace_id, payload),
                self.quality_agent.run(trace_id, payload),
            ]
        if stage_name == "risk":
            return [
                self.sentinel_agent.run(trace_id, payload),
                self.prognostics_agent.run(trace_id, payload),
            ]
        if stage_name == "plan":
            return [
                self.causal_agent.run(trace_id, payload),
                self.planner_agent.run(trace_id, payload),
            ]
        if stage_name == "optimize":
            return [self.optimizer_agent.run(trace_id, payload)]
        if stage_name == "simulate":
            return [self.simulation_agent.run(trace_id, payload)]
        if stage_name == "report":
            return [self.reporter_agent.run(trace_id, payload)]
        return []

    def _normalize_final_plans(self, plans: list[dict], default_confidence: float) -> list[dict]:
        """Fill optional planner fields so plans satisfy shared schema contracts."""
        normalized: list[dict] = []
        for plan in plans:
            candidate = dict(plan)
            candidate.setdefault("estimated_duration_minutes", int(candidate.get("expected_downtime_minutes", 0)))
            candidate.setdefault("maintenance_window", "any")
            candidate.setdefault("confidence", round(default_confidence, 4))
            candidate.setdefault("assumptions", [])
            candidate.setdefault("rollback_plan", "fallback to manual intervention playbook")
            normalized.append(candidate)
        return normalized
