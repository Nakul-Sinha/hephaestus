"""
Agent package — 10 specialized agents + orchestrator.

Orchestration: Directed graph with conditional routing, retry policies,
circuit-breaker behavior, and human-in-the-loop fallback.

All agents inherit from BaseAgent and emit standardized AgentOutput objects.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

from ml.aegis.data.schemas import AgentOutput


class BaseAgent(ABC):
    """
    Abstract base class for all Hephaestus agents.

    Every agent must:
        1. Inherit from BaseAgent
        2. Set a unique `name` attribute
        3. Implement the `run()` method
        4. Return an AgentOutput via `_build_output()`
    """

    name: str = "base_agent"
    description: str = "Base agent"

    def execute(self, context: dict[str, Any]) -> AgentOutput:
        """Public entry point. Wraps run() with timing and error handling."""
        start = time.perf_counter()
        pipeline_run_id = context.get("pipeline_run_id", str(uuid.uuid4()))

        try:
            output = self.run(context)
            elapsed_ms = (time.perf_counter() - start) * 1000
            output.execution_time_ms = round(elapsed_ms, 2)
            output.input_context_id = pipeline_run_id
            output.agent_name = self.name
            return output

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return AgentOutput(
                input_context_id=pipeline_run_id,
                agent_name=self.name,
                output_payload={"error_type": type(e).__name__, "error_detail": str(e)},
                confidence_score=0.0,
                assumptions=[],
                evidence_refs=[],
                errors=[f"{type(e).__name__}: {e}"],
                next_recommended_agent=None,
                execution_time_ms=round(elapsed_ms, 2),
            )

    @abstractmethod
    def run(self, context: dict[str, Any]) -> AgentOutput:
        """Core agent logic. Subclasses must implement this."""
        ...

    def _build_output(
        self,
        payload: dict[str, Any],
        confidence: float,
        assumptions: list[str] | None = None,
        evidence: list[str] | None = None,
        errors: list[str] | None = None,
        next_agent: str | None = None,
    ) -> AgentOutput:
        """Helper to construct a standardized AgentOutput."""
        return AgentOutput(
            input_context_id="",  # Filled by execute()
            agent_name=self.name,
            output_payload=payload,
            confidence_score=max(0.0, min(1.0, confidence)),
            assumptions=assumptions or [],
            evidence_refs=evidence or [],
            errors=errors or [],
            next_recommended_agent=next_agent,
        )

# Export all agents for easy importing
from ml.aegis.agents.intake_agent import IntakeAgent
from ml.aegis.agents.quality_agent import QualityAgent
from ml.aegis.agents.sentinel_agent import SentinelAgent
from ml.aegis.agents.prognostics_agent import PrognosticsAgent
from ml.aegis.agents.causal_agent import CausalAgent
from ml.aegis.agents.planner_agent import PlannerAgent
from ml.aegis.agents.optimizer_agent import OptimizerAgent
from ml.aegis.agents.simulation_agent import SimulationAgent
from ml.aegis.agents.reporter_agent import ReporterAgent
from ml.aegis.agents.governance_agent import GovernanceAgent
from ml.aegis.agents.orchestrator import Orchestrator

__all__ = [
    "BaseAgent",
    "IntakeAgent",
    "QualityAgent",
    "SentinelAgent",
    "PrognosticsAgent",
    "CausalAgent",
    "PlannerAgent",
    "OptimizerAgent",
    "SimulationAgent",
    "ReporterAgent",
    "GovernanceAgent",
    "Orchestrator",
]
