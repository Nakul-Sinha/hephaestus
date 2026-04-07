"""
Agent Orchestrator

Manages the Directed Acyclic Graph (DAG) of the 10 Hephaestus agents.
Handles routing, retries, and context propagation.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

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

from ml.aegis.data.schemas import AgentOutput, PipelineResult, GovernanceVerdict


class Orchestrator:
    """End-to-End Pipeline Orchestrator."""

    def __init__(self, models: dict[str, Any] | None = None):
        """
        Initialize orchestrator with optional pre-trained models.
        """
        self.models = models or {}
        
        # Instantiate the agents
        self.agents = {
            "intake_agent": IntakeAgent(),
            "quality_agent": QualityAgent(),
            "sentinel_agent": SentinelAgent(self.models.get("anomaly")),
            "prognostics_agent": PrognosticsAgent(self.models.get("failure_risk")),
            "causal_agent": CausalAgent(),
            "planner_agent": PlannerAgent(),
            "optimizer_agent": OptimizerAgent(),
            "simulation_agent": SimulationAgent(),
            "reporter_agent": ReporterAgent(),
            "governance_agent": GovernanceAgent(),
        }

    def run_pipeline(
        self,
        data_source: str | dict | None = None,
        context_override: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """
        Run the full Hephaestus intelligence pipeline.
        
        Args:
            data_source: Path to telemetry data or a dict of DataFrames.
            context_override: Pre-populated context data (used in tests/demos).
            
        Returns:
            PipelineResult containing traces, verdicts, and final reports.
        """
        pipeline_run_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Initialize context
        context = {
            "pipeline_run_id": pipeline_run_id,
            "data_source": data_source,
            "agent_outputs": []
        }
        
        if context_override:
            context.update(context_override)
            
        agent_outputs: list[AgentOutput] = []
        current_agent_name = "intake_agent"
        
        # Main execution loop (DAG traversal)
        while current_agent_name:
            agent = self.agents.get(current_agent_name)
            if not agent:
                print(f"[Orchestrator] Error: Agent '{current_agent_name}' not found.")
                break
                
            # Execute with basic retry
            output = None
            max_retries = 2
            
            for attempt in range(max_retries):
                try:
                    output = agent.execute(context)
                    break 
                except Exception as e:
                    if attempt == max_retries - 1:
                        output = AgentOutput(
                            input_context_id=pipeline_run_id,
                            agent_name=current_agent_name,
                            output_payload={"status": "failed_critical"},
                            confidence_score=0.0,
                            assumptions=[],
                            evidence_refs=[],
                            errors=[f"Agent completely failed: {e}"],
                            next_recommended_agent="reporter_agent", # Escalate early on total failure
                        )
                    time.sleep(0.5) # simple backoff
            
            # Record output
            agent_outputs.append(output)
            context["agent_outputs"] = agent_outputs
            
            # Print trace for console visibility
            print(f"[{pipeline_run_id[:8]}] {current_agent_name.ljust(20)} | "
                  f"Conf: {output.confidence_score:.2f} | "
                  f"Time: {output.execution_time_ms}ms")
            
            # Determine routing
            if output.output_payload.get("status") == "failed_critical" and current_agent_name != "reporter_agent":
                current_agent_name = "reporter_agent"
            else:
                current_agent_name = output.next_recommended_agent
                
        # Pipeline finished. Construct PipelineResult
        end_time = time.time()
        total_time_ms = round((end_time - start_time) * 1000, 2)
        
        # Extract verdict from governance agent
        gov_output = next((o for o in agent_outputs if o.agent_name == "governance_agent"), None)
        final_verdict = GovernanceVerdict.NEEDS_HUMAN_REVIEW
        if gov_output:
            raw_verdict = gov_output.output_payload.get("verdict")
            try:
                final_verdict = GovernanceVerdict(raw_verdict)
            except Exception:
                pass
                
        # Extract reporter outputs
        rep_output = next((o for o in agent_outputs if o.agent_name == "reporter_agent"), None)
        reports = {}
        if rep_output:
            reports = rep_output.output_payload.get("reports", {})

        return PipelineResult(
            pipeline_run_id=pipeline_run_id,
            timestamp=datetime.now(),
            total_execution_time_ms=total_time_ms,
            agent_traces=agent_outputs,
            final_verdict=final_verdict,
            recommended_plan=context.get("recommended_plan_id"),
            causal_hypotheses=context.get("causal_hypotheses", []),
            generated_reports=reports,
        )
