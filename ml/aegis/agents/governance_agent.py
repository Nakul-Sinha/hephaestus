"""
Governance Agent (Agent 10)

Responsibilities:
    - Final safety gate before output goes to users
    - Confidence floor checks across the pipeline
    - Policy validation (e.g. tier 1 assets require human review)
    - Emits final status: approved, needs_human_review, rejected

Next agent: None (End of Pipeline).
"""

from __future__ import annotations

from typing import Any

from ml.aegis.agents import BaseAgent
from ml.aegis.data.schemas import AgentOutput, GovernanceVerdict


class GovernanceAgent(BaseAgent):

    name = "governance_agent"
    description = "Safety gate and policy enforcement"

    def run(self, context: dict[str, Any]) -> AgentOutput:
        # Context should have an agent_outputs list tracking previous results
        agent_outputs: list[AgentOutput] = context.get("agent_outputs", [])
        assets = context.get("assets", [])
        recommended_plan_id = context.get("recommended_plan_id")
        optimized_plans = context.get("optimized_plans", [])

        evidence: list[str] = []
        violations: list[str] = []
        
        # 1. Confidence Floor Check
        confidence_floor = 0.60
        low_confidence_agents = [
            out.agent_name for out in agent_outputs
            if out.confidence_score < confidence_floor
        ]
        
        if low_confidence_agents:
            violations.append(f"Low confidence (<{confidence_floor}) in agents: {', '.join(low_confidence_agents)}")

        # 2. Extract Recommended Plan
        best_plan = next((p for p in optimized_plans if p.plan_id == recommended_plan_id), None)
        
        if best_plan:
            # 3. Cost Approval Limits
            auto_approve_limit = 10000.0
            if best_plan.estimated_cost > auto_approve_limit:
                violations.append(
                    f"Cost ${best_plan.estimated_cost:,.0f} exceeds auto-approval limit "
                    f"(${auto_approve_limit:,.0f})"
                )
                
            # 4. Criticality Check
            target_asset_id = best_plan.plan_id.split("-")[0]
            asset = next((a for a in assets if a.asset_id == target_asset_id), None)
            
            if asset and asset.criticality_tier.value == 1:
                violations.append(f"Asset {target_asset_id} is Tier 1 (Mission Critical) - mandates human review")
                
            # 5. Downtime limits
            if best_plan.expected_downtime_minutes > 480: # 8 hours
                violations.append("Recommended downtime exceeds 8 hour automatic threshold")

        # Determine Verdict
        if not best_plan and not context.get("quality_summary", {}).get("overall_verdict") == "unreliable":
            # Just healthy fleet, approve the 'all clear'
            verdict = GovernanceVerdict.APPROVED
            evidence.append("No active interventions required. Approved.")
        elif len(violations) > 2:
            verdict = GovernanceVerdict.REJECTED
            evidence.append("Plan rejected due to multiple policy violations.")
        elif len(violations) > 0:
            verdict = GovernanceVerdict.NEEDS_HUMAN_REVIEW
            evidence.append("Flagged for human review based on policy guidelines.")
        else:
            verdict = GovernanceVerdict.APPROVED
            evidence.append("All safety and policy checks passed. Approved for execution.")

        return self._build_output(
            payload={
                "status": "completed",
                "verdict": verdict.value,
                "violations_found": violations,
                "approved_plan_id": recommended_plan_id if verdict == GovernanceVerdict.APPROVED else None
            },
            confidence=1.0, # Governance is absolute
            assumptions=["Policy constraints are up to date"],
            evidence=evidence,
            errors=violations if violations else None,
            next_agent=None,
        )
