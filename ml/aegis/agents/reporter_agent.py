"""Reporter agent runtime implementation."""

from __future__ import annotations

from ml.aegis.data.schemas import AgentOutput


class ReporterAgent:
  """Emit compact reporting diagnostics from report-stage payload."""

  def run(self, trace_id: str, report_payload: dict) -> AgentOutput:
    playbook = dict(report_payload.get("operator_playbook", {}))
    manager_summary = dict(report_payload.get("manager_summary", {}))
    confidence = float(manager_summary.get("confidence", 0.7))

    return AgentOutput(
      input_context_id=trace_id,
      agent_name="ReporterAgent",
      output_payload={
        "recommended_plan_id": playbook.get("recommended_plan_id", "unknown"),
        "operator_step_count": len(list(playbook.get("steps", []))),
        "audit_trace_length": len(list(report_payload.get("audit_trace", []))),
      },
      confidence_score=max(0.4, min(0.95, confidence)),
      assumptions=["report payload contains complete stage snapshots for stakeholder delivery"],
      evidence_refs=["report-stage"],
      errors=[],
      next_recommended_agent="GovernanceAgent",
      execution_time_ms=0.0,
    )
