# CONTEXT.MD

## Master Build Context For The Project

This document is the single source of truth for building the project in this repository.

All AI coding agents and human contributors must read this file before writing code.

---

## 1. Project Identity

### 1.1 Project Name
hephaestus

### 1.2 One-Line Definition
A software-only, multi-agent AI reliability commander that predicts failures, explains root causes, simulates interventions, and auto-generates optimal maintenance decisions.

### 1.3 Core Theme
Agentic AI for operations intelligence and predictive maintenance.

### 1.4 Hardware Requirement
No hardware is required for the hackathon or development phase.

The system must work entirely on:
- synthetic data
- historical CSV/JSON datasets
- optional replay streams

Future real-world integration with hardware and IoT is optional and out-of-scope for initial build.

---

## 2. Why This Project Exists

### 2.1 Problem
Most maintenance systems are reactive or shallowly predictive:
- they generate alerts but do not decide actions
- they cannot explain likely root cause with confidence
- they do not optimize maintenance plans under cost and downtime constraints
- they do not provide operator-ready incident playbooks

### 2.2 Opportunity
We can build a software-first system where agents collaborate as an operations brain:
- detect abnormal behavior early
- estimate failure probability and remaining useful life
- infer likely cause chain
- compare intervention options
- recommend the best action plan with justification and projected impact

### 2.3 Success Definition
The system must demonstrate end-to-end intelligence:
1. Ingest data.
2. Detect risk.
3. Explain risk.
4. Propose plans.
5. Optimize plan.
6. Simulate expected outcome.
7. Generate actionable report.

---

## 3. Product Scope

### 3.1 In-Scope
- Multi-agent orchestration for reliability operations.
- Time-series anomaly detection and failure risk scoring.
- Remaining Useful Life (RUL) or failure horizon estimation.
- Root cause analysis with confidence.
- Maintenance policy planning and optimization.
- What-if simulation and projected business impact.
- Human-readable incident and maintenance report generation.
- REST API and internal agent event bus.
- Web dashboard for monitoring and decisions.

### 3.2 Out-of-Scope For Initial Build
- Live PLC/SCADA integration.
- Hard real-time controls.
- Robotics actuation.
- Edge deployment on embedded devices.
- Full enterprise IAM and SSO.

---

## 4. Target Users

### 4.1 Primary Users
- Operations Manager
- Reliability Engineer
- Maintenance Planner
- Data Analyst

### 4.2 Secondary Users
- Plant Head / Executive stakeholders
- Audit and compliance reviewers

### 4.3 User Outcomes
- Know what is likely to fail and when.
- Understand why the system thinks so.
- Receive ranked intervention options.
- Choose or auto-approve optimal plan.
- Track avoided downtime and cost savings.

---

## 5. Product Capabilities

### 5.1 Capability Matrix

1. Data Ingestion
- Batch ingest from CSV, parquet, JSON.
- Replay simulated streams.
- Validate schema and timestamp integrity.

2. Data Quality and Drift Monitoring
- Missingness checks.
- Sensor freeze detection.
- Outlier burst detection.
- Feature drift and concept drift signals.

3. Risk Detection
- Unsupervised anomaly score per asset.
- Event-level risk score.
- Dynamic thresholding by asset class.

4. Prognostics
- Short-term failure probability forecast.
- Optional RUL estimate where labels exist.

5. Causal Reasoning
- Hypothesis graph for probable causes.
- Confidence and evidence weighting.
- Contradiction handling.

6. Plan Synthesis
- Generate intervention options.
- Include required parts, skills, and maintenance windows.

7. Optimization
- Multi-objective plan ranking:
  - minimize risk
  - minimize cost
  - minimize downtime
  - satisfy SLA constraints

8. Simulation
- Counterfactual compare of selected plans.
- Risk reduction forecast and uncertainty intervals.

9. Reporting
- Operator report, manager summary, and audit trace.
- Structured JSON + natural language.

---

## 6. Agentic System Design

## 6.1 Agent Roles

1. Intake Agent
- Parses incoming data packages.
- Standardizes schema and units.

2. Quality Agent
- Runs quality checks and drift diagnostics.
- Flags unreliable input channels.

3. Sentinel Agent
- Performs anomaly detection and regime change detection.

4. Prognostics Agent
- Estimates failure likelihood and optional RUL.

5. Causal Agent
- Produces ranked root-cause hypotheses.
- Correlates telemetry, events, and maintenance history.

6. Planner Agent
- Creates candidate intervention plans.

7. Optimizer Agent
- Scores and ranks plans under constraints and policies.

8. Simulation Agent
- Runs what-if outcomes for each candidate plan.

9. Reporter Agent
- Converts technical output to stakeholder-specific summaries.

10. Governance Agent
- Checks policy constraints, confidence floor, and fallback rules.

### 6.2 Agent Orchestration Model
- Directed graph orchestration with conditional routing.
- Retry policies per agent.
- Circuit-breaker behavior for low-confidence inputs.
- Human-in-the-loop fallback on policy violations.

### 6.3 Agent Communication Contract
Each agent emits:
- input_context_id
- output_payload
- confidence_score
- assumptions
- evidence_refs
- errors
- next_recommended_agent

---

## 7. End-to-End Pipeline

1. Ingest data batch or stream window.
2. Validate and enrich data.
3. Detect anomalies and risk spikes.
4. Forecast failure probability and horizon.
5. Infer probable root causes.
6. Generate intervention candidates.
7. Optimize under cost and downtime constraints.
8. Simulate expected outcomes.
9. Generate decision package and report.
10. Persist traces, metrics, and recommendations.

---

## 8. Data Strategy

### 8.1 Core Data Entities

1. Asset Master
- asset_id
- asset_type
- site_id
- installation_date
- maintenance_policy
- criticality_tier

2. Telemetry Stream
- timestamp
- asset_id
- sensor_name
- sensor_value
- unit
- quality_flag

3. Event Log
- event_id
- timestamp
- asset_id
- event_type
- severity
- event_text

4. Maintenance Log
- work_order_id
- asset_id
- action_type
- parts_used
- duration_minutes
- cost
- outcome

5. Failure Ground Truth
- asset_id
- failure_time
- failure_mode
- impact_cost
- downtime_minutes

### 8.2 Synthetic Data Requirements
- Minimum 3 asset classes.
- Minimum 200 assets for realistic fleet behavior.
- Sensor drift, sudden spikes, and gradual degradation patterns.
- Inject at least 4 failure modes with overlap.
- Include noisy and missing channels.

### 8.3 Data Volume Targets (Hackathon)
- 1M+ telemetry rows generated/replayed.
- 6-12 months equivalent historical horizon.

---

## 9. Model Strategy

### 9.1 Baseline Models
- Anomaly: Isolation Forest and robust z-score hybrid.
- Failure Probability: Gradient boosting classifier.
- Optional RUL: survival regression or sequence model baseline.

### 9.2 LLM Role
LLM is not the anomaly model.

LLM is used for:
- plan synthesis
- explanation generation
- policy reasoning
- report composition

### 9.3 Explainability
- SHAP or feature contribution summary for predictive model outputs.
- Root-cause confidence decomposition.

### 9.4 Uncertainty Handling
- Confidence intervals for risk and simulated impact.
- Abstain and escalate when uncertainty exceeds threshold.

---

## 10. Decision and Optimization Logic

### 10.1 Objective Function
Optimize a weighted objective:

score = w1 * risk_reduction - w2 * cost - w3 * downtime + w4 * sla_compliance

Weights are configurable by policy profiles.

### 10.2 Hard Constraints
- Budget ceiling per planning window.
- Crew and skill availability.
- Spare parts inventory constraints.
- Maintenance blackout windows.

### 10.3 Plan Output Schema
- plan_id
- recommended_action
- predicted_risk_reduction
- estimated_cost
- expected_downtime
- confidence
- assumptions
- rollback_plan

---

## 11. Technical Architecture

### 11.1 High-Level Components
- API Layer
- Agent Orchestrator
- Model Service Layer
- Data Processing Layer
- Persistence Layer
- UI Layer
- Observability Layer

### 11.2 Runtime Pattern
- Synchronous API for user-triggered workflows.
- Asynchronous job queue for heavy model runs and simulation batches.

### 11.3 Persistence
- PostgreSQL for metadata and decisions.
- Time-series storage in PostgreSQL partitions or TimescaleDB extension (optional).
- Object storage for model artifacts and simulation outputs.

---

## 12. Tech Stack

### 12.1 Language and Core Frameworks
- Python 3.11+
- FastAPI
- Pydantic v2
- Uvicorn

### 12.2 Data and ML
- pandas
- numpy
- scikit-learn
- xgboost or lightgbm
- scipy
- shap

### 12.3 Agentic and LLM Layer
- LangGraph or custom agent graph orchestrator
- Primary LLM: Ollama local inference (e.g., Mistral, Llama)
- Fallback LLM: Google Gemini API (when local inference is unavailable or under load)
- Structured tool-calling with JSON schemas

### 12.4 Storage and Messaging
- PostgreSQL
- Redis (cache + job queue)
- Celery or RQ for async tasks

### 12.5 Frontend
- Next.js (App Router)
- React
- TypeScript
- Tailwind CSS
- Charting library (ECharts or Recharts)

### 12.6 Dev and Ops
- Docker + Docker Compose
- pytest
- Ruff + mypy
- GitHub Actions for CI

### 12.7 Optional Enhancements
- MLflow for experiment tracking
- Prefect for pipeline orchestration
- DuckDB for local analytics workloads

---

## 13. Repository Implementation Plan

This project is built in the existing repository and can leverage patterns from the current `ml/` module.

### 13.1 Reuse From Existing `ml/`
- Config patterns from `ml/config/settings.py`
- Agent loop patterns from `ml/agent/agent_loop.py`
- Memory persistence concept from `ml/memory/memory.py`
- Verification and retry patterns from `ml/agent/verification.py`

### 13.2 New Module Layout To Build

```text
ml/
  aegis/
    __init__.py
    api/
      app.py
      routes/
        health.py
        ingest.py
        risk.py
        planning.py
        simulation.py
        reports.py
    agents/
      orchestrator.py
      intake_agent.py
      quality_agent.py
      sentinel_agent.py
      prognostics_agent.py
      causal_agent.py
      planner_agent.py
      optimizer_agent.py
      simulation_agent.py
      reporter_agent.py
      governance_agent.py
    data/
      schemas.py
      loaders.py
      validators.py
      synthetic_generator.py
      feature_store.py
    models/
      anomaly.py
      failure_risk.py
      rul.py
      explainability.py
    planning/
      constraints.py
      objective.py
      optimizer.py
    simulation/
      scenario_engine.py
      monte_carlo.py
      impact_estimator.py
    storage/
      db.py
      repositories.py
      migrations/
    telemetry/
      metrics.py
      tracing.py
      logging.py
    reporting/
      templates.py
      composer.py
    tests/
      unit/
      integration/
      e2e/
```

---

## 14. API Contract (Initial)

### 14.1 Endpoints

1. `POST /ingest/batch`
- Uploads telemetry/events/maintenance dataset.

2. `POST /risk/analyze`
- Runs anomaly + failure risk computation.

3. `POST /incident/plan`
- Produces causal explanation + candidate plans.

4. `POST /incident/optimize`
- Returns ranked intervention plan under constraints.

5. `POST /incident/simulate`
- Runs what-if and returns projected outcomes.

6. `GET /incident/{id}/report`
- Returns final report in JSON and markdown.

7. `GET /health`
- Health and dependency status.

### 14.2 Response Standard
All responses include:
- request_id
- status
- timestamp
- payload
- confidence
- warnings

---

## 15. UI Requirements

### 15.1 Screens

1. Fleet Overview
- Asset risk heatmap.
- Trend of anomaly score by asset class.

2. Incident Workbench
- Root-cause graph.
- Candidate action plans.
- Constraint and policy panel.

3. Simulation Panel
- Compare plan A/B/C.
- Risk, cost, downtime projected curves.

4. Report and Audit View
- Timeline of decisions.
- Agent-by-agent trace and confidence.

### 15.2 UX Goals
- Fast interpretation under pressure.
- Confidence and uncertainty visible everywhere.
- One-click export for stakeholders.

---

## 16. Evaluation Framework

### 16.1 Predictive Metrics
- ROC-AUC
- PR-AUC
- F1 on failure class
- mean lead time to failure alert

### 16.2 Planning Metrics
- predicted risk reduction
- expected downtime reduction
- cost efficiency index

### 16.3 Agent Quality Metrics
- plan acceptance rate
- hallucination/error rate in explanations
- policy violation rate

### 16.4 Business Metrics (Demo)
- downtime avoided versus reactive baseline
- maintenance cost saved versus fixed-schedule baseline
- false alarm reduction

---

## 17. Testing Strategy

### 17.1 Unit Tests
- Data validators
- Model wrappers
- objective function correctness

### 17.2 Integration Tests
- agent handoff correctness
- API + DB + queue integration

### 17.3 End-to-End Tests
- synthetic incident scenario from ingest to final report

### 17.4 Non-Functional Tests
- latency under concurrent incidents
- retry/failure handling
- deterministic replay tests

---

## 18. Reliability, Safety, and Governance

### 18.1 Reliability Controls
- Retries with exponential backoff.
- Graceful degradation when an agent fails.
- fallback heuristics when model confidence is low.

### 18.2 Safety Controls
- No automatic irreversible actions in MVP.
- Human approval gate for high-impact recommendations.

### 18.3 Governance
- Full decision trace logs.
- Assumptions and evidence fields mandatory.
- versioned policy profiles.

---

## 19. Security and Privacy

### 19.1 Security
- API key authentication for MVP.
- Role-based permissions for admin/operator in phase 2.
- Secrets via environment variables only.

### 19.2 Privacy
- No personal data required.
- if any user data appears in logs, redact by policy.

---

## 20. Build Phases

### Phase 0: Foundation (Day 1)
- scaffold modules
- setup DB and API skeleton
- add synthetic data generator

### Phase 1: Baseline Intelligence (Day 2-3)
- anomaly + failure risk models
- basic incident object and API flow

### Phase 2: Agentic Flow (Day 3-4)
- implement orchestrator and core agents
- handoff contracts and confidence logic

### Phase 3: Optimization + Simulation (Day 4-5)
- implement objective and constraints
- simulation engine with uncertainty bands

### Phase 4: Reporting + UI (Day 5-6)
- dashboard views
- report generation and export

### Phase 5: Hardening and Demo (Day 6-7)
- tests, instrumentation, polish, benchmark runs

---

## 21. Demo Scenario Blueprint

### 21.1 Setup
- 200 assets across 3 classes.
- Introduce hidden degradation in one class.

### 21.2 Live Narrative
1. Risk spike appears for a critical asset group.
2. Agent system identifies probable failure in next 24-48 hours.
3. Causal graph explains top contributors.
4. Three intervention plans generated.
5. Optimizer selects best plan under budget and downtime constraints.
6. Simulation shows risk and cost outcomes.
7. Final maintenance playbook exported.

### 21.3 What Judges Should See
- Not just detection.
- End-to-end autonomous reasoning and planning.
- explainable and measurable impact.

---

## 22. Definition of Done

Project is considered MVP-complete when:
- data ingest works with schema validation
- anomaly and risk models produce stable outputs
- multi-agent pipeline runs end-to-end
- optimization selects ranked plans under constraints
- simulation compares alternatives with uncertainty
- report generation creates actionable outputs
- UI displays incident lifecycle and key metrics
- end-to-end tests pass on synthetic dataset

---

## 23. Implementation Instructions For AI Coding Agents

1. Always read this file first.
2. Build in small vertical slices with tests.
3. Do not skip schema and contract definitions.
4. Treat confidence and uncertainty as first-class outputs.
5. Keep all agent outputs structured and serializable.
6. Add logging and trace IDs to all major operations.
7. Prefer deterministic baselines before advanced models.
8. Keep interfaces stable and versioned.
9. Avoid tight coupling between agents.
10. Update this file when architecture changes.

---

## 24. Immediate Next Build Tasks

1. Create `ml/aegis/` package scaffold.
2. Add `api/app.py` and `/health` endpoint.
3. Add schema models for asset, telemetry, event, maintenance.
4. Build synthetic data generator.
5. Add baseline anomaly and risk model wrappers.
6. Implement first orchestrator path:
   ingest -> detect -> plan -> report.
7. Add integration test for the full path.

---

## 25. Notes

- This is software-only by design.
- Hardware integration can be added later using connectors.
- The MVP must prove reliable autonomous decision support using data.

End of file.
