# ml-backend integration plan

## Objective

Connect the implemented backend workflow to executable ML intelligence in ml/aegis so backend endpoints return real model and agent outputs instead of deterministic fallback payloads.

## Current state summary

### Backend is ready for integration

The backend already has:

- route surface for full incident lifecycle
- request/response contracts
- orchestration entrypoint in pipeline_service
- governance checks
- request tracing and error envelopes
- integration tests for endpoint flow

Key files:

- backend/services/pipeline_service.py
- backend/services/incident_service.py
- backend/contracts.py
- backend/routes/pipeline.py

### ML module is partially ready

Implemented and usable now:

- schemas and typed contracts
- data loaders
- synthetic data generator
- anomaly model
- failure risk model
- explainability module

Partially implemented / placeholder:

- agent orchestrator and most agent modules are docstring-only stubs
- planning constraints/objective/optimizer modules are stubs
- simulation monte_carlo and scenario modules are stubs

Key files:

- ml/aegis/data/schemas.py
- ml/aegis/data/loaders.py
- ml/aegis/data/synthetic_generator.py
- ml/aegis/models/anomaly.py
- ml/aegis/models/failure_risk.py
- ml/aegis/models/explainability.py
- ml/aegis/agents/orchestrator.py (not executable)

## Integration strategy

Use a two-layer integration approach:

1. Add a backend-side ML adapter facade that wraps ml/aegis callable functionality.
2. Keep backend service contracts stable and swap stage implementations behind the facade.

This avoids route churn and allows progressive replacement of fallback logic stage by stage.

## target architecture

- backend/routes -> backend/services/pipeline_service
- pipeline_service -> backend/services/ml_adapter_service (new)
- ml_adapter_service -> ml/aegis models and, later, ml/aegis orchestrator
- backend/storage -> persists outputs and traces

## staged execution plan

### stage 1 - integration contracts and adapter scaffold

Deliverables:

- create backend/services/ml_adapter_service.py
- define typed adapter input/output DTOs in backend/contracts.py
- add adapter interface methods:
  - run_ingest
  - run_risk
  - run_plan
  - run_optimize
  - run_simulate
  - run_report
- wire adapter into pipeline_service via dependency injection

Acceptance criteria:

- pipeline_service can call adapter methods without changing route signatures
- all existing tests still pass

### stage 2 - real risk pipeline (ingest + anomaly + failure probability)

Deliverables:

- map backend ingest payload to telemetry/event/maintenance DataFrame inputs
- call ml/aegis/data/loaders.py where path-based sources are provided
- call anomaly feature + detector flow from ml/aegis/models/anomaly.py
- call failure probability model from ml/aegis/models/failure_risk.py
- return asset-level risk payload in current backend response structure

Acceptance criteria:

- /risk/analyze returns values derived from model execution
- fallback branch only used on explicit model execution failure

### stage 3 - explainability and planning bridge

Deliverables:

- integrate ml explainability outputs from ml/aegis/models/explainability.py
- introduce temporary rule-based planner in backend adapter until planner agent is implemented
- include assumptions/evidence_refs from real model feature contributions

Acceptance criteria:

- /incident/plan uses model evidence for root-cause payload
- payload contains deterministic schema and confidence values

### stage 4 - optimization and simulation engine implementation

Deliverables:

- implement ml/aegis/planning/constraints.py
- implement ml/aegis/planning/objective.py
- implement ml/aegis/planning/optimizer.py
- implement ml/aegis/simulation/monte_carlo.py
- implement ml/aegis/simulation/scenario_engine.py
- connect backend adapter methods run_optimize and run_simulate to these modules

Acceptance criteria:

- /incident/optimize applies real constraint logic and scoring
- /incident/simulate produces uncertainty-aware scenario outputs

### stage 5 - orchestrator activation and backend fallback retirement

Deliverables:

- implement executable class in ml/aegis/agents/orchestrator.py
- implement agent modules required by orchestrator path (minimum viable versions)
- replace pipeline_service._ml_agents_available check with real orchestrator health check
- use adapter to call orchestrator for full /incident/run flow

Acceptance criteria:

- /incident/run can execute full ML path end-to-end
- deterministic fallback only triggers on runtime exceptions, not by default

### stage 6 - storage integration and production hardening

Deliverables:

- add postgres repository implementation in backend/storage
- persist incident stages, confidence trail, governance trail, model metadata
- add model-version + feature-schema metadata to persisted outputs
- add retry/timeouts for long-running ML calls

Acceptance criteria:

- incident lifecycle survives process restart
- reports can be reconstructed from persisted data

## required code changes map

Backend additions/changes:

- backend/services/ml_adapter_service.py (new)
- backend/services/pipeline_service.py (use adapter)
- backend/services/incident_service.py (remove deterministic hardcoded payloads as stages migrate)
- backend/contracts.py (integration DTOs)
- backend/tests/integration/test_api_flow.py (assert model-backed fields)

ML additions/changes:

- ml/aegis/agents/orchestrator.py (implement runtime class)
- ml/aegis/agents/*_agent.py (incremental implementation)
- ml/aegis/planning/constraints.py (implement)
- ml/aegis/planning/objective.py (implement)
- ml/aegis/planning/optimizer.py (implement)
- ml/aegis/simulation/monte_carlo.py (implement)
- ml/aegis/simulation/scenario_engine.py (implement)

## contract mapping

Backend request -> ML input mapping:

- IngestBatchRequest -> telemetry/events/maintenance datasets
- RiskAnalyzeRequest -> latest feature window + lookahead horizon
- IncidentOptimizeRequest -> plan candidates + OptimizationConstraints
- IncidentSimulateRequest -> ranked plans + horizon

ML output -> backend envelope payload:

- anomaly/risk outputs -> payload.risk
- explainability + hypothesis -> payload.plan
- optimizer output -> payload.optimize
- simulation output -> payload.simulate
- reporter output -> payload.report

## testing plan

### unit tests

- adapter method tests with synthetic fixtures
- contract mapping tests for each stage
- governance behavior on low-confidence model outputs

### integration tests

- backend endpoint flow with ML adapter enabled
- fallback behavior when ML adapter raises exceptions

### e2e tests

- /incident/run full path on generated synthetic dataset
- deterministic replay baseline for CI confidence

## risks and mitigations

- risk: orchestrator is currently stub-only
- mitigation: adapter-first rollout allows model-stage integration before full orchestrator

- risk: schema drift between backend contracts and ml schemas
- mitigation: add mapper functions + strict pydantic validation at adapter boundary

- risk: long model latency impacts API responsiveness
- mitigation: move heavy runs to async queue in stage 6 while preserving sync endpoints for quick operations

## done definition for integration

Integration is complete when:

- backend endpoints use real ML computations for risk/plan/optimize/simulate/report
- /incident/run executes orchestrator path successfully
- fallback is exception-only
- integration + e2e tests pass in CI
- persisted incident outputs include model metadata and governance trace
