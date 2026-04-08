# ml-backend integration execution tasks

This file is the implementation backlog for integrating ml/aegis with backend.
It is derived from integration.md and converted into concrete, code-level tasks.

## Execution rules

- Complete stages in order.
- Keep backend route contracts stable while replacing internals.
- Do not remove fallback until real ML path is validated.
- After each stage:
  - run tests
  - update stage notes in backend/comments if behavior changed
  - commit with a conventional message

---

## Stage 1 - adapter scaffold and contracts

### Goal

Introduce an ML adapter boundary so backend services can call ML code through one interface.

### Tasks

- [x] Create backend/services/ml_adapter_service.py with methods:
  - run_ingest
  - run_risk
  - run_plan
  - run_optimize
  - run_simulate
  - run_report
- [x] Add adapter DTOs and mapping contracts in backend/contracts.py.
- [x] Wire adapter into backend/services/pipeline_service.py via dependency injection.
- [x] Keep current response shape unchanged.
- [x] Add unit tests for adapter interface smoke behavior.

### Files

- backend/services/ml_adapter_service.py
- backend/contracts.py
- backend/services/pipeline_service.py
- backend/services/__init__.py
- backend/tests/unit/test_incident_service.py

### Validate

- [x] Existing backend integration tests still pass.
- [x] No route signature changes.

### Commit

- feat: add ml adapter scaffold and integration contracts

---

## Stage 2 - real ingest and risk path

### Goal

Replace hardcoded risk generation with model-backed execution.

### Tasks

- [x] Map ingest input to datasets accepted by ml loaders.
- [x] Use ml/aegis/data/loaders.py for path-based input mode.
- [x] Implement in-memory DataFrame mode for direct payload mode.
- [x] Integrate anomaly flow from ml/aegis/models/anomaly.py.
- [x] Integrate failure probability flow from ml/aegis/models/failure_risk.py.
- [x] Persist model outputs and confidence into incident stages.
- [x] Keep deterministic fallback only on runtime exceptions.

### Files

- backend/services/ml_adapter_service.py
- backend/services/incident_service.py
- backend/services/pipeline_service.py
- ml/aegis/models/anomaly.py (only if fixes are required)
- ml/aegis/models/failure_risk.py (only if fixes are required)

### Validate

- [x] /risk/analyze returns computed model values.
- [x] Fallback warning appears only on execution failure.

### Commit

- feat: integrate model-backed ingest and risk analysis

---

## Stage 3 - explainability and planning bridge

### Goal

Use explainability output as evidence and produce planning payload from ML-supported signals.

### Tasks

- [x] Integrate ml/aegis/models/explainability.py in adapter plan path.
- [x] Create temporary deterministic planner bridge using explainability evidence.
- [x] Populate root cause assumptions and evidence_refs from real feature contributions.
- [x] Preserve current plan payload schema for frontend compatibility.

### Files

- backend/services/ml_adapter_service.py
- backend/services/incident_service.py
- ml/aegis/models/explainability.py (if compatibility patches are needed)

### Validate

- [x] /incident/plan includes explainability-derived evidence.
- [x] Plan payload remains schema-compatible with current backend routes.

### Commit

- feat: connect explainability and planning bridge

---

## Stage 4 - optimization and simulation engine activation

### Goal

Implement real optimization and simulation modules and wire them into backend.

### Tasks

- [x] Implement ml/aegis/planning/constraints.py.
- [x] Implement ml/aegis/planning/objective.py.
- [x] Implement ml/aegis/planning/optimizer.py.
- [x] Implement ml/aegis/simulation/monte_carlo.py.
- [x] Implement ml/aegis/simulation/scenario_engine.py.
- [x] Connect adapter run_optimize and run_simulate to these modules.
- [x] Replace hardcoded optimization/simulation payload generation.

### Files

- ml/aegis/planning/constraints.py
- ml/aegis/planning/objective.py
- ml/aegis/planning/optimizer.py
- ml/aegis/simulation/monte_carlo.py
- ml/aegis/simulation/scenario_engine.py
- backend/services/ml_adapter_service.py
- backend/services/incident_service.py

### Validate

- [x] /incident/optimize reflects real constraints and scoring.
- [x] /incident/simulate returns uncertainty-aware outputs.

### Commit

- feat: implement optimization and simulation integration

---

## Stage 5 - orchestrator runtime and full pipeline switch

### Goal

Activate executable orchestrator path and use it for /incident/run.

### Tasks

- [x] Implement runnable Orchestrator class in ml/aegis/agents/orchestrator.py.
- [x] Implement minimum viable runtime logic in required agent modules.
- [x] Replace _ml_agents_available probe with orchestrator health check.
- [x] Route /incident/run through adapter orchestrator execution.
- [x] Keep per-stage endpoints working with same response contracts.

### Files

- ml/aegis/agents/orchestrator.py
- ml/aegis/agents/*_agent.py (minimum required set)
- backend/services/ml_adapter_service.py
- backend/services/pipeline_service.py

### Validate

- [x] /incident/run executes ML path end-to-end.
- [x] Fallback path triggers only on exception.

### Commit

- feat: activate orchestrator-backed full pipeline

---

## Stage 6 - persistence and hardening

### Goal

Move integration to production-safe persistence and runtime behavior.

### Tasks

- [ ] Add postgres repository in backend/storage.
- [ ] Persist stages, confidence trail, governance trail, and model metadata.
- [ ] Add model/version and feature-schema metadata fields.
- [ ] Add retry + timeout handling for heavy ML execution calls.
- [ ] Add performance guardrails and structured metrics for ML calls.

### Files

- backend/storage/repositories.py (new/expanded)
- backend/storage/db.py (new/expanded)
- backend/services/incident_service.py
- backend/services/pipeline_service.py
- backend/services/ml_adapter_service.py

### Validate

- [ ] Incident records survive restart.
- [ ] Report reconstruction works from persisted records.

### Commit

- update: add persistent storage and runtime hardening for ml integration

---

## Test matrix

### Unit

- [ ] Adapter mapping tests.
- [ ] Contract validation tests at adapter boundary.
- [ ] Governance behavior tests on low-confidence outputs.

### Integration

- [ ] Existing backend API flow tests pass.
- [ ] Add tests for model-backed outputs in /risk, /plan, /optimize, /simulate.
- [ ] Add tests for fallback-on-error behavior.

### End-to-end

- [ ] /incident/run full ML path on synthetic dataset.
- [ ] Deterministic replay baseline in CI.

---

## Final done checklist

- [ ] Backend uses real ML outputs for risk, plan, optimize, simulate, report.
- [ ] /incident/run uses orchestrator runtime path.
- [ ] Fallback is exception-only.
- [ ] Integration + e2e tests pass.
- [ ] Metadata and governance traces are persisted.
- [ ] integration.md and backend/readme.md updated to reflect final architecture.
