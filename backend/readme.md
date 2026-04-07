# hephaestus backend

The backend is the execution and control layer for Hephaestus. It exposes APIs, runs the incident workflow, applies governance checks, and returns auditable decision payloads for the frontend and downstream consumers.

## What the backend currently does

The backend is functional and supports both step-by-step and single-call incident processing.

1. Creates an incident context from ingest input.
2. Produces risk output for the incident.
3. Produces intervention plan options.
4. Optimizes plans under provided constraints.
5. Simulates projected outcomes.
6. Generates report payloads including audit and governance traces.

## Implemented capabilities

- FastAPI application bootstrapped with route registration.
- Standard response envelope for success and error responses.
- Request-id middleware with response header propagation.
- Structured request logging.
- Centralized exception handling that preserves the envelope format.
- API key protection for write endpoints via x-api-key.
- In-memory incident repository with:
  - stage snapshots
  - event timeline
  - confidence trail
  - governance trail
- Incident service for staged workflow operations.
- Pipeline service for full workflow orchestration.
- Governance service with confidence-floor checks and escalation verdicts.
- Deterministic fallback mode when ML runtime is unavailable.

## Implemented API surface

- GET /health
- POST /ingest/batch
- POST /risk/analyze
- POST /incident/plan
- POST /incident/optimize
- POST /incident/simulate
- GET /incident/{id}/report
- POST /incident/run

Standard envelope:

```json
{
  "request_id": "req-9f30f8",
  "status": "success",
  "timestamp": "2026-04-07T11:25:00Z",
  "payload": {},
  "confidence": 0.87,
  "warnings": []
}
```

## Current architecture

- routes/: HTTP handlers only
- services/: business logic and orchestration
- storage/: repository abstraction (current implementation: in-memory)
- config/: settings and dependency status helpers
- models.py and contracts.py: API and workflow contracts
- security.py: API key guard
- tests/: unit and integration coverage

## Stage documentation

All stage handoff notes are now stored in:

- backend/comments/stage-1.md
- backend/comments/stage-2.md
- backend/comments/stage-3.md
- backend/comments/stage-4.md
- backend/comments/stage-5.md

## Test status

Current backend tests pass for:

- service lifecycle behavior
- write endpoint authentication
- staged incident API flow
- single-call pipeline API flow

## Known current limits

- Persistence is in-memory, not PostgreSQL yet.
- Risk/planning/simulation outputs are deterministic fallback logic until full ML orchestration is wired.
- Role-based authorization is not implemented yet.

## Local development

```bash
pip install -r ../ml/requirements.txt
uvicorn backend.app:app --reload
```

## Next backend upgrades

- Replace in-memory storage with PostgreSQL repositories.
- Connect to executable ML orchestrator outputs from ml/aegis agents.
- Add policy profile versioning and richer governance rules.
- Add role-based access and production auth strategy.
