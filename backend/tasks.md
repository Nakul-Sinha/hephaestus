# backend tasks

This file is the execution loop for completing the backend to the scope defined in context.md.

## Rules For This Loop

- Work by stage only.
- At the end of each stage, create a commit with a conventional prefix:
  - feat
  - fix
  - chore
  - update
- At the end of each stage, write a stage handoff note:
  - stage-1.md
  - stage-2.md
  - ...
- Each stage handoff must include:
  - implemented work
  - issues faced
  - resolution details
  - next-stage guidance for the next LLM pass

## Stage 1 - Foundation API Skeleton

- [x] Create backend API structure for route modules and service layer.
- [x] Add shared response envelope models.
- [x] Add settings and runtime configuration helpers.
- [x] Implement /health endpoint with dependency health status.
- [x] Wire FastAPI app with router registration and startup metadata.
- [x] Create stage-1.md and commit stage.

## Stage 2 - Incident Workflow Endpoints

- [x] Add ingest endpoint and request models.
- [x] Add risk analyze endpoint.
- [x] Add incident plan endpoint.
- [x] Add incident optimize endpoint.
- [x] Add incident simulate endpoint.
- [x] Add incident report endpoint.
- [x] Create in-memory repository for incident lifecycle persistence.
- [x] Create stage-2.md and commit stage.

## Stage 3 - Orchestration + Governance

- [x] Add backend pipeline service that coordinates endpoint flow.
- [x] Add confidence propagation rules.
- [x] Add governance gate checks and warnings.
- [x] Add deterministic fallback behavior when ML agents are unavailable.
- [x] Add audit timeline payload for report responses.
- [x] Create stage-3.md and commit stage.

## Stage 4 - Reliability + Security

- [x] Add API key auth guard for write endpoints (MVP).
- [x] Add request-id middleware and structured logging hooks.
- [x] Add centralized error handling with envelope responses.
- [x] Add input validation edge-case handling and clear warning paths.
- [x] Create stage-4.md and commit stage.

## Stage 5 - Testing + Hardening

- [x] Add unit tests for service and repository logic.
- [x] Add API integration tests for health and incident flow.
- [x] Validate full flow: ingest -> risk -> plan -> optimize -> simulate -> report.
- [x] Update backend/readme.md with final implemented status notes.
- [x] Create stage-5.md and commit stage.

## Definition Of Completion

- [x] All stage checklists complete.
- [x] Stage 1 through Stage 5 commits created.
- [x] Stage handoff files stage-1.md through stage-5.md exist.
- [x] Backend endpoints are functional and return standardized envelopes.