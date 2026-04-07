# Hephaestus — ML Module (aegis)

Core intelligence layer: 10 specialized agents, ML models, optimization, simulation, and reporting.

## Structure
- `agents/`       — 10 agents + orchestrator (DAG with conditional routing)
- `data/`         — Schemas, loaders, validators, synthetic data generator
- `models/`       — Anomaly detection, failure risk, RUL, explainability (SHAP)
- `planning/`     — Constraints, objective function, multi-objective optimizer
- `simulation/`   — Monte Carlo engine, scenario comparison, impact estimation
- `reporting/`    — Report templates and natural language composer
- `telemetry/`    — Metrics, tracing, structured logging
- `config/`       — Settings, policy profiles, model hyperparameters
- `tests/`        — Unit, integration, and E2E tests
