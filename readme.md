<div align="center">
  <h1>⚒️ Hephaestus</h1>
  <p><strong>Multi-Agent AI Reliability Commander</strong></p>
  <p><em>The AI that doesn't just tell you something broke it tells you what will break, why, what to do about it, and proves it with numbers.</em></p>
</div>

---

## 📖 Overview

Most modern maintenance operations rely heavily on either slow reactionary metrics (fixing it when it breaks) or blind-scheduled preventive tasks (costly and inefficient). Even modern predictive maintenance platforms stop at "anomaly detected," leaving you guessing why, and what actions you should take next.

**Hephaestus** is an end-to-end, multi-agent artificial intelligence predictive operations brain. It closes the full operational loop autonomously by using 10 specialized intelligent agents configured in a directed graph. 

Hephaestus tracks telemetry data to predict equipment failure, constructs an evidence-backed hypothesis root cause, builds a human-readable maintenance playbook, dynamically optimizes resource planning constraints (crew, parts, budgets), and mathematically simulates projected cost-savings prior to human intervention.

---

## ✨ System Capabilities

* **High-Precision Risk Detection:** Uses machine-learning models (Isolation Forests, Gradient Boosting) to compute unsupervised anomalies and failure probabilities (RUL).
* **Causal Reasoning:** Not just black-box predictions. Hephaestus generates hypothesis graphs with detailed certainty levels using SHAP explainability.
* **Auto-Generated Intervention Options:** Predicts the skills, parts, and downtime requirements for several maintenance scenarios.
* **Mathematical Optimization:** Balances options against your operational constraints via a multi-objective scoring function (Cost / Downtime / SLA Compliance).
* **Monte Carlo Simulation Projector:** Projects hypothetical 30-day trajectories and risk curves with uncertainty bounds.
* **Stakeholder Reporting:** Dynamically writes step-by-step playbooks for operators, financial summaries for managers, and step-by-step decision audit trails for compliance.

---

## 🧠 The 10-Agent Pipeline

Hephaestus coordinates 10 autonomous software agents. Each uses different models suited to their specialty (ML models for tracking telemetry, LLMs for reasoning and synthesis, mathematical solvers for simulations):

1. **Intake Agent:** Parses and standardizes incoming datasets.
2. **Quality Agent:** Checks for feature drift, data missingness, or frozen sensors.
3. **Sentinel Agent:** Analyzes the telemetry window and flags anomalous behaviour.
4. **Prognostics Agent:** Estimates precise failure probability risk horizons.
5. **Causal Agent:** Examines maintenance history to correlate root-cause reasons.
6. **Planner Agent:** Creates intervention schedules (parts/skills dependencies).
7. **Optimizer Agent:** Adjusts plans algorithmically inside hard constraints (e.g. budgets).
8. **Simulation Agent:** Verifies expected outcomes vs. "Doing nothing".
9. **Reporter Agent:** Converts output payload intelligence to readable summaries. 
10. **Governance Agent:** Final safety check before pushing human-in-the-loop playbooks.

---

## 📂 Repository Architecture

This is a monorepo partitioned into three core domain layers:

```text
hephaestus/
├── frontend/             # Next.js / TypeScript / React / Tailwind dashboard app
├── backend/              # FastAPI / Python backend hosting the primary generic REST routes 
└── ml/                   # The Core intelligence module (`aegis`) 
    └── aegis/
        ├── agents/       # Holds all 10 specialized Python agents + the Orchestrator
        ├── config/       # Decision weighting values, hyperparams and security profiles
        ├── data/         # Schemas, real-time validators and synthetic asset generation
        ├── models/       # Predictive ML models (anomaly, failure risk, survival regression)
        ├── planning/     # Engine housing hard/soft constraints & mathematical solvers
        ├── reporting/    # Final composition templates for output summaries
        ├── simulation/   # Scenario engine logic and Monte Carlo calculations
        ├── storage/      # Specialized layer interacting with DB (PostgreSQL internals)
        ├── telemetry/    # Tracing, scoring metrics on agent behaviour and system logs
        └── tests/        # Segmented suite (Unit, Integration, E2E)
```

---

## 💻 Tech Stack

- **Core & Backend API:** Python 3.11+, FastAPI, Pydantic v2, Uvicorn, Celery/Redis
- **Artificial Intelligence Framework:** LangGraph (Orchestration), Scikit-Learn (Isolation Forest), XGBoost/LightGBM, Survival regression, SHAP (Explainability features), Ollama local inference (Primary) / Google Gemini API (Fallback).
- **Solver & Simulation:** Numpy / Scipy
- **Frontend Dashboard:** Next.js (App Router), React, TypeScript, Tailwind CSS, ECharts/Recharts
- **Storage / Database:** PostgreSQL, Redis, optionally TimescaleDB
- **Development / DevOps:** Docker + Docker Compose, GitHub Actions, Pytest, Ruff / Mypy

---

## Naming Convention Policy

- Repository docs/config files should use kebab-case when multi-word naming is needed.
- API route paths use kebab-case segment style only when segments are multi-word.
- Python modules remain snake_case by design to preserve valid imports and tooling compatibility.



## 🚀 Getting Started

*(Development Guide coming soon once modules are initialized)*


---


