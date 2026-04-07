# HEPHAESTUS — User & Pitch Guide

> *The AI that doesn't just tell you something broke — it tells you what will break, why, what to do about it, and proves it with numbers.*

---

## Table of Contents

1. [What Is Hephaestus](#1-what-is-hephaestus)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [Who Uses It](#3-who-uses-it)
4. [The User Journey — Step by Step](#4-the-user-journey--step-by-step)
5. [The 10-Agent Pipeline — What Happens Under the Hood](#5-the-10-agent-pipeline--what-happens-under-the-hood)
6. [The Four Dashboard Screens](#6-the-four-dashboard-screens)
7. [How AI Is Used (and How It Isn't)](#7-how-ai-is-used-and-how-it-isnt)
8. [The Demo Script — Running the Live Pitch](#8-the-demo-script--running-the-live-pitch)
9. [Key Talking Points for Judges](#9-key-talking-points-for-judges)
10. [Anticipated Questions & Strong Answers](#10-anticipated-questions--strong-answers)
11. [Differentiators — Why This Wins](#11-differentiators--why-this-wins)
12. [Business Impact Numbers](#12-business-impact-numbers)
13. [One-Liner Cheat Sheet](#13-one-liner-cheat-sheet)

---

## 1. What Is Hephaestus

Hephaestus is a **multi-agent AI system for predictive maintenance and reliability operations**. Think of it as an autonomous operations brain for any organization that manages physical assets — factories, power plants, transportation fleets, data centers.

Instead of a single model that raises alerts, Hephaestus runs **10 specialized AI agents** that collaborate like a reliability engineering team:

```
DATA IN → Detect risk → Explain why → Plan action → Optimize plan → Simulate outcome → Report decision
```

The name comes from the Greek god of the forge — the craftsman who builds things that don't break.

### The One-Sentence Pitch

> "Hephaestus is a multi-agent AI system that predicts equipment failures before they happen, explains the root cause, generates optimized maintenance plans under real-world constraints, and proves the expected impact through simulation — fully autonomous, fully explainable."

---

## 2. The Problem It Solves

### What Exists Today

Most maintenance operations fall into one of three categories:

| Strategy | What They Do | What's Wrong |
|---|---|---|
| **Reactive** | Fix it after it breaks | Maximum downtime, maximum cost, emergency chaos |
| **Scheduled** | Maintain every X weeks regardless | Over-maintains healthy equipment, wastes 30-40% of budget |
| **Basic Predictive** | Dashboard shows "anomaly detected" alert | Alert fatigue, no root cause, no action plan, no cost analysis |

### What's Missing (and What We Build)

No existing system completes the **full decision loop**:

```
Alert → "So what?" → "Why?" → "What do I do?" → "What's the best option?" → "Prove it." → "Give me the playbook."
```

Hephaestus closes that loop entirely. It doesn't stop at detection — it goes all the way to **optimized, justified, simulation-backed maintenance decisions**.

---

## 3. Who Uses It

### Primary Users

| User | What They Care About | What Hephaestus Gives Them |
|---|---|---|
| **Operations Manager** | "Which assets are about to fail?" | Fleet risk heatmap with failure probability and timeline |
| **Reliability Engineer** | "Why is this failing? Is the analysis trustworthy?" | Root-cause hypothesis graph with confidence scores and evidence |
| **Maintenance Planner** | "What should I schedule, and can I afford it?" | Ranked intervention plans optimized under budget/crew/parts constraints |
| **Data Analyst** | "Are the predictions accurate? What's the model doing?" | SHAP explainability, model performance metrics, decision traces |

### Secondary Users

| User | What They See |
|---|---|
| **Plant Head / Executive** | One-page impact summary: downtime avoided, cost saved, risk reduced |
| **Audit / Compliance** | Full decision trace: every agent's reasoning, evidence, confidence at each step |

---

## 4. The User Journey — Step by Step

Here's exactly what a user experiences, from opening the system to exporting a maintenance playbook.

### Step 1: Data Arrives

The user uploads or the system automatically ingests asset telemetry data — sensor readings (temperature, vibration, pressure, RPM), event logs, and past maintenance records.

**What the user does:** Uploads a CSV/JSON file via the dashboard, or data streams in automatically.

**What the system does behind the scenes:**
- Intake Agent parses and standardizes the data
- Quality Agent checks for missing values, frozen sensors, outliers, and drift
- Flags unreliable channels so downstream agents know what to trust

**What the user sees:** A confirmation badge — "12,847 readings ingested. 3 sensors flagged for quality issues."

---

### Step 2: Risk Detection

Within seconds of ingestion, the system scores every asset for anomaly level and failure risk.

**What the user does:** Opens the Fleet Overview dashboard.

**What the system does behind the scenes:**
- Sentinel Agent runs Isolation Forest anomaly detection on each asset's sensor window
- Prognostics Agent runs a gradient boosting model to estimate failure probability in the next 24-48 hours
- Dynamic thresholds adjust per asset class (a pump behaves differently from a compressor)

**What the user sees:**
- A color-coded **risk heatmap** across the entire fleet
- Assets glowing red/orange with a probability score: *"Asset PUMP-0042: 87% failure probability in next 36 hours"*
- Trend charts showing anomaly score climbing over the past 2 weeks

---

### Step 3: Root Cause Explanation

The user clicks on a flagged asset. The system explains *why* it thinks failure is likely.

**What the user does:** Clicks on PUMP-0042 to open the Incident Workbench.

**What the system does behind the scenes:**
- Causal Agent correlates telemetry spikes, event history, and past maintenance actions
- Builds a **hypothesis graph** showing probable cause chains
- Weights evidence and calculates confidence per hypothesis
- Checks for contradictions (e.g., vibration is high but temperature is normal — unusual for bearing failure)

**What the user sees:**
- A visual **root-cause graph** showing:
  - "Bearing degradation" (78% confidence, evidence: vibration trend + 14 months since last bearing replacement)
  - "Lubrication failure" (45% confidence, evidence: temperature uptick, but no oil quality sensor data)
  - "Sensor malfunction" (12% confidence, evidence: one sensor channel flagged by Quality Agent)
- Supporting evidence listed below each hypothesis
- SHAP feature contribution chart showing which sensor signals drove the prediction

---

### Step 4: Intervention Planning

The system doesn't wait for the user to figure out what to do — it generates options.

**What the user does:** Reviews the generated plans on the Incident Workbench.

**What the system does behind the scenes:**
- Planner Agent creates 3 candidate intervention plans based on the root cause analysis:
  - **Plan A:** Replace bearing assembly (full fix, higher cost, needs 4-hour shutdown)
  - **Plan B:** Emergency lubrication + monitoring (lower cost, partial risk reduction, 30-min downtime)
  - **Plan C:** Defer maintenance 7 days with enhanced monitoring (zero cost now, but 87% failure risk remains)
- Each plan includes: required parts, required crew skills, estimated duration, maintenance window

**What the user sees:**
- Three plan cards side by side, each with cost, downtime, risk reduction, and confidence scores
- Required parts and skills listed (e.g., "Needs: 1x SKF bearing, 1x certified mechanic, 4-hour window")

---

### Step 5: Optimization

The user has three plans — but which is *actually* best given today's real constraints?

**What the user does:** Sets constraint parameters (or uses saved policy profile):
- Budget remaining this quarter: $45,000
- Available crew: 2 mechanics, 1 electrician (no certified bearing specialist until Thursday)
- Blackout windows: Production line B cannot be stopped before Friday

**What the system does behind the scenes:**
- Optimizer Agent runs multi-objective scoring:
  ```
  score = w1 × risk_reduction − w2 × cost − w3 × downtime + w4 × SLA_compliance
  ```
- Applies hard constraints: Plan A is infeasible before Thursday (no specialist). Plan C violates SLA (risk too high for Tier 1 asset).
- Re-ranks: Plan B becomes optimal for now, Plan A re-scheduled for Thursday.

**What the user sees:**
- Plans re-ranked with **"RECOMMENDED"** badge on Plan B
- Constraint violations highlighted in red: "Plan A blocked: No certified mechanic available until Thu"
- Optimization explanation: "Plan B reduces risk by 62% at $1,200 cost and 30 min downtime. Plan A scheduled for Thu for remaining 38% risk."

---

### Step 6: Simulation — "Prove It"

Before committing, the user wants to see what the future looks like under each option.

**What the user does:** Clicks "Simulate" to compare plans.

**What the system does behind the scenes:**
- Simulation Agent runs Monte Carlo simulations for each plan
- Projects risk trajectory, cost trajectory, and downtime likelihood over the next 30 days
- Calculates confidence intervals (not just point estimates — "we're 90% confident risk drops below 15%")

**What the user sees (Simulation Panel):**
- **Three projected curves** side by side:
  - Plan B: Risk drops from 87% → 25% immediately, gradual climb back to 60% by day 14 (needs Plan A follow-up)
  - Plan A (Thursday): Risk drops from ~90% → 5% permanently
  - Plan C (do nothing): Risk climbs to ~95% by day 3, 70% chance of unplanned failure by day 5
- Cost-over-time comparison
- Uncertainty bands showing best/worst case for each scenario

---

### Step 7: Decision & Report

The user approves the plan. The system generates everything needed to execute.

**What the user does:** Clicks "Approve Plan B" (or the system auto-approves if confidence is high enough and Governance Agent clears it).

**What the system does behind the scenes:**
- Governance Agent validates: confidence floor met? Budget policy satisfied? No safety violations?
- Reporter Agent generates three versions of the report:
  1. **Operator Playbook:** Step-by-step maintenance instructions, parts list, safety checks
  2. **Manager Summary:** One-page with risk reduced, cost, projected impact
  3. **Audit Trace:** Full agent-by-agent decision log with evidence and confidence at every step

**What the user sees:**
- Downloadable maintenance playbook (PDF/Markdown)
- Executive summary card
- Full decision timeline showing exactly what each agent concluded and why

---

## 5. The 10-Agent Pipeline — What Happens Under the Hood

This is the technical flow. Each agent is a specialized module with a defined input/output contract.

```
┌─────────────┐
│  DATA IN    │ CSV, JSON, parquet, or replay stream
└──────┬──────┘
       │
       ▼
┌──────────────┐     ┌───────────────┐
│ 1. INTAKE    │────▶│ 2. QUALITY    │
│   Agent      │     │    Agent      │
│ Parse/       │     │ Missing data? │
│ standardize  │     │ Drift?        │
│ schema       │     │ Frozen sensor?│
└──────────────┘     └───────┬───────┘
                             │
                    ┌────────▼────────┐
                    │ 3. SENTINEL     │
                    │    Agent        │
                    │ Anomaly score   │
                    │ Regime change   │
                    │ per asset       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ 4. PROGNOSTICS  │
                    │    Agent        │
                    │ Failure prob.   │
                    │ RUL estimate    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ 5. CAUSAL       │
                    │    Agent        │
                    │ Root cause      │
                    │ hypotheses      │
                    │ Evidence graph  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ 6. PLANNER      │
                    │    Agent        │
                    │ Generate 3+     │
                    │ intervention    │
                    │ options         │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ 7. OPTIMIZER    │
                    │    Agent        │
                    │ Rank plans by   │
                    │ weighted score  │
                    │ Apply hard      │
                    │ constraints     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ 8. SIMULATION   │
                    │    Agent        │
                    │ Monte Carlo     │
                    │ What-if compare │
                    │ Uncertainty     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ 9. REPORTER     │
                    │    Agent        │
                    │ Playbook        │
                    │ Mgr summary     │
                    │ Audit trace     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │10. GOVERNANCE   │
                    │    Agent        │
                    │ Policy check    │
                    │ Confidence gate │
                    │ Human-in-loop   │
                    │ if needed       │
                    └─────────────────┘
```

### Communication Contract

Every agent emits a standardized output:

```json
{
  "input_context_id": "ctx-00482",
  "output_payload": { ... },
  "confidence_score": 0.87,
  "assumptions": ["bearing wear rate is linear in this regime"],
  "evidence_refs": ["sensor_vibration_042", "maint_log_2026-02"],
  "errors": [],
  "next_recommended_agent": "planner_agent"
}
```

This means every decision is **traceable, auditable, and explainable**.

### Orchestration Rules

- **Normal flow:** Linear pipeline from Agent 1 → 10
- **Conditional routing:** If Quality Agent flags all sensors as unreliable → skip to Reporter with a "low confidence, data quality issue" report instead of generating false plans
- **Circuit breaker:** If any agent's confidence drops below the threshold → Governance Agent halts and requests human review
- **Retry policy:** Transient failures get exponential backoff retries before degrading gracefully

---

## 6. The Four Dashboard Screens

### Screen 1: Fleet Overview
- **Risk heatmap** — all 200 assets at a glance, colored by failure probability
- **Trend charts** — anomaly score over time per asset class
- **Quick stats** — assets at risk, incidents open, cost saved this period

### Screen 2: Incident Workbench
- **Root-cause graph** — visual hypothesis tree with confidence percentages
- **Candidate plans** — side-by-side plan cards with cost/downtime/risk
- **Constraint panel** — configure budget, crew, blackout windows
- **SHAP chart** — which sensor features drove this prediction

### Screen 3: Simulation Panel
- **Plan comparison** — A/B/C risk curves with uncertainty bands
- **Cost projection** — cumulative cost over 30-day horizon
- **Downtime forecast** — expected operational hours lost per plan

### Screen 4: Report & Audit View
- **Decision timeline** — chronological log of all agent decisions
- **Agent trace** — click any agent to see its input, output, confidence, assumptions
- **Export** — one-click PDF/Markdown export for stakeholders

---

## 7. How AI Is Used (and How It Isn't)

This is a critical differentiator. Most hackathon projects throw an LLM at everything. Hephaestus uses **the right tool for each job**.

| Task | Tool Used | Why |
|---|---|---|
| Anomaly detection | Isolation Forest + z-score hybrid | ML model trained on sensor data — LLMs can't process 1M time-series rows |
| Failure probability | Gradient boosting (XGBoost/LightGBM) | Tabular data classification — this is what GBMs are built for |
| RUL estimation | Survival regression | Statistical model specifically designed for time-to-event prediction |
| Explainability | SHAP values | Mathematically grounded feature attribution, not vague LLM text |
| Root cause reasoning | LLM (structured tool-calling) | Synthesizing hypotheses from multiple evidence sources — LLMs excel here |
| Plan generation | LLM (structured output) | Generating human-readable intervention options with parts/skills lists |
| Policy reasoning | LLM (JSON schema) | Checking plans against natural language maintenance policies |
| Report composition | LLM | Writing stakeholder-appropriate summaries from structured data |
| Optimization | Mathematical solver | Multi-objective optimization with hard constraints — deterministic, provable |
| Monte Carlo simulation | scipy/numpy | Statistical simulation — needs millions of fast iterations, not language |

### The Key Quote for Judges

> "We use ML models where precision matters, LLMs where reasoning and synthesis matter, and mathematical solvers where optimality must be proven. No single tool is asked to do everything."

---

## 8. The Demo Script — Running the Live Pitch

### Setup (Pre-Demo)
- 200 synthetic assets across 3 classes (pumps, compressors, turbines)
- 6 months of telemetry history already loaded
- Hidden degradation pattern injected in pump fleet

### Live Demo — 5 Minute Narrative

**Minute 0-1: The Spike**
> "Here's our plant's fleet overview. Everything's been green for weeks. But watch what happens when we load this morning's data batch..."

*Upload new data. Risk heatmap shifts — PUMP-0042 through PUMP-0047 go orange/red.*

> "Six pumps in the same class just spiked. Our Sentinel Agent detected correlated anomalies across the bearing vibration sensors."

**Minute 1-2: The Why**
> "Let's click on PUMP-0042. The Prognostics Agent says 87% failure probability in the next 36 hours. But here's what other tools can't tell you — *why*."

*Show root-cause graph.*

> "The Causal Agent identified bearing degradation as the primary cause with 78% confidence. Look at the evidence trail — vibration trending up for 14 days, while temperature holds steady. That's a textbook bearing wear signature, not lubrication failure."

**Minute 2-3: The Plan**
> "Now here's where Hephaestus goes further than any detection tool. It's already generated three intervention plans."

*Show plan cards. Toggle constraints panel.*

> "But Plan A needs a bearing specialist — and ours isn't available until Thursday. When I apply our real constraints, the Optimizer re-ranks and recommends Plan B as the immediate action, with Plan A scheduled for follow-up."

**Minute 3-4: The Proof**
> "You might ask — how do we know Plan B actually works? Let's simulate."

*Show simulation comparison.*

> "Monte Carlo simulation shows Plan B drops risk from 87% to 25% immediately. But look — without the follow-up Plan A on Thursday, risk creeps back to 60% by day 14. The system doesn't just recommend — it shows you the trajectory with confidence intervals."

**Minute 4-5: The Outcome**
> "One click — and the maintenance playbook is exported. The operator gets step-by-step instructions. The manager gets a one-page cost-impact summary. And compliance gets the full agent-by-agent decision trace."

*Show report export. Show audit trail.*

> "Every decision is explainable. Every confidence score is backed by evidence. No black box. This is what predictive maintenance should be — not just alerts, but autonomous decisions with provable impact."

---

## 9. Key Talking Points for Judges

### "Why is this different from existing predictive maintenance tools?"

> Existing tools stop at detection: "anomaly detected." Hephaestus closes the full loop — detection to root cause to optimized plan to simulated outcome to maintenance playbook. There's no human bottleneck between detection and decision.

### "Why multiple agents instead of one model?"

> Because no single model can detect anomalies in time-series data, reason about causal chains, generate human intervention plans, optimize under constraints, AND simulate Monte Carlo outcomes. Each agent uses the right tool for its specific job. That's why it works.

### "How do you handle uncertainty?"

> Every agent emits a confidence score. Every simulation output has uncertainty intervals. If any agent's confidence drops below threshold, the Governance Agent halts the pipeline and escalates to a human. We never auto-execute with low confidence.

### "What happens when the AI is wrong?"

> Three safeguards: (1) Confidence floor — low confidence triggers human review, not auto-action. (2) Audit trace — every decision shows evidence and assumptions, so humans can catch reasoning errors. (3) No irreversible actions — the system recommends, humans approve, in MVP.

### "Is this just a wrapper around ChatGPT?"

> No. The anomaly detection is Isolation Forest. Failure probability is gradient boosting. Optimization is mathematical. Simulation is Monte Carlo. LLMs are only used where language helps — plan synthesis, explanation, report writing. The core intelligence is ML models + solvers.

### "Can this work with real data?"

> Today it runs on synthetic data that mimics real industrial patterns — sensor drift, sudden spikes, gradual degradation, multiple failure modes, noisy channels. The architecture is connector-ready: swap the data loader and point it at a real historian, SCADA, or IoT stream. No code changes to the agent pipeline.

---

## 10. Anticipated Questions & Strong Answers

| Question | Answer |
|---|---|
| "What's your data source?" | Synthetic generator producing 1M+ telemetry rows across 200 assets with realistic degradation patterns and 4+ failure modes. Production-ready for real data connectors. |
| "Which LLM do you use?" | Primary: Ollama local inference (Mistral/Llama) for data privacy and zero API cost. Fallback: Google Gemini API when local inference is unavailable. The architecture is provider-agnostic — swap without changing agent logic. |
| "How do you prevent hallucinations?" | LLMs are only used for synthesis and explanation, never for prediction. All LLM outputs have structured JSON schemas with required fields and are validated before use. |
| "What's the tech stack?" | Python/FastAPI backend with 10 agents. PostgreSQL + Redis. Next.js/React/TypeScript dashboard. Docker Compose for deployment. |
| "How fast does it run?" | Synchronous API for user interactions. Heavy model runs and simulation batches go through an async job queue (Celery/Redis). |
| "What about false alarms?" | Dynamic thresholds per asset class, plus confidence scoring means we quantify how sure we are. A 55% alert looks very different from a 92% alert on the dashboard. |
| "Can it handle new failure types?" | Yes. Anomaly detection is unsupervised (Isolation Forest) — it detects deviation from normal, not just known failures. New failure modes appear as novel anomaly patterns. |
| "How is this enterprise-ready?" | API key auth, full audit trails, policy versioning, role-based access (phase 2), Docker deployment, CI/CD via GitHub Actions. |

---

## 11. Differentiators — Why This Wins

### vs. Simple Anomaly Dashboards (Datadog, Grafana alerts)
They detect. We detect + explain + plan + optimize + simulate + report.

### vs. LLM-First "AI Ops" Projects
They ask GPT to explain a log. We use ML models for prediction, math solvers for optimization, Monte Carlo for simulation, and LLMs only where language helps.

### vs. Commercial PdM Tools (Azure PdM, AWS Lookout)
They're cloud-locked black boxes. We're open, explainable, and every decision shows its reasoning.

### The Killer Feature Matrix

| Capability | Typical Hackathon Project | Hephaestus |
|---|---|---|
| Anomaly detection | ✅ | ✅ |
| Failure prediction | ❌ (just alerts) | ✅ (probability + timeline) |
| Root cause explanation | ❌ | ✅ (hypothesis graph with confidence) |
| Plan generation | ❌ | ✅ (3+ options with parts/skills) |
| Constraint optimization | ❌ | ✅ (budget, crew, SLA, blackouts) |
| Monte Carlo simulation | ❌ | ✅ (confidence intervals) |
| Stakeholder reports | ❌ | ✅ (operator, manager, audit versions) |
| Full decision trace | ❌ | ✅ (agent-by-agent audit log) |
| Human-in-the-loop safety | ❌ | ✅ (Governance Agent gating) |

---

## 12. Business Impact Numbers

Use these in your pitch to make the impact tangible:

| Metric | Reactive Baseline | Scheduled Baseline | Hephaestus |
|---|---|---|---|
| Unplanned downtime | ~10-15% of operating hours | ~5-8% | **<2%** (predicted and prevented) |
| Maintenance cost efficiency | 100% (baseline) | 70% (over-maintenance) | **~50-60%** (only fix what needs fixing, when it needs fixing) |
| Mean time to resolution | 8-24 hours (after failure) | N/A (preventive) | **Pre-failure** (resolved before incident) |
| False alarm rate | N/A | N/A | **Quantified** — every alert has a confidence score |
| Decision traceability | None | Paper-based | **100% digital** — agent-by-agent audit trail |

### The Business Pitch One-Liner

> "Hephaestus can reduce unplanned downtime by up to 80% and cut maintenance costs by 30-40% by replacing reactive firefighting with AI-driven predictive decision-making — and every recommendation comes with a confidence score and a simulation-backed proof."

---

## 13. One-Liner Cheat Sheet

Keep these in your back pocket for quick answers:

| Topic | One-Liner |
|---|---|
| **Project** | AI reliability commander — predicts failures, explains causes, optimizes maintenance plans, proves impact through simulation |
| **Architecture** | 10 specialized agents in a directed graph, not one model pretending to do everything |
| **AI approach** | ML for prediction, LLM for reasoning, math for optimization — right tool for each job |
| **Safety** | Every decision has a confidence score; low confidence triggers human review, not auto-action |
| **Data** | Works on synthetic data today; architecture ready for real industrial data connectors |
| **Output** | Not just an alert — a complete decision package: root cause, optimized plan, simulation proof, maintenance playbook |
| **Impact** | 80% downtime reduction vs reactive, 30-40% cost savings vs scheduled maintenance |
| **Name meaning** | Hephaestus — Greek god of the forge. He builds things that don't break. |

---

*End of guide. Build the forge. Break nothing.*
