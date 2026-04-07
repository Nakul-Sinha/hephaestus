# ML Build Planner — Hephaestus Aegis Module

This document is the task-by-task execution plan for building the entire ML intelligence layer. Each task is self-contained and descriptive — you should be able to hand any task to a developer and have them build it without asking questions.

Tasks are ordered by dependency. Do not skip ahead.

---

## Legend

- 🔴 **Critical Path** — Blocks multiple downstream tasks
- 🟡 **Important** — Needed for full pipeline but not a hard blocker
- 🟢 **Enhancement** — Adds polish, can be deferred if time-pressed
- **File** — The exact file(s) to implement
- **Depends On** — Which tasks must be finished first
- **Outputs** — What this task produces that downstream tasks consume
- **Done When** — Concrete, testable definition of completion

---

## Phase 0: Foundation — Data Layer

These tasks create the data backbone. Nothing else works until the system can ingest, validate, and generate data.

---

### Task 0.1 🔴 Define All Pydantic Schemas

**File:** `ml/aegis/data/schemas.py`
**Depends On:** Nothing
**Outputs:** Importable Pydantic v2 models used by every agent and every API endpoint

Create strict Pydantic v2 `BaseModel` classes for every data entity in the system. These are not optional — they are the contract that prevents garbage data from propagating through the pipeline.

**Models to define:**

1. **`AssetMaster`** — Represents one physical asset in the fleet.
   - `asset_id: str` — Unique identifier (e.g., `"PUMP-0042"`)
   - `asset_type: str` — Classification (e.g., `"pump"`, `"compressor"`, `"turbine"`)
   - `site_id: str` — Which facility this asset is in
   - `installation_date: datetime` — When it was installed (needed for age-based risk)
   - `maintenance_policy: str` — Current policy (e.g., `"reactive"`, `"scheduled"`, `"condition-based"`)
   - `criticality_tier: int` — 1 (critical) to 4 (low). Tier 1 assets get stricter confidence thresholds
   
2. **`TelemetryReading`** — One sensor measurement at one point in time.
   - `timestamp: datetime`
   - `asset_id: str`
   - `sensor_name: str` — (e.g., `"vibration_x"`, `"temperature"`, `"pressure"`)
   - `sensor_value: float`
   - `unit: str` — (e.g., `"mm/s"`, `"°C"`, `"bar"`)
   - `quality_flag: str` — `"ok"`, `"suspect"`, `"missing"`. Set by Quality Agent later

3. **`EventLog`** — A discrete event that happened to an asset (alarm, warning, state change).
   - `event_id: str`
   - `timestamp: datetime`
   - `asset_id: str`
   - `event_type: str` — (e.g., `"alarm"`, `"warning"`, `"shutdown"`, `"restart"`)
   - `severity: str` — `"info"`, `"warning"`, `"critical"`
   - `event_text: str`

4. **`MaintenanceLog`** — A past maintenance action performed on an asset.
   - `work_order_id: str`
   - `asset_id: str`
   - `action_type: str` — (e.g., `"bearing_replacement"`, `"lubrication"`, `"inspection"`)
   - `parts_used: list[str]`
   - `duration_minutes: int`
   - `cost: float`
   - `outcome: str` — `"success"`, `"partial"`, `"failed"`

5. **`FailureGroundTruth`** — Known failure events (used for model training and evaluation).
   - `asset_id: str`
   - `failure_time: datetime`
   - `failure_mode: str` — (e.g., `"bearing_failure"`, `"seal_leak"`, `"overheating"`, `"electrical_fault"`)
   - `impact_cost: float`
   - `downtime_minutes: int`

6. **`AgentOutput`** — The standardized communication contract every agent must emit.
   - `input_context_id: str` — Trace ID linking to the triggering input
   - `agent_name: str`
   - `output_payload: dict` — Agent-specific results
   - `confidence_score: float` — 0.0 to 1.0
   - `assumptions: list[str]` — What the agent assumed to reach its conclusion
   - `evidence_refs: list[str]` — Pointers to data that supports the conclusion
   - `errors: list[str]` — Non-fatal issues encountered
   - `next_recommended_agent: str | None` — Which agent should run next

7. **`InterventionPlan`** — A candidate maintenance plan produced by the Planner Agent.
   - `plan_id: str`
   - `recommended_action: str`
   - `required_parts: list[str]`
   - `required_skills: list[str]`
   - `estimated_duration_minutes: int`
   - `maintenance_window: str`
   - `predicted_risk_reduction: float`
   - `estimated_cost: float`
   - `expected_downtime_minutes: int`
   - `confidence: float`
   - `assumptions: list[str]`
   - `rollback_plan: str`

**Done When:**
- All models are importable with `from ml.aegis.data.schemas import AssetMaster, TelemetryReading, ...`
- Each model validates types strictly (invalid data raises `ValidationError`)
- A quick test script creates one instance of each model without error

---

### Task 0.2 🔴 Build the Synthetic Data Generator

**File:** `ml/aegis/data/synthetic_generator.py`
**Depends On:** Task 0.1 (schemas)
**Outputs:** CSV/JSON files or in-memory DataFrames containing all 5 core data tables

This is the most important ML task. Without realistic synthetic data, no model can be trained, no agent can be tested, and no demo can be run. The generator must simulate a real industrial fleet over 6-12 months.

**What to generate:**

1. **Asset Fleet (200+ assets across 3 classes)**
   - ~80 pumps, ~70 compressors, ~50 turbines
   - Each with random installation dates (1-15 years ago)
   - Each assigned a criticality tier (more Tier 1 in turbines, more Tier 3/4 in pumps)
   - Distributed across 3-5 site IDs

2. **Telemetry Data (1M+ rows)**
   - For each asset, generate readings every 15 minutes for 6-12 months
   - Each asset class has different sensor sets:
     - Pumps: `vibration_x`, `vibration_y`, `temperature`, `pressure`, `flow_rate`
     - Compressors: `vibration`, `discharge_temp`, `suction_pressure`, `discharge_pressure`, `oil_temp`
     - Turbines: `vibration_axial`, `vibration_radial`, `exhaust_temp`, `bearing_temp`, `rpm`
   - **Normal regime:** Sensor values fluctuate within a healthy band with Gaussian noise
   - **Degradation patterns (inject into ~15-20% of assets):**
     - *Gradual drift:* One sensor slowly trends upward over weeks (bearing wear signature)
     - *Sudden spike:* Abrupt jump in value then partial recovery (seal leak or thermal shock)
     - *Oscillation:* Growing oscillation amplitude (imbalance or resonance)
     - *Freeze:* Sensor reports exact same value for hours (sensor malfunction)
   - **Missing data:** Randomly drop 2-5% of readings for some sensors
   - **Noisy channels:** Add extra Gaussian noise to 1-2 sensors per asset class

3. **Event Logs**
   - Generate alarm/warning events correlated with degradation patterns
   - E.g., when vibration exceeds 2× normal, emit a `"warning"` event
   - When an asset hits critical degradation, emit `"critical"` events
   - ~5-15 events per degraded asset, 0-2 per healthy asset

4. **Maintenance Logs**
   - Generate historical maintenance for all assets
   - Scheduled maintenance: every 3-6 months for each asset (inspections, lubrication)
   - Reactive maintenance: after failure events
   - Costs: vary by action type ($200 for inspection, $1,500 for bearing replacement, etc.)

5. **Failure Ground Truth**
   - For the degraded assets, mark the failure time (end of degradation pattern)
   - Assign one of 4 failure modes: `bearing_failure`, `seal_leak`, `overheating`, `electrical_fault`
   - Some assets should have overlapping failure modes (e.g., overheating AND bearing failure)

**Critical design rule:** The degradation patterns must start 2-4 weeks BEFORE the failure event. This is what the Prognostics Agent will learn to detect — the window between "something is changing" and "it broke."

**Done When:**
- Running the generator produces 5 DataFrames/CSVs matching the Pydantic schemas
- Telemetry has 1M+ rows
- At least 30 assets show degradation → failure patterns
- A simple plot of a degraded asset's vibration shows visible trend change before failure

---

### Task 0.3 🟡 Implement Data Loaders

**File:** `ml/aegis/data/loaders.py`
**Depends On:** Task 0.1 (schemas)
**Outputs:** Functions that load CSV/JSON/parquet into validated Pydantic models or DataFrames

Build loader functions that read data from disk and return it in a standardized format:

- `load_asset_master(path: str) -> list[AssetMaster]`
- `load_telemetry(path: str) -> pd.DataFrame` — Returns DataFrame with validated column types
- `load_events(path: str) -> list[EventLog]`
- `load_maintenance(path: str) -> list[MaintenanceLog]`
- `load_failures(path: str) -> list[FailureGroundTruth]`
- Support CSV, JSON, and parquet auto-detection based on file extension
- All loaders should validate that required columns exist and types are correct

**Done When:**
- Can round-trip: generate → save → load → validate without errors
- Raises clear `ValueError` on missing/incorrect columns

---

### Task 0.4 🟡 Implement Data Validators

**File:** `ml/aegis/data/validators.py`
**Depends On:** Task 0.1 (schemas)
**Outputs:** Validation functions used by the Quality Agent

These validators run data quality checks that the Quality Agent will use:

- `check_missingness(df: pd.DataFrame) -> dict` — Returns % missing per column
- `check_sensor_freeze(df: pd.DataFrame, threshold_hours: float = 2.0) -> list[dict]` — Finds sensors reporting constant values for too long
- `check_outlier_bursts(df: pd.DataFrame, z_threshold: float = 4.0) -> list[dict]` — Finds sudden value spikes
- `check_timestamp_integrity(df: pd.DataFrame) -> dict` — Checks for gaps, duplicates, out-of-order
- `check_feature_drift(df: pd.DataFrame, reference_df: pd.DataFrame) -> dict` — Compares distribution of recent window against historical baseline (KS test or similar)

Each function returns a structured dict (not print statements) so the Quality Agent can include it in its `AgentOutput`.

**Done When:**
- Running validators on synthetic data correctly identifies the injected problems (missing data, frozen sensors, drift)
- Each function returns a structured result, not just True/False

---

## Phase 1: ML Models — The Prediction Engine

These are the core ML models that power detection and forecasting. They use classical ML, not LLMs.

---

### Task 1.1 🔴 Build the Anomaly Detection Model

**File:** `ml/aegis/models/anomaly.py`
**Depends On:** Task 0.2 (synthetic data)
**Outputs:** Anomaly scores per asset, used by Sentinel Agent

This is the first line of defense — detecting that something is abnormal before it becomes a failure.

**Approach: Isolation Forest + Z-Score Hybrid**

1. **Feature Engineering (per asset, per time window):**
   - For each sensor, compute rolling statistics over the last 24h window:
     - `mean`, `std`, `min`, `max`, `range`, `rate_of_change` (slope of linear fit)
   - This turns raw time-series into a tabular feature vector per asset per window
   - Example: PUMP-0042 at time T has features `[vibration_x_mean=3.2, vibration_x_std=0.4, vibration_x_roc=0.02, temp_mean=72.1, ...]`

2. **Isolation Forest:**
   - Train one Isolation Forest per asset class (pumps, compressors, turbines) on healthy data only
   - Healthy data = time windows where no failure occurs within the next 30 days
   - The model learns what "normal" looks like for each asset class
   - At inference, output anomaly score: -1 (anomalous) to +1 (normal), then normalize to 0.0-1.0 risk scale

3. **Robust Z-Score guard:**
   - For each sensor, also compute `z = (current_value - rolling_median) / rolling_MAD`
   - If any sensor has |z| > 4, flag it as a point anomaly regardless of Isolation Forest score
   - This catches sudden spikes that the IF might miss (IF is better at detecting slow drift)

4. **Dynamic Thresholding:**
   - Different asset classes have different normal operating ranges
   - Store per-class threshold calibration: the anomaly score above which we flag as "at risk"
   - Calibrate using healthy data: set threshold at 95th percentile of healthy anomaly scores

**API:**
```python
class AnomalyDetector:
    def fit(self, telemetry_df: pd.DataFrame, asset_master: list[AssetMaster]) -> None: ...
    def score(self, telemetry_window: pd.DataFrame) -> pd.DataFrame:
        """Returns DataFrame: asset_id, anomaly_score (0-1), is_anomalous (bool), top_contributing_sensors"""
```

**Done When:**
- Model trains on synthetic healthy data without error
- Degraded assets consistently score higher than healthy assets (AUC > 0.75 on synthetic data)
- Output is a clean DataFrame with scores, not raw sklearn output

---

### Task 1.2 🔴 Build the Failure Risk Model

**File:** `ml/aegis/models/failure_risk.py`
**Depends On:** Task 0.2 (synthetic data), Task 1.1 (anomaly features)
**Outputs:** Failure probability per asset within next 24-48h, used by Prognostics Agent

This model answers: "Given current sensor readings and anomaly scores, what is the probability this asset fails in the next N hours?"

**Approach: Gradient Boosting Classifier (XGBoost or LightGBM)**

1. **Label Construction:**
   - For each asset at each time window: label = 1 if a failure occurs within the next 48 hours, else 0
   - Use the `FailureGroundTruth` table to compute this
   - This creates a heavily imbalanced dataset (~2-5% positive class) — use class weights or SMOTE

2. **Feature Vector:**
   - Same rolling sensor statistics from Task 1.1 features
   - PLUS: anomaly score from the Isolation Forest
   - PLUS: asset age (days since installation)
   - PLUS: days since last maintenance action
   - PLUS: number of events in last 7 days
   - PLUS: criticality tier (as categorical feature)

3. **Training:**
   - Train/validation split: time-based split (train on first 70% of timeline, validate on last 30%)
   - Do NOT use random split — this is time-series, random split causes data leakage
   - Tune with basic hyperparameter grid (max_depth, learning_rate, n_estimators)
   - Track: ROC-AUC, PR-AUC, F1 at threshold 0.5

4. **Output:**
   - `predict_proba()` returns calibrated probability
   - Also return a "failure horizon" estimate: if probability > threshold, estimate hours until probable failure based on degradation rate

**API:**
```python
class FailureRiskModel:
    def fit(self, features_df: pd.DataFrame, labels: pd.Series) -> dict:
        """Returns training metrics dict"""
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Returns: asset_id, failure_probability, failure_horizon_hours, confidence_interval"""
```

**Done When:**
- ROC-AUC > 0.80 on the time-based validation split
- PR-AUC > 0.40 (given class imbalance, this is a strong result)
- Degraded assets near failure consistently show probability > 0.7

---

### Task 1.3 🟡 Build the RUL Estimation Model (Optional)

**File:** `ml/aegis/models/rul.py`
**Depends On:** Task 0.2 (synthetic data)
**Outputs:** Estimated remaining useful life in hours/days per asset

This is a stretch goal. Only implement after Tasks 1.1 and 1.2 are solid.

**Approach:** Survival regression (Cox Proportional Hazards or Accelerated Failure Time) using the `lifelines` library, or a simple regression model predicting hours-until-failure.

**Done When:**
- For assets with known failure times, the model estimates RUL within ±20% for failures occurring within 14 days

---

### Task 1.4 🔴 Build the Explainability Module

**File:** `ml/aegis/models/explainability.py`
**Depends On:** Task 1.1 (anomaly model), Task 1.2 (failure risk model)
**Outputs:** Per-prediction feature contribution explanations, used by Causal Agent

This is critical for trust and for the demo. Judges will ask "why does the model think this?" and this module provides the answer.

**Approach: SHAP (SHapley Additive exPlanations)**

1. **Wrap both models:**
   - Create a function that takes a trained model + a feature row and returns SHAP values
   - For the Isolation Forest: use `shap.TreeExplainer` (if using sklearn's IF, you may need `shap.KernelExplainer` with a small background sample)
   - For the Failure Risk GBM: use `shap.TreeExplainer` (native support for XGBoost/LightGBM)

2. **Output format:**
   ```python
   {
       "asset_id": "PUMP-0042",
       "prediction": 0.87,
       "top_contributors": [
           {"feature": "vibration_x_mean", "shap_value": 0.23, "direction": "increases_risk"},
           {"feature": "days_since_maintenance", "shap_value": 0.18, "direction": "increases_risk"},
           {"feature": "temperature_std", "shap_value": -0.05, "direction": "decreases_risk"}
       ]
   }
   ```

3. **Use for root cause:**
   - The Causal Agent will use these SHAP outputs to support its hypothesis graph
   - Example: if `vibration_x_mean` is the top SHAP contributor, the Causal Agent adds "bearing degradation" as a hypothesis with high evidential support

**Done When:**
- Can generate SHAP explanations for any single prediction from either model
- Output is a structured dict, not a matplotlib plot
- Top contributors make physical sense (vibration features rank high for bearing failures)

---

## Phase 2: Agent Implementation — The Intelligence Layer

Each agent is a Python class following the same interface pattern. All agents consume and produce `AgentOutput` objects.

---

### Task 2.0 🔴 Define the Base Agent Interface

**File:** `ml/aegis/agents/__init__.py` (update existing)
**Depends On:** Task 0.1 (schemas — specifically `AgentOutput`)
**Outputs:** Abstract base class that all 10 agents inherit from

```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """All agents inherit from this. Enforces the communication contract."""
    
    name: str  # e.g., "sentinel_agent"
    
    @abstractmethod
    def run(self, context: dict) -> AgentOutput:
        """Execute this agent's logic. Returns standardized AgentOutput."""
        ...
    
    def _build_output(self, payload, confidence, assumptions, evidence, errors=None, next_agent=None) -> AgentOutput:
        """Helper to construct AgentOutput with proper trace IDs."""
        ...
```

**Done When:**
- All 10 agent files can `from ml.aegis.agents import BaseAgent` and subclass it

---

### Task 2.1 🔴 Implement the Intake Agent

**File:** `ml/aegis/agents/intake_agent.py`
**Depends On:** Task 2.0 (base agent), Task 0.3 (loaders)
**Outputs:** Parsed, schema-validated, unit-standardized data ready for downstream agents

**What it does:**
- Receives raw data paths or DataFrames
- Calls the loaders to parse CSV/JSON/parquet
- Validates all rows against Pydantic schemas
- Standardizes units if needed (e.g., temperature always in °C, pressure always in bar)
- Tags each row with a `batch_id` and `ingestion_timestamp`
- Reports: how many rows ingested, how many failed validation, which sensors are present

**Confidence logic:** High if >95% of rows pass validation, medium if 80-95%, low if <80%.

**Next agent:** Always routes to `quality_agent`.

**Done When:**
- Feed it the synthetic data CSVs → get clean DataFrames + AgentOutput with ingestion summary

---

### Task 2.2 🔴 Implement the Quality Agent

**File:** `ml/aegis/agents/quality_agent.py`
**Depends On:** Task 2.0 (base agent), Task 0.4 (validators)
**Outputs:** Data quality report + flagged unreliable channels

**What it does:**
- Runs all validators from Task 0.4 on the ingested data
- For each sensor channel, assigns a quality verdict: `"ok"`, `"suspect"`, `"unreliable"`
- Produces a quality summary: % missing by sensor, frozen sensors detected, drift signals
- Updates the `quality_flag` field on telemetry readings

**Confidence logic:** Based on overall data quality. If >90% of channels are "ok", confidence is high. If many channels are "suspect" or "unreliable", confidence drops and the agent includes a warning.

**Critical behavior:** If ALL channels for a critical asset are "unreliable", the agent should flag this in `errors` and set `next_recommended_agent` to `reporter_agent` (skip the prediction pipeline — you can't predict on garbage data).

**Next agent:** Routes to `sentinel_agent` (normal) or `reporter_agent` (data quality crisis).

**Done When:**
- Correctly identifies frozen sensors and missing data in synthetic dataset
- Routes to reporter instead of sentinel when data is too poor

---

### Task 2.3 🔴 Implement the Sentinel Agent

**File:** `ml/aegis/agents/sentinel_agent.py`
**Depends On:** Task 2.0 (base agent), Task 1.1 (anomaly model)
**Outputs:** Anomaly scores per asset, list of at-risk assets

**What it does:**
- Takes the quality-checked telemetry data
- Computes feature windows (rolling statistics)
- Runs the anomaly detection model from Task 1.1
- Applies dynamic thresholds per asset class
- Produces a ranked list of assets by anomaly score
- Filters: only assets scoring above threshold are passed downstream

**Confidence logic:** Confidence per asset based on how far above/below threshold the score is. An asset barely above threshold has lower confidence than one far above.

**Next agent:** Routes to `prognostics_agent`.

**Done When:**
- Given synthetic data with injected degradation, correctly identifies degraded assets in top-ranked positions
- Healthy assets are filtered out (scored below threshold)

---

### Task 2.4 🔴 Implement the Prognostics Agent

**File:** `ml/aegis/agents/prognostics_agent.py`
**Depends On:** Task 2.0 (base agent), Task 1.2 (failure risk model)
**Outputs:** Failure probability + estimated time-to-failure for flagged assets

**What it does:**
- For each at-risk asset from Sentinel Agent:
  - Compute the full feature vector (sensor stats + anomaly score + asset age + maintenance recency)
  - Run the failure risk model to get probability
  - Estimate failure horizon (hours until probable failure)
- Adds confidence intervals on the probability estimate

**Confidence logic:** Based on model calibration. If the model's predicted probability is 0.87 but the calibration error at that range is ±0.10, confidence reflects that.

**Abstain-and-escalate rule:** If uncertainty is too wide (e.g., 90% CI spans more than 0.40), the agent should note this in assumptions: "High uncertainty — recommend manual inspection" and set confidence below the governance threshold.

**Next agent:** Routes to `causal_agent`.

**Done When:**
- Outputs for degraded assets show probability > 0.7 and reasonable failure horizon (12-72h)
- Outputs include confidence intervals, not just point estimates

---

### Task 2.5 🔴 Implement the Causal Agent

**File:** `ml/aegis/agents/causal_agent.py`
**Depends On:** Task 2.0 (base agent), Task 1.4 (explainability), LLM integration (Ollama primary / Gemini fallback)
**Outputs:** Ranked root-cause hypotheses with evidence

**What it does:**
This is the first agent that uses an LLM. It synthesizes evidence from multiple sources into a causal hypothesis graph. The agent calls **Ollama** (local Mistral/Llama model) as the primary inference engine. If Ollama is unreachable or returns an error after retries, it falls back to the **Google Gemini API**.

1. **Gather evidence:**
   - SHAP feature contributions from explainability module (Task 1.4)
   - Recent event logs for the asset
   - Maintenance history (when was the last bearing replacement? last lubrication?)
   - Asset age and criticality

2. **Build a structured prompt for the LLM (same prompt works for both Ollama and Gemini):**
   - Include all evidence as structured data
   - Ask the LLM to produce a JSON response with:
     ```json
     {
       "hypotheses": [
         {
           "cause": "bearing_degradation",
           "confidence": 0.78,
           "evidence_for": ["vibration_x increasing 23% over 14 days", "14 months since last bearing replacement"],
           "evidence_against": ["temperature stable — unusual for bearing failure"],
           "contradiction_notes": "Temperature stability may indicate early-stage degradation where thermal effects haven't manifested"
         }
       ]
     }
     ```
   - Use structured tool-calling / JSON schema enforcement to prevent freeform hallucination
   - **Ollama note:** Use the `/api/generate` endpoint with `format: "json"` to enforce JSON output
   - **Gemini fallback note:** Use `response_mime_type="application/json"` with `response_schema` for structured output

3. **Cross-validate:** If the top SHAP contributor is vibration, but the LLM hypothesizes "electrical fault" (which wouldn't show in vibration), flag a contradiction.

**Confidence logic:** Based on evidence strength. More evidence_for + fewer contradictions = higher confidence.

**Next agent:** Routes to `planner_agent`.

**Done When:**
- Given a degraded pump with vibration trend, correctly produces "bearing_degradation" as top hypothesis
- Output is valid JSON matching the schema
- Contradictions are noted, not ignored

---

### Task 2.6 🔴 Implement the Planner Agent

**File:** `ml/aegis/agents/planner_agent.py`
**Depends On:** Task 2.0 (base agent), LLM integration (Ollama primary / Gemini fallback)
**Outputs:** 3+ candidate `InterventionPlan` objects

**What it does:**
Uses an LLM to generate realistic intervention options based on the root-cause analysis. Calls **Ollama locally** first; if unavailable, falls back to **Google Gemini API**.

1. **Input:** Causal Agent's hypothesis graph + asset details + maintenance history
2. **LLM prompt asks for 3 plans:**
   - One aggressive fix (highest cost, highest risk reduction)
   - One moderate fix (balanced cost/risk)
   - One conservative option (monitor/defer, lowest cost, lowest risk reduction)
3. **Each plan must include:** action description, required parts, required skills, estimated duration, maintenance window, and a rollback plan
4. **Validate LLM output** against `InterventionPlan` Pydantic schema — reject and retry if schema validation fails (up to 3 retries, then switch to fallback provider)

**Confidence logic:** High if the root cause has high confidence (good plans come from good diagnosis).

**Next agent:** Routes to `optimizer_agent`.

**Done When:**
- Produces 3 valid `InterventionPlan` objects for a given root-cause analysis
- Plans are physically sensible (bearing replacement plan includes bearing parts, not random components)

---

### Task 2.7 🔴 Implement the Optimizer Agent

**File:** `ml/aegis/agents/optimizer_agent.py`
**Depends On:** Task 2.0 (base agent), Task 2.6 (plans), Planning module (Phase 3)
**Outputs:** Ranked plans with feasibility assessment

**What it does:**
This agent does NOT use an LLM. It uses the mathematical optimizer from the planning module.

1. **Score each plan** using the multi-objective function:
   ```
   score = w1 * risk_reduction - w2 * cost - w3 * downtime + w4 * sla_compliance
   ```
2. **Check hard constraints:**
   - Is the plan within budget ceiling?
   - Are required crew skills available?
   - Are spare parts in inventory?
   - Does the maintenance window avoid blackout periods?
3. **Eliminate infeasible plans** (mark as `"infeasible"` with reason)
4. **Rank remaining plans** by score
5. Mark the top plan as `"RECOMMENDED"`

**Confidence logic:** Based on how much the top plan's score dominates the second-best. Large gap = high confidence in the recommendation. Close scores = lower confidence ("either plan is reasonable").

**Next agent:** Routes to `simulation_agent`.

**Done When:**
- Given 3 plans + constraints, correctly eliminates infeasible ones and ranks the rest
- Changing constraints (e.g., removing a crew member) changes which plans are feasible/ranked

---

### Task 2.8 🔴 Implement the Simulation Agent

**File:** `ml/aegis/agents/simulation_agent.py`
**Depends On:** Task 2.0 (base agent), Simulation module (Phase 3)
**Outputs:** Projected risk/cost/downtime trajectories for each plan

**What it does:**
Runs Monte Carlo simulations to project what happens under each plan over a 30-day horizon.

1. **For each plan (including "do nothing"):**
   - Run N=1000 Monte Carlo iterations
   - Each iteration: sample from probability distributions for degradation rate, repair effectiveness, cost variance
   - Track: risk trajectory (probability of failure over time), cumulative cost, cumulative downtime
2. **Compute statistics:**
   - Mean trajectory + 5th/95th percentile bands (90% confidence interval)
   - Expected total cost, expected total downtime after 30 days
3. **Compare:** For each pair of plans, compute the probability that Plan A outperforms Plan B
4. **Output:** Structured data ready for the frontend Simulation Panel charts

**Confidence logic:** Based on how tight the uncertainty bands are. Narrow bands = high confidence in projections.

**Next agent:** Routes to `reporter_agent`.

**Done When:**
- "Do nothing" trajectory shows increasing risk, while intervention plans show risk reduction
- Uncertainty bands are wider for longer time horizons (as they should be)
- Output data structure is directly chartable

---

### Task 2.9 🟡 Implement the Reporter Agent

**File:** `ml/aegis/agents/reporter_agent.py`
**Depends On:** Task 2.0 (base agent), LLM integration (Ollama primary / Gemini fallback)
**Outputs:** Three report variants (operator, manager, audit)

**What it does:**
Uses an LLM (Ollama primary, Gemini fallback) to compose human-readable reports from the structured pipeline output.

1. **Operator Playbook:**
   - Step-by-step maintenance instructions
   - Parts checklist
   - Safety precautions
   - Estimated time and crew requirements

2. **Manager Summary:**
   - One-page overview: what's at risk, recommended action, expected cost savings, projected impact
   - Business language, not technical jargon

3. **Audit Trace:**
   - Chronological log of every agent's `AgentOutput`
   - For each agent: what it input, what it concluded, confidence, assumptions, evidence
   - Full decision traceability

**Output format:** JSON structure + rendered Markdown for each report type.

**Done When:**
- All three reports generate from pipeline output
- Operator report is actionable (a mechanic could follow it)
- Manager report fits in one page
- Audit trace includes every agent's output

---

### Task 2.10 🔴 Implement the Governance Agent

**File:** `ml/aegis/agents/governance_agent.py`
**Depends On:** Task 2.0 (base agent)
**Outputs:** Approval/rejection + policy violation flags

**What it does:**
The final safety gate before output reaches a human.

1. **Confidence floor check:** If any critical-path agent's confidence is below the configured threshold (e.g., 0.60), BLOCK and recommend human review
2. **Policy validation:**
   - Is the recommended plan within the organization's maintenance policy?
   - Does the cost exceed the auto-approval limit?
   - Is this a Tier 1 critical asset? (always requires human approval)
3. **Safety rules:**
   - No irreversible actions in MVP — all recommendations are advisory
   - High-impact recommendations (cost > $10K or downtime > 8h) require human sign-off
4. **Emit verdict:**
   - `"approved"` — pipeline output can be auto-delivered
   - `"needs_human_review"` — flag for manual approval with reasons
   - `"rejected"` — policy violation, output must not be delivered

**Done When:**
- Low-confidence pipeline outputs trigger `"needs_human_review"`
- Tier 1 asset recommendations always require human approval
- Policy violations are caught and described in the output

---

### Task 2.11 🔴 Implement the Orchestrator

**File:** `ml/aegis/agents/orchestrator.py`
**Depends On:** Tasks 2.1-2.10 (all agents)
**Outputs:** Runs the full pipeline end-to-end

**What it does:**
The orchestrator is the conductor. It manages the agent DAG:

1. **Normal flow:**
   ```
   Intake → Quality → Sentinel → Prognostics → Causal → Planner → Optimizer → Simulation → Reporter → Governance
   ```
2. **Conditional routing:**
   - Quality Agent flags data crisis → skip to Reporter
   - Sentinel Agent finds no anomalies → skip to Reporter (nothing to report)
   - Governance Agent rejects → route back to Reporter with rejection explanation
3. **Retry logic:**
   - If an agent raises an exception, retry up to 3 times with exponential backoff
   - If all retries fail, log the error, mark the agent as `"failed"`, and try to continue with degraded output
4. **Circuit breaker:**
   - If the same agent fails 5 times across different pipeline runs, disable it and use a fallback heuristic
5. **Trace propagation:**
   - Generate a `pipeline_run_id` at the start
   - Pass `input_context_id` through each agent so the full chain is traceable
6. **Collect all AgentOutputs** into a `PipelineResult` that includes timing, confidence trail, and all intermediate outputs

**API:**
```python
class Orchestrator:
    def run_pipeline(self, data_source: str | dict) -> PipelineResult:
        """Runs the full 10-agent pipeline. Returns aggregated result."""
```

**Done When:**
- Can run the full pipeline end-to-end on synthetic data
- Conditional routing works (no anomalies → short pipeline)
- Retry logic handles simulated agent failures
- Full trace is captured in PipelineResult

---

## Phase 3: Optimization & Simulation Engines

These are the mathematical modules the Optimizer Agent and Simulation Agent call internally.

---

### Task 3.1 🔴 Implement Constraints Module

**File:** `ml/aegis/planning/constraints.py`
**Depends On:** Task 0.1 (schemas)
**Outputs:** Constraint validation functions

Define constraint objects and a constraint checker:

```python
@dataclass
class PlanningConstraints:
    budget_ceiling: float          # e.g., 45000.0
    available_crew: dict           # e.g., {"mechanic": 2, "electrician": 1, "bearing_specialist": 0}
    spare_parts_inventory: list[str]  # e.g., ["SKF_bearing", "gasket_set", "lubricant"]
    blackout_windows: list[tuple[datetime, datetime]]  # Time ranges where no work is allowed

def check_plan_feasibility(plan: InterventionPlan, constraints: PlanningConstraints) -> tuple[bool, list[str]]:
    """Returns (is_feasible, list_of_violation_reasons)"""
```

**Done When:**
- Can correctly identify infeasible plans (needs a part not in inventory, needs a skill not available, etc.)

---

### Task 3.2 🔴 Implement Objective Function

**File:** `ml/aegis/planning/objective.py`
**Depends On:** Task 0.1 (schemas)
**Outputs:** Scoring function for plans

```python
@dataclass
class PolicyWeights:
    w_risk_reduction: float = 0.40
    w_cost: float = 0.25
    w_downtime: float = 0.20
    w_sla_compliance: float = 0.15

def score_plan(plan: InterventionPlan, weights: PolicyWeights) -> float:
    """Returns scalar score. Higher is better."""
    # Normalize each dimension to 0-1 range before weighting
```

Weights are configurable via policy profiles (e.g., a "cost_sensitive" profile lowers w_risk and raises w_cost).

**Done When:**
- Same plan scores differently under different policy profiles
- Plans with higher risk reduction score higher under default weights

---

### Task 3.3 🔴 Implement Monte Carlo Simulation Engine

**File:** `ml/aegis/simulation/monte_carlo.py`
**Depends On:** Task 0.1 (schemas)
**Outputs:** Projected trajectories with uncertainty intervals

```python
def simulate_plan(
    plan: InterventionPlan,
    current_risk: float,
    degradation_rate: float,
    n_iterations: int = 1000,
    horizon_days: int = 30
) -> SimulationResult:
    """
    Runs Monte Carlo simulation for one plan.
    
    Returns SimulationResult containing:
        - daily_risk_trajectory: array of shape (horizon_days, 3) — [mean, p5, p95]
        - daily_cost_trajectory: array of shape (horizon_days, 3)
        - total_expected_cost: float
        - total_expected_downtime_hours: float
        - probability_of_failure: float (over the horizon)
    """
```

**Key modeling decisions:**
- Degradation rate is sampled from a distribution (not fixed) — this is what creates the uncertainty bands
- Repair effectiveness is also sampled (e.g., bearing replacement reduces risk by 85-95%, not exactly 90%)
- "Do nothing" scenario uses the current degradation rate to project forward

**Done When:**
- "Do nothing" shows rising risk. Intervention plans show risk drop then gradual rise.
- Uncertainty bands are wider for longer horizons
- Running 1000 iterations takes < 2 seconds (numpy vectorized, no Python loops)

---

### Task 3.4 🟡 Implement Scenario Comparison Engine

**File:** `ml/aegis/simulation/scenario_engine.py`
**Depends On:** Task 3.3 (Monte Carlo)
**Outputs:** Side-by-side comparison of multiple plans

```python
def compare_plans(
    plans: list[InterventionPlan],
    current_risk: float,
    degradation_rate: float,
    include_do_nothing: bool = True
) -> ComparisonResult:
    """
    Simulates all plans (+ "do nothing" baseline) and returns comparative results.
    Includes: pairwise probability of one plan outperforming another.
    """
```

**Done When:**
- Can compare 3 plans + do-nothing and produce chartable data for the frontend Simulation Panel

---

### Task 3.5 🟡 Implement Impact Estimator

**File:** `ml/aegis/simulation/impact_estimator.py`
**Depends On:** Task 3.3 (Monte Carlo)
**Outputs:** Business impact metrics (cost saved, downtime avoided)

Computes the delta between "do nothing" and the selected plan:
- `downtime_avoided_hours`
- `cost_saved_vs_reactive` (what it would cost if we waited for failure)
- `cost_saved_vs_scheduled` (what a fixed-schedule policy would cost)
- `risk_reduction_percentage`

These are the numbers that go in the Manager Summary report and the hackathon pitch slides.

**Done When:**
- Produces concrete dollar/hour savings numbers from simulation results

---

## Phase 4: Feature Engineering & Storage

---

### Task 4.1 🟡 Implement Feature Store

**File:** `ml/aegis/data/feature_store.py`
**Depends On:** Task 1.1 (anomaly model features)
**Outputs:** Cached, computed features for fast model inference

The feature store computes and caches the rolling window features used by both the anomaly model and the failure risk model:
- Input: raw telemetry DataFrame
- Output: feature DataFrame with one row per (asset_id, window_end_time)
- Features: rolling mean, std, min, max, range, rate_of_change for each sensor
- Also includes: asset_age, days_since_last_maintenance, event_count_last_7d

Cache in memory or Redis to avoid recomputing on every pipeline run.

**Done When:**
- Feature computation on 1M rows completes in < 10 seconds
- Features are reproducible (same input → same output)

---

## Phase 5: Telemetry & Observability

---

### Task 5.1 🟡 Implement Pipeline Tracing

**File:** `ml/aegis/telemetry/tracing.py`
**Depends On:** Task 2.11 (orchestrator)
**Outputs:** Trace IDs and span tracking across agents

Every pipeline run gets a unique `pipeline_run_id`. Each agent call is a "span" with:
- `span_id`, `agent_name`, `start_time`, `end_time`, `duration_ms`
- `input_summary` (hashed or truncated), `output_summary`
- `confidence_score`, `status` (success/failed/retried)

This feeds the Audit View in the dashboard.

**Done When:**
- A pipeline run produces a complete trace JSON showing all 10 agents' timing and status

---

### Task 5.2 🟢 Implement Metrics Collection

**File:** `ml/aegis/telemetry/metrics.py`
**Depends On:** Task 2.11 (orchestrator)
**Outputs:** Counters and histograms for system monitoring

Track:
- `pipeline_runs_total` — count of complete pipeline executions
- `pipeline_duration_seconds` — histogram of end-to-end latency
- `agent_failures_total` — count of agent errors by agent name
- `confidence_scores` — histogram of final confidence per pipeline run
- `plans_generated_total`, `plans_approved_total`, `plans_rejected_total`

**Done When:**
- After 10 pipeline runs, can print a summary of all metrics

---

## Phase 6: Testing

---

### Task 6.1 🔴 Unit Tests

**File:** `ml/aegis/tests/unit/`
**Depends On:** All Phase 0-3 tasks
**Tests:**
- Schema validation: valid data passes, invalid data raises `ValidationError`
- Validators: synthetic data with injected problems is correctly flagged
- Objective function: known inputs produce expected scores
- Constraint checker: infeasible plans correctly identified
- Feature computation: correct output shape and values

---

### Task 6.2 🔴 Integration Tests

**File:** `ml/aegis/tests/integration/`
**Depends On:** All Phase 2 tasks
**Tests:**
- Agent handoff: output of Agent N is valid input for Agent N+1
- Conditional routing: Quality crisis → skips to Reporter
- Retry: Simulated agent failure → retry succeeds
- Full pipeline: synthetic data → complete PipelineResult

---

### Task 6.3 🟡 End-to-End Tests

**File:** `ml/aegis/tests/e2e/`
**Depends On:** All phases
**Tests:**
- The demo scenario: 200 assets, inject degradation, run pipeline, verify output includes correct root cause, feasible plans, and simulation results
- Deterministic replay: same input data → same pipeline output (for reproducibility)

---

## Task Dependency Graph

```
Phase 0 (Data)         Phase 1 (Models)        Phase 2 (Agents)         Phase 3 (Math)
═══════════════        ════════════════        ═════════════════        ══════════════
                                                                       
T0.1 Schemas ────────► T1.1 Anomaly ─────────► T2.3 Sentinel           T3.1 Constraints
     │                      │                       │                       │
     ├───► T0.2 SynGen ─────┤                       ▼                       ▼
     │                      ├───► T1.2 Failure ► T2.4 Prognostics      T3.2 Objective
     ├───► T0.3 Loaders     │         │              │                       │
     │                      │         ▼              ▼                       ▼
     └───► T0.4 Valid. ─────┼──► T1.4 SHAP ───► T2.5 Causal           T3.3 MonteCarlo
                            │                       │                       │
                            │                       ▼                       ▼
T2.0 BaseAgent ─────────────┼──────────────────► T2.6 Planner          T3.4 Scenario
     │                      │                       │                       │
     ├──► T2.1 Intake       │                       ▼                       │
     ├──► T2.2 Quality      │               T2.7 Optimizer ◄───────────┘
     ├──► T2.9 Reporter     │                       │
     ├──► T2.10 Governance  │                       ▼
     │                      │               T2.8 Simulation ◄──── T3.3
     │                      │                       │
     └──── T2.11 Orchestrator ◄─────────────────────┘
```

---

## Priority Order for Hackathon

If time is short, build in this order to get a working demo fastest:

1. **T0.1** Schemas (30 min)
2. **T0.2** Synthetic Data Generator (2-3 hrs)
3. **T1.1** Anomaly Model (2 hrs)
4. **T1.2** Failure Risk Model (2 hrs)
5. **T1.4** Explainability (1 hr)
6. **T2.0** Base Agent Interface (30 min)
7. **T2.1-T2.4** First 4 agents — Intake through Prognostics (3 hrs)
8. **T3.1-T3.2** Constraints + Objective (1 hr)
9. **T3.3** Monte Carlo (1.5 hrs)
10. **T2.5-T2.8** Remaining core agents — Causal through Simulation (3 hrs)
11. **T2.11** Orchestrator (2 hrs)
12. **T2.9-T2.10** Reporter + Governance (2 hrs)
13. **T6.2** Integration tests (1 hr)

**Total estimated: ~20-22 hours of focused ML work.**

Everything else (feature store, RUL model, telemetry, E2E tests) is polish you add if time permits.
