"""
Intake Agent (Agent 1)

Responsibilities:
    - Parse incoming data packages (CSV, JSON, parquet)
    - Standardize schema and units across heterogeneous sources
    - Validate timestamp integrity and ordering
    - Tag data with ingestion metadata (batch_id, source, timestamp)

Confidence logic: High if >95% rows valid, medium 80-95%, low <80%.
Next agent: Always routes to quality_agent.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import pandas as pd

from ml.aegis.agents import BaseAgent
from ml.aegis.data.schemas import AgentOutput, AssetMaster


class IntakeAgent(BaseAgent):

    name = "intake_agent"
    description = "Parses and standardizes incoming data packages"

    def run(self, context: dict[str, Any]) -> AgentOutput:
        """
        Ingest raw data from context and produce standardized outputs.

        Expected context keys:
            - data_source: str (file path) or dict with DataFrames
            - Or pre-loaded: telemetry_df, assets, events, maintenance, failures
        """
        batch_id = f"BATCH-{uuid.uuid4().hex[:8].upper()}"
        ingestion_time = datetime.now()
        errors: list[str] = []
        evidence: list[str] = []

        # --- Load data ---
        telemetry_df = context.get("telemetry_df")
        assets = context.get("assets", [])
        events = context.get("events", [])
        maintenance = context.get("maintenance", [])
        failures = context.get("failures", [])

        if telemetry_df is None:
            data_source = context.get("data_source")
            if data_source and isinstance(data_source, str):
                from ml.aegis.data.loaders import load_telemetry, load_asset_master
                try:
                    telemetry_df = load_telemetry(data_source)
                    evidence.append(f"Loaded telemetry from {data_source}")
                except Exception as e:
                    errors.append(f"Failed to load telemetry: {e}")
                    return self._build_output(
                        payload={"status": "ingestion_failed", "batch_id": batch_id},
                        confidence=0.0,
                        errors=errors,
                        next_agent="reporter_agent",
                    )

        if telemetry_df is None:
            return self._build_output(
                payload={"status": "no_data"},
                confidence=0.0,
                errors=["No telemetry data provided in context"],
                next_agent="reporter_agent",
            )

        # --- Validate and standardize ---
        total_rows = len(telemetry_df)

        # Ensure timestamp column is datetime
        telemetry_df["timestamp"] = pd.to_datetime(
            telemetry_df["timestamp"], errors="coerce",
        )
        invalid_timestamps = int(telemetry_df["timestamp"].isna().sum())
        if invalid_timestamps > 0:
            errors.append(f"{invalid_timestamps} rows have invalid timestamps")
            telemetry_df = telemetry_df.dropna(subset=["timestamp"])

        # Ensure sensor_value is numeric
        telemetry_df["sensor_value"] = pd.to_numeric(
            telemetry_df["sensor_value"], errors="coerce",
        )
        invalid_values = int(telemetry_df["sensor_value"].isna().sum())

        # Sort by time
        telemetry_df = telemetry_df.sort_values(["asset_id", "sensor_name", "timestamp"])

        # Ensure quality_flag column exists
        if "quality_flag" not in telemetry_df.columns:
            telemetry_df["quality_flag"] = "ok"

        # Tag with batch metadata
        telemetry_df["batch_id"] = batch_id
        telemetry_df["ingestion_time"] = ingestion_time

        valid_rows = len(telemetry_df)
        valid_pct = (valid_rows / total_rows * 100) if total_rows > 0 else 0

        # Collect summary stats
        unique_assets = telemetry_df["asset_id"].nunique()
        unique_sensors = telemetry_df["sensor_name"].nunique()
        time_range_start = telemetry_df["timestamp"].min()
        time_range_end = telemetry_df["timestamp"].max()

        # --- Confidence ---
        if valid_pct >= 95:
            confidence = 0.95
        elif valid_pct >= 80:
            confidence = 0.75
        else:
            confidence = 0.50

        evidence.append(f"Ingested {valid_rows:,} of {total_rows:,} rows ({valid_pct:.1f}% valid)")
        evidence.append(f"Assets: {unique_assets}, Sensors: {unique_sensors}")
        evidence.append(f"Time range: {time_range_start} → {time_range_end}")

        # --- Store in context for downstream agents ---
        context["telemetry_df"] = telemetry_df
        context["batch_id"] = batch_id

        return self._build_output(
            payload={
                "status": "ingested",
                "batch_id": batch_id,
                "total_rows": total_rows,
                "valid_rows": valid_rows,
                "valid_pct": round(valid_pct, 2),
                "invalid_timestamps": invalid_timestamps,
                "invalid_values": invalid_values,
                "unique_assets": unique_assets,
                "unique_sensors": unique_sensors,
                "time_range_start": str(time_range_start),
                "time_range_end": str(time_range_end),
                "asset_count": len(assets),
                "event_count": len(events),
                "maintenance_count": len(maintenance),
            },
            confidence=confidence,
            assumptions=["All timestamps are in UTC or local consistent timezone"],
            evidence=evidence,
            errors=errors if errors else None,
            next_agent="quality_agent",
        )
