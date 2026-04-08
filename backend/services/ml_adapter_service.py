"""ML adapter boundary between backend services and ml/aegis runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib import import_module

import numpy as np
import pandas as pd

from backend.contracts import (
    IncidentOptimizeRequest,
    IncidentPlanRequest,
    IncidentSimulateRequest,
    IngestBatchRequest,
    MLAdapterHealth,
    MLAdapterStageResult,
    RiskAnalyzeRequest,
)
from backend.services.incident_service import IncidentService

from ml.aegis.data.loaders import (
    load_asset_master,
    load_events,
    load_failures,
    load_maintenance,
    load_telemetry,
)
from ml.aegis.data.schemas import (
    ActionType,
    AssetMaster,
    AssetType,
    CriticalityTier,
    EventLog,
    EventType,
    FailureGroundTruth,
    FailureMode,
    MaintenanceLog,
    MaintenanceOutcome,
    MaintenancePolicy,
    Severity,
)
from ml.aegis.models.anomaly import AnomalyDetector, compute_rolling_features_all_windows
from ml.aegis.models.failure_risk import FailureRiskModel, build_failure_labels, enrich_features


@dataclass
class _IncidentInputContext:
    assets: list[AssetMaster]
    telemetry_df: pd.DataFrame
    events: list[EventLog]
    maintenance_logs: list[MaintenanceLog]
    failures: list[FailureGroundTruth]
    source_mode: str


class MLAdapterService:
    """Adapter facade used by pipeline service to call ML-backed stages."""

    def __init__(self, incident_service: IncidentService) -> None:
        self.incident_service = incident_service
        self._incident_inputs: dict[str, _IncidentInputContext] = {}

    @property
    def repository(self):
        """Expose underlying repository for governance trail updates."""
        return self.incident_service.repository

    def health(self) -> MLAdapterHealth:
        """Return import health of key ML modules used by integration path."""
        details: list[str] = []

        def _importable(path: str) -> bool:
            try:
                import_module(path)
                return True
            except Exception as exc:
                details.append(f"{path} import failed: {exc}")
                return False

        models_ok = _importable("ml.aegis.models.anomaly") and _importable("ml.aegis.models.failure_risk")
        orchestrator_ok = _importable("ml.aegis.agents.orchestrator")

        return MLAdapterHealth(
            ml_models_importable=models_ok,
            orchestrator_importable=orchestrator_ok,
            details=details,
        )

    def run_ingest(self, request: IngestBatchRequest) -> MLAdapterStageResult:
        payload = self.incident_service.ingest_batch(request)
        ingest = payload.get("ingest", {})

        try:
            if request.telemetry_path:
                context = self._build_context_from_paths(request)
            else:
                context = self._build_context_from_direct_request(request)

            incident_id = payload["incident_id"]
            self._incident_inputs[incident_id] = context

            ingest["data_mode"] = context.source_mode
            ingest["dataset_ready"] = True
        except Exception as exc:
            ingest.setdefault("warnings", []).append(f"ml dataset context not prepared: {exc}")
            ingest["dataset_ready"] = False

        return MLAdapterStageResult(
            payload=payload,
            confidence=float(ingest.get("confidence", 1.0)),
            warnings=list(ingest.get("warnings", [])),
        )

    def run_risk(self, request: RiskAnalyzeRequest) -> MLAdapterStageResult:
        context = self._incident_inputs.get(request.incident_id)
        if context is None:
            payload, confidence, warnings = self.incident_service.analyze_risk(request)
            warnings = list(warnings)
            warnings.append("ml context missing for incident; fallback risk path used")
            payload["model_source"] = "fallback"
            return MLAdapterStageResult(payload=payload, confidence=confidence, warnings=warnings)

        try:
            payload, confidence = self._run_model_backed_risk(context, request.lookahead_hours)
            payload["model_source"] = "ml"
            self.repository.save_stage(
                request.incident_id,
                "risk",
                payload,
                confidence,
                [],
            )
            return MLAdapterStageResult(payload=payload, confidence=confidence, warnings=[])
        except Exception as exc:
            payload, confidence, warnings = self.incident_service.analyze_risk(request)
            warnings = list(warnings)
            warnings.append(f"ml risk execution failed; fallback used: {exc}")
            payload["model_source"] = "fallback"
            return MLAdapterStageResult(payload=payload, confidence=confidence, warnings=warnings)

    def run_plan(self, request: IncidentPlanRequest) -> MLAdapterStageResult:
        payload, confidence, warnings = self.incident_service.plan_incident(request)
        return MLAdapterStageResult(payload=payload, confidence=confidence, warnings=warnings)

    def run_optimize(self, request: IncidentOptimizeRequest) -> MLAdapterStageResult:
        payload, confidence, warnings = self.incident_service.optimize_incident(request)
        return MLAdapterStageResult(payload=payload, confidence=confidence, warnings=warnings)

    def run_simulate(self, request: IncidentSimulateRequest) -> MLAdapterStageResult:
        payload, confidence, warnings = self.incident_service.simulate_incident(request)
        return MLAdapterStageResult(payload=payload, confidence=confidence, warnings=warnings)

    def run_report(self, incident_id: str) -> MLAdapterStageResult:
        payload, confidence, warnings = self.incident_service.generate_report(incident_id)
        return MLAdapterStageResult(payload=payload, confidence=confidence, warnings=warnings)

    def _build_context_from_paths(self, request: IngestBatchRequest) -> _IncidentInputContext:
        """Build typed integration context from user-provided dataset paths."""
        assets = load_asset_master(request.asset_path) if request.asset_path else []
        telemetry_df = load_telemetry(request.telemetry_path) if request.telemetry_path else pd.DataFrame()

        if not assets and not telemetry_df.empty:
            assets = self._derive_assets_from_telemetry(telemetry_df)

        events = load_events(request.event_path) if request.event_path else []
        maintenance = load_maintenance(request.maintenance_path) if request.maintenance_path else []
        failures = load_failures(request.failure_path) if request.failure_path else []

        if telemetry_df.empty or not assets:
            raise ValueError("path-based ingest requires telemetry and asset context")

        return _IncidentInputContext(
            assets=assets,
            telemetry_df=telemetry_df,
            events=events,
            maintenance_logs=maintenance,
            failures=failures,
            source_mode="paths",
        )

    def _build_context_from_direct_request(self, request: IngestBatchRequest) -> _IncidentInputContext:
        """Build small deterministic dataset for model execution in direct mode."""
        now = datetime.utcnow()
        timestamps = pd.date_range(end=now, periods=192, freq="15min")

        assets = [
            AssetMaster(
                asset_id="PUMP-0042",
                asset_type=AssetType.PUMP,
                site_id="SITE-ALPHA",
                installation_date=now - timedelta(days=1300),
                maintenance_policy=MaintenancePolicy.CONDITION_BASED,
                criticality_tier=CriticalityTier.TIER_1,
            ),
            AssetMaster(
                asset_id="PUMP-0101",
                asset_type=AssetType.PUMP,
                site_id="SITE-ALPHA",
                installation_date=now - timedelta(days=900),
                maintenance_policy=MaintenancePolicy.SCHEDULED,
                criticality_tier=CriticalityTier.TIER_2,
            ),
            AssetMaster(
                asset_id="PUMP-0110",
                asset_type=AssetType.PUMP,
                site_id="SITE-BETA",
                installation_date=now - timedelta(days=600),
                maintenance_policy=MaintenancePolicy.SCHEDULED,
                criticality_tier=CriticalityTier.TIER_3,
            ),
        ]

        rng = np.random.default_rng(42)
        rows: list[dict] = []
        sensors = [
            ("vibration_x", "mm/s", 2.4, 0.15),
            ("vibration_y", "mm/s", 2.1, 0.14),
            ("temperature", "°C", 66.0, 0.5),
            ("pressure", "bar", 4.4, 0.08),
            ("flow_rate", "m3/h", 11.5, 0.2),
        ]

        for asset in assets:
            for ts_idx, ts in enumerate(timestamps):
                trend = 0.0
                if asset.asset_id == "PUMP-0042" and ts_idx > len(timestamps) * 0.65:
                    trend = (ts_idx - len(timestamps) * 0.65) / len(timestamps)
                for name, unit, base, noise in sensors:
                    value = float(base + rng.normal(0, noise))
                    if asset.asset_id == "PUMP-0042" and name in {"vibration_x", "vibration_y"}:
                        value += float(trend * 1.6)
                    rows.append(
                        {
                            "timestamp": ts,
                            "asset_id": asset.asset_id,
                            "sensor_name": name,
                            "sensor_value": value,
                            "unit": unit,
                            "quality_flag": "ok",
                        }
                    )

        telemetry_df = pd.DataFrame(rows)

        events = [
            EventLog(
                event_id="evt-1",
                timestamp=now - timedelta(hours=4),
                asset_id="PUMP-0042",
                event_type=EventType.WARNING,
                severity=Severity.WARNING,
                event_text="bearing vibration rising",
            )
        ]

        maintenance_logs = [
            MaintenanceLog(
                work_order_id="wo-100",
                asset_id="PUMP-0042",
                timestamp=now - timedelta(days=420),
                action_type=ActionType.INSPECTION,
                parts_used=[],
                duration_minutes=45,
                cost=250.0,
                outcome=MaintenanceOutcome.SUCCESS,
            )
        ]

        failures = [
            FailureGroundTruth(
                asset_id="PUMP-0042",
                failure_time=now + timedelta(hours=24),
                failure_mode=FailureMode.BEARING_FAILURE,
                impact_cost=6200.0,
                downtime_minutes=300,
            )
        ]

        # Honor request size hints for confidence sensitivity while keeping runtime stable.
        source_mode = "direct"
        if request.telemetry_rows and request.telemetry_rows < 1000:
            telemetry_df = telemetry_df.tail(telemetry_df.shape[0] // 2)
            source_mode = "direct-sparse"

        return _IncidentInputContext(
            assets=assets,
            telemetry_df=telemetry_df,
            events=events,
            maintenance_logs=maintenance_logs,
            failures=failures,
            source_mode=source_mode,
        )

    def _derive_assets_from_telemetry(self, telemetry_df: pd.DataFrame) -> list[AssetMaster]:
        """Create minimal asset master records when only telemetry is supplied."""
        now = datetime.utcnow()
        assets: list[AssetMaster] = []
        for idx, asset_id in enumerate(sorted(set(telemetry_df["asset_id"].astype(str).tolist()))):
            assets.append(
                AssetMaster(
                    asset_id=asset_id,
                    asset_type=AssetType.PUMP,
                    site_id="SITE-UNKNOWN",
                    installation_date=now - timedelta(days=365 * (idx + 1)),
                    maintenance_policy=MaintenancePolicy.SCHEDULED,
                    criticality_tier=CriticalityTier.TIER_3,
                )
            )
        return assets

    def _run_model_backed_risk(self, context: _IncidentInputContext, lookahead_hours: int) -> tuple[dict, float]:
        """Run anomaly + failure probability models and return normalized backend payload."""
        detector = AnomalyDetector(contamination=0.15, n_estimators=64, random_state=42)
        failure_ids = {f.asset_id for f in context.failures}
        detector.fit(
            telemetry_df=context.telemetry_df,
            assets=context.assets,
            failure_asset_ids=failure_ids,
            window_hours=24,
            stride_hours=6,
        )

        anomaly_scores = detector.score(context.telemetry_df, window_hours=24)
        if anomaly_scores.empty:
            raise ValueError("anomaly model produced no scores")

        features_df = compute_rolling_features_all_windows(
            context.telemetry_df,
            window_hours=24,
            stride_hours=6,
        )
        if features_df.empty:
            raise ValueError("insufficient telemetry windows for risk model")

        normalized_failures = [
            FailureGroundTruth(
                asset_id=f.asset_id,
                failure_time=(
                    pd.to_datetime(f.failure_time).tz_localize(None).to_pydatetime()
                    if pd.to_datetime(f.failure_time).tzinfo is not None
                    else pd.to_datetime(f.failure_time).to_pydatetime()
                ),
                failure_mode=f.failure_mode,
                impact_cost=f.impact_cost,
                downtime_minutes=f.downtime_minutes,
            )
            for f in context.failures
        ]

        labels = build_failure_labels(features_df, normalized_failures, horizon_hours=lookahead_hours)
        enriched = enrich_features(
            features_df,
            assets=context.assets,
            maintenance_logs=context.maintenance_logs,
            events=context.events,
            anomaly_scores=anomaly_scores[["asset_id", "anomaly_score"]],
        )

        risk_model = FailureRiskModel(horizon_hours=lookahead_hours, train_ratio=0.75, random_state=42)
        metrics = risk_model.fit(enriched, labels)

        latest_rows = (
            enriched.sort_values("window_end")
            .groupby("asset_id", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
        predictions = risk_model.predict(latest_rows)
        if predictions.empty:
            raise ValueError("risk model produced no predictions")

        top = predictions.sort_values("failure_probability", ascending=False).iloc[0]
        anomaly_map = dict(zip(anomaly_scores["asset_id"], anomaly_scores["anomaly_score"]))
        top_asset_id = str(top["asset_id"])
        top_anomaly = float(anomaly_map.get(top_asset_id, 0.0))
        probability = float(top["failure_probability"])

        risk_band = "high" if probability >= 0.7 else "medium" if probability >= 0.4 else "low"
        confidence = max(0.2, min(0.95, 1.0 - float(top["confidence_upper"] - top["confidence_lower"])))

        payload = {
            "lookahead_hours": lookahead_hours,
            "asset_id": top_asset_id,
            "failure_probability": round(probability, 4),
            "anomaly_score": round(top_anomaly, 4),
            "failure_horizon_hours": int(top["failure_horizon_hours"]),
            "risk_band": risk_band,
            "training_metrics": metrics,
            "assumptions": [
                "latest per-asset rolling window is representative of short-term failure risk",
                "anomaly and failure models are calibrated on adapter-provided context",
            ],
            "evidence_refs": [
                "ml.aegis.models.anomaly.AnomalyDetector",
                "ml.aegis.models.failure_risk.FailureRiskModel",
            ],
        }
        return payload, confidence
