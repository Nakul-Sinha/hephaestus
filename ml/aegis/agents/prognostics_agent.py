"""
Prognostics Agent (Agent 4)

Responsibilities:
    - Run failure risk model on at-risk assets from Sentinel
    - Estimate failure probability and time-to-failure
    - Generate SHAP explanations for each prediction
    - Abstain-and-escalate when uncertainty is too wide

Next agent: causal_agent.
"""

from __future__ import annotations

from typing import Any

from ml.aegis.agents import BaseAgent
from ml.aegis.data.schemas import AgentOutput
from ml.aegis.models.anomaly import compute_rolling_features
from ml.aegis.models.explainability import ModelExplainer, get_top_risk_drivers
from ml.aegis.models.failure_risk import FailureRiskModel, enrich_features


class PrognosticsAgent(BaseAgent):

    name = "prognostics_agent"
    description = "Estimates failure probability, RUL, and generates explanations"

    def __init__(
        self,
        risk_model: FailureRiskModel | None = None,
        explainer: ModelExplainer | None = None,
    ):
        self._risk_model = risk_model
        self._explainer = explainer or ModelExplainer()

    def run(self, context: dict[str, Any]) -> AgentOutput:
        telemetry_df = context.get("telemetry_df")
        assets = context.get("assets", [])
        at_risk_ids = context.get("at_risk_asset_ids", set())
        anomaly_scores = context.get("anomaly_scores")
        maintenance = context.get("maintenance", [])
        events = context.get("events", [])

        risk_model = self._risk_model or context.get("risk_model")

        if risk_model is None:
            return self._build_output(
                payload={"status": "no_model"},
                confidence=0.0,
                errors=["No trained FailureRiskModel available"],
                next_agent="reporter_agent",
            )

        if not at_risk_ids:
            return self._build_output(
                payload={"status": "no_at_risk_assets", "predictions": []},
                confidence=0.90,
                evidence=["No assets flagged by Sentinel — fleet is healthy"],
                next_agent="reporter_agent",
            )

        evidence: list[str] = []
        errors: list[str] = []

        # Filter telemetry to at-risk assets only
        risk_telemetry = telemetry_df[telemetry_df["asset_id"].isin(at_risk_ids)]

        # Compute features
        features_df = compute_rolling_features(risk_telemetry)
        features_df = enrich_features(
            features_df, assets, maintenance, events, anomaly_scores,
        )

        # Predict failure probability
        predictions_df = risk_model.predict(features_df)
        evidence.append(f"Predicted failure risk for {len(predictions_df)} assets")

        # Generate explanations
        explanations = []
        feature_columns = risk_model._feature_columns
        if feature_columns and hasattr(risk_model._model, "predict_proba"):
            try:
                explanations = self._explainer.explain_batch(
                    model=risk_model._model,
                    features_df=features_df,
                    feature_columns=feature_columns,
                    model_name="failure_risk",
                    top_k=5,
                )
                evidence.append(f"Generated SHAP explanations for {len(explanations)} assets")
            except Exception as e:
                errors.append(f"SHAP explanation failed: {e}")

        # Build per-asset prediction summaries
        prediction_summaries = []
        high_uncertainty_count = 0

        for _, row in predictions_df.iterrows():
            ci_width = row["confidence_upper"] - row["confidence_lower"]
            is_uncertain = ci_width > 0.40

            if is_uncertain:
                high_uncertainty_count += 1

            # Find matching explanation
            asset_explanation = None
            risk_drivers = []
            for exp in explanations:
                if exp.asset_id == row["asset_id"]:
                    asset_explanation = exp
                    risk_drivers = get_top_risk_drivers(exp)
                    break

            summary = {
                "asset_id": row["asset_id"],
                "failure_probability": row["failure_probability"],
                "failure_horizon_hours": row["failure_horizon_hours"],
                "risk_level": row["risk_level"],
                "confidence_lower": row["confidence_lower"],
                "confidence_upper": row["confidence_upper"],
                "high_uncertainty": is_uncertain,
                "top_risk_drivers": risk_drivers[:3],
            }
            prediction_summaries.append(summary)

        # Sort by failure probability descending
        prediction_summaries.sort(key=lambda x: x["failure_probability"], reverse=True)

        # Confidence based on prediction quality
        quality_modifier = context.get("quality_confidence_modifier", 1.0)
        critical_count = sum(1 for p in prediction_summaries if p["risk_level"] == "critical")

        if high_uncertainty_count > len(prediction_summaries) * 0.5:
            confidence = 0.50 * quality_modifier
            errors.append(
                f"{high_uncertainty_count} assets have high uncertainty — "
                "recommend manual inspection for validation"
            )
        elif critical_count > 0:
            confidence = 0.85 * quality_modifier
        else:
            confidence = 0.80 * quality_modifier

        evidence.append(f"Critical risk: {critical_count} assets")
        evidence.append(f"High uncertainty: {high_uncertainty_count} assets")

        # Store in context
        context["risk_predictions"] = predictions_df
        context["risk_features"] = features_df
        context["explanations"] = explanations
        context["prediction_summaries"] = prediction_summaries

        return self._build_output(
            payload={
                "status": "completed",
                "assets_analyzed": len(predictions_df),
                "critical_count": critical_count,
                "high_risk_count": sum(1 for p in prediction_summaries if p["risk_level"] in ("critical", "high")),
                "high_uncertainty_count": high_uncertainty_count,
                "predictions": prediction_summaries[:20],
            },
            confidence=round(confidence, 4),
            assumptions=[
                f"Failure horizon is {self._risk_model.horizon_hours if self._risk_model else 48}h lookahead",
                "Confidence intervals are approximate (±uncertainty based on calibration)",
                "SHAP values computed on the failure risk model's feature space",
            ],
            evidence=evidence,
            errors=errors if errors else None,
            next_agent="causal_agent",
        )
