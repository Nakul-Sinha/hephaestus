"""Risk route implementation."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from backend.contracts import RiskAnalyzeRequest
from backend.models import ResponseEnvelope, build_envelope
from backend.services import PipelineService, get_pipeline_service
from backend.storage import IncidentNotFoundError

router = APIRouter(tags=["risk"])


@router.post("/risk/analyze", response_model=ResponseEnvelope)
def analyze_risk(
    request: RiskAnalyzeRequest,
    service: PipelineService = Depends(get_pipeline_service),
) -> ResponseEnvelope:
    """Analyze failure risk for an existing incident."""
    try:
        payload, confidence, warnings = service.analyze_risk(request)
    except IncidentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"incident not found: {exc}",
        ) from exc

    return build_envelope(
        request_id=f"req-{uuid4().hex[:10]}",
        payload=payload,
        confidence=confidence,
        warnings=warnings,
    )