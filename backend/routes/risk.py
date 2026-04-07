"""Risk route implementation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.contracts import RiskAnalyzeRequest
from backend.models import ResponseEnvelope, build_envelope
from backend.security import require_api_key
from backend.services import PipelineService, get_pipeline_service
from backend.storage import IncidentNotFoundError

router = APIRouter(tags=["risk"], dependencies=[Depends(require_api_key)])


@router.post("/risk/analyze", response_model=ResponseEnvelope)
def analyze_risk(
    request_context: Request,
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
        request_id=request_context.state.request_id,
        payload=payload,
        confidence=confidence,
        warnings=warnings,
    )