"""Single-call full incident pipeline route."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from backend.contracts import RunIncidentRequest
from backend.models import ResponseEnvelope, build_envelope
from backend.security import require_api_key
from backend.services import PipelineService, get_pipeline_service

router = APIRouter(tags=["pipeline"], dependencies=[Depends(require_api_key)])


@router.post("/incident/run", response_model=ResponseEnvelope)
def run_incident_pipeline(
    request_context: Request,
    request: RunIncidentRequest,
    service: PipelineService = Depends(get_pipeline_service),
) -> ResponseEnvelope:
    """Run the full decision loop in a single request."""
    payload, confidence, warnings = service.run_full_pipeline(request)
    return build_envelope(
        request_id=request_context.state.request_id,
        payload=payload,
        confidence=confidence,
        warnings=warnings,
    )