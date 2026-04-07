"""Single-call full incident pipeline route."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends

from backend.contracts import RunIncidentRequest
from backend.models import ResponseEnvelope, build_envelope
from backend.services import PipelineService, get_pipeline_service

router = APIRouter(tags=["pipeline"])


@router.post("/incident/run", response_model=ResponseEnvelope)
def run_incident_pipeline(
    request: RunIncidentRequest,
    service: PipelineService = Depends(get_pipeline_service),
) -> ResponseEnvelope:
    """Run the full decision loop in a single request."""
    payload, confidence, warnings = service.run_full_pipeline(request)
    return build_envelope(
        request_id=f"req-{uuid4().hex[:10]}",
        payload=payload,
        confidence=confidence,
        warnings=warnings,
    )