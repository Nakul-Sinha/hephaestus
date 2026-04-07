"""Ingest route implementation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from backend.contracts import IngestBatchRequest
from backend.models import ResponseEnvelope, build_envelope
from backend.security import require_api_key
from backend.services import PipelineService, get_pipeline_service

router = APIRouter(tags=["ingest"], dependencies=[Depends(require_api_key)])


@router.post("/ingest/batch", response_model=ResponseEnvelope)
def ingest_batch(
    request_context: Request,
    request: IngestBatchRequest,
    service: PipelineService = Depends(get_pipeline_service),
) -> ResponseEnvelope:
    """Create a new incident context from a batch ingest request."""
    payload, confidence, warnings = service.ingest_batch(request)
    return build_envelope(
        request_id=request_context.state.request_id,
        payload=payload,
        confidence=confidence,
        warnings=warnings,
    )