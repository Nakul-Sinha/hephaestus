"""Ingest route implementation."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends

from backend.contracts import IngestBatchRequest
from backend.models import ResponseEnvelope, build_envelope
from backend.services import IncidentService, get_incident_service

router = APIRouter(tags=["ingest"])


@router.post("/ingest/batch", response_model=ResponseEnvelope)
def ingest_batch(
    request: IngestBatchRequest,
    service: IncidentService = Depends(get_incident_service),
) -> ResponseEnvelope:
    """Create a new incident context from a batch ingest request."""
    payload = service.ingest_batch(request)
    warnings = payload["ingest"].get("warnings", [])
    confidence = float(payload["ingest"].get("confidence", 1.0))
    return build_envelope(
        request_id=f"req-{uuid4().hex[:10]}",
        payload=payload,
        confidence=confidence,
        warnings=warnings,
    )