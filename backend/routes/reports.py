"""Reporting route implementation."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from backend.models import ResponseEnvelope, build_envelope
from backend.services import IncidentService, get_incident_service
from backend.storage import IncidentNotFoundError

router = APIRouter(tags=["reports"])


@router.get("/incident/{incident_id}/report", response_model=ResponseEnvelope)
def get_report(
    incident_id: str,
    service: IncidentService = Depends(get_incident_service),
) -> ResponseEnvelope:
    """Return stakeholder report payload for an incident."""
    try:
        payload, confidence, warnings = service.generate_report(incident_id)
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