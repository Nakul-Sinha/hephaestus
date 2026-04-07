"""Incident planning and optimization routes."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from backend.contracts import IncidentOptimizeRequest, IncidentPlanRequest
from backend.models import ResponseEnvelope, build_envelope
from backend.services import IncidentService, get_incident_service
from backend.storage import IncidentNotFoundError

router = APIRouter(tags=["planning"])


@router.post("/incident/plan", response_model=ResponseEnvelope)
def plan_incident(
    request: IncidentPlanRequest,
    service: IncidentService = Depends(get_incident_service),
) -> ResponseEnvelope:
    """Generate intervention plan options for an incident."""
    try:
        payload, confidence, warnings = service.plan_incident(request)
    except IncidentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"incident not found: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return build_envelope(
        request_id=f"req-{uuid4().hex[:10]}",
        payload=payload,
        confidence=confidence,
        warnings=warnings,
    )


@router.post("/incident/optimize", response_model=ResponseEnvelope)
def optimize_incident(
    request: IncidentOptimizeRequest,
    service: IncidentService = Depends(get_incident_service),
) -> ResponseEnvelope:
    """Rank and select plans under constraints."""
    try:
        payload, confidence, warnings = service.optimize_incident(request)
    except IncidentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"incident not found: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return build_envelope(
        request_id=f"req-{uuid4().hex[:10]}",
        payload=payload,
        confidence=confidence,
        warnings=warnings,
    )