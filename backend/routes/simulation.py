"""Simulation route implementation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.contracts import IncidentSimulateRequest
from backend.models import ResponseEnvelope, build_envelope
from backend.security import require_api_key
from backend.services import PipelineService, get_pipeline_service
from backend.storage import IncidentNotFoundError

router = APIRouter(tags=["simulation"], dependencies=[Depends(require_api_key)])


@router.post("/incident/simulate", response_model=ResponseEnvelope)
def simulate_incident(
    request_context: Request,
    request: IncidentSimulateRequest,
    service: PipelineService = Depends(get_pipeline_service),
) -> ResponseEnvelope:
    """Run what-if simulation for ranked plans."""
    try:
        payload, confidence, warnings = service.simulate_incident(request)
    except IncidentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"incident not found: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return build_envelope(
        request_id=request_context.state.request_id,
        payload=payload,
        confidence=confidence,
        warnings=warnings,
    )