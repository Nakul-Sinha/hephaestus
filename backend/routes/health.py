"""Health route implementation."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter

from backend.config import dependency_health_snapshot, get_settings
from backend.models import ResponseEnvelope, build_envelope

router = APIRouter(tags=["health"])


@router.get("/health", response_model=ResponseEnvelope)
def health_check() -> ResponseEnvelope:
    """Return service and dependency configuration health."""
    settings = get_settings()
    dependencies = dependency_health_snapshot(settings)
    warnings = [
        f"{name} is not configured"
        for name, info in dependencies.items()
        if info["status"] != "configured"
    ]

    payload = {
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "dependencies": dependencies,
    }

    return build_envelope(
        request_id=f"req-{uuid4().hex[:10]}",
        payload=payload,
        confidence=1.0 if not warnings else 0.75,
        warnings=warnings,
    )