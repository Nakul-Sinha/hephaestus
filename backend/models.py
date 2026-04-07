"""Shared API models for backend responses."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ResponseEnvelope(BaseModel):
    """Standard API response envelope for every endpoint."""

    request_id: str
    status: Literal["success", "error"]
    timestamp: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


def build_envelope(
    request_id: str,
    payload: dict[str, Any],
    confidence: float = 1.0,
    warnings: list[str] | None = None,
    status: Literal["success", "error"] = "success",
) -> ResponseEnvelope:
    """Create a response envelope with UTC timestamp."""
    return ResponseEnvelope(
        request_id=request_id,
        status=status,
        timestamp=datetime.now(timezone.utc),
        payload=payload,
        confidence=confidence,
        warnings=warnings or [],
    )