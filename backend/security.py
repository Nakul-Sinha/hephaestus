"""Security dependencies for backend endpoints."""

from __future__ import annotations

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from backend.config import get_settings

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    """Validate API key for write operations in MVP."""
    settings = get_settings()
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing api key",
        )