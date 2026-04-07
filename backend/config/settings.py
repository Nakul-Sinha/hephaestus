"""Runtime settings for the backend service."""

from __future__ import annotations

from functools import lru_cache
from os import getenv

from pydantic import BaseModel, Field


class BackendSettings(BaseModel):
    """Application settings loaded from environment variables."""

    app_name: str = Field(default="hephaestus-backend")
    app_version: str = Field(default="0.1.0")
    environment: str = Field(default=getenv("HEPHAESTUS_ENV", "dev"))

    api_key: str = Field(default=getenv("HEPHAESTUS_API_KEY", "hephaestus-dev-key"))
    postgres_dsn: str = Field(default=getenv("HEPHAESTUS_POSTGRES_DSN", ""))
    redis_url: str = Field(default=getenv("HEPHAESTUS_REDIS_URL", ""))
    ollama_url: str = Field(default=getenv("HEPHAESTUS_OLLAMA_URL", ""))
    gemini_api_key: str = Field(default=getenv("HEPHAESTUS_GEMINI_API_KEY", ""))

    confidence_floor: float = Field(default=0.60, ge=0.0, le=1.0)


@lru_cache(maxsize=1)
def get_settings() -> BackendSettings:
    """Return cached settings to avoid repeated env parsing."""
    return BackendSettings()


def dependency_health_snapshot(settings: BackendSettings) -> dict[str, dict[str, str]]:
    """Expose configuration-level dependency status for health endpoint."""
    return {
        "postgres": {
            "status": "configured" if settings.postgres_dsn else "missing",
            "detail": "dsn provided" if settings.postgres_dsn else "set HEPHAESTUS_POSTGRES_DSN",
        },
        "redis": {
            "status": "configured" if settings.redis_url else "missing",
            "detail": "url provided" if settings.redis_url else "set HEPHAESTUS_REDIS_URL",
        },
        "ollama": {
            "status": "configured" if settings.ollama_url else "missing",
            "detail": "url provided" if settings.ollama_url else "set HEPHAESTUS_OLLAMA_URL",
        },
        "gemini": {
            "status": "configured" if settings.gemini_api_key else "missing",
            "detail": "api key provided" if settings.gemini_api_key else "set HEPHAESTUS_GEMINI_API_KEY",
        },
    }