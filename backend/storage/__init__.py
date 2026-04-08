"""Storage exports and repository factory for backend services."""

from __future__ import annotations

from functools import lru_cache

from backend.config import get_settings
from backend.storage.db import postgres_driver_available
from backend.storage.memory_repository import IncidentNotFoundError, InMemoryIncidentRepository
from backend.storage.repositories import IncidentRepository, PostgresIncidentRepository


@lru_cache(maxsize=1)
def get_incident_repository() -> IncidentRepository:
	"""Return repository implementation based on runtime configuration."""
	settings = get_settings()
	if settings.postgres_dsn and postgres_driver_available():
		return PostgresIncidentRepository(dsn=settings.postgres_dsn)
	return InMemoryIncidentRepository()


__all__ = [
	"IncidentNotFoundError",
	"IncidentRepository",
	"InMemoryIncidentRepository",
	"PostgresIncidentRepository",
	"get_incident_repository",
]
