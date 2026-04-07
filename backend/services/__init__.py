"""Service layer exports."""

from functools import lru_cache

from backend.services.incident_service import IncidentService
from backend.storage import get_incident_repository


@lru_cache(maxsize=1)
def get_incident_service() -> IncidentService:
	"""Return singleton incident service for request handlers."""
	return IncidentService(repository=get_incident_repository())


__all__ = ["IncidentService", "get_incident_service"]
