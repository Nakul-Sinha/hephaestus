"""Service layer exports."""

from functools import lru_cache

from backend.services.governance_service import GovernanceService
from backend.services.incident_service import IncidentService
from backend.services.pipeline_service import PipelineService, build_pipeline_service
from backend.storage import get_incident_repository
from backend.config import get_settings


@lru_cache(maxsize=1)
def get_incident_service() -> IncidentService:
	"""Return singleton incident service for request handlers."""
	return IncidentService(repository=get_incident_repository())


@lru_cache(maxsize=1)
def get_governance_service() -> GovernanceService:
	"""Return singleton governance service for request handlers."""
	return GovernanceService(settings=get_settings())


@lru_cache(maxsize=1)
def get_pipeline_service() -> PipelineService:
	"""Return singleton pipeline service for request handlers."""
	return build_pipeline_service(get_incident_service())


__all__ = [
	"GovernanceService",
	"IncidentService",
	"PipelineService",
	"get_governance_service",
	"get_incident_service",
	"get_pipeline_service",
]
