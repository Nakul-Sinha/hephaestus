"""API route registration."""

from fastapi import APIRouter

from backend.routes.health import router as health_router
from backend.routes.ingest import router as ingest_router
from backend.routes.pipeline import router as pipeline_router
from backend.routes.planning import router as planning_router
from backend.routes.reports import router as reports_router
from backend.routes.risk import router as risk_router
from backend.routes.simulation import router as simulation_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(ingest_router)
api_router.include_router(pipeline_router)
api_router.include_router(risk_router)
api_router.include_router(planning_router)
api_router.include_router(simulation_router)
api_router.include_router(reports_router)

__all__ = ["api_router"]
