"""API route registration."""

from fastapi import APIRouter

from backend.routes.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)

__all__ = ["api_router"]
