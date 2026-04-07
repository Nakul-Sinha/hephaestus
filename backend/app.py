"""Hephaestus backend FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI

from backend.config import get_settings
from backend.routes import api_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Hephaestus backend API for predictive maintenance workflows.",
)

app.include_router(api_router)
