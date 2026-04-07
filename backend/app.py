"""Hephaestus backend FastAPI application."""

from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.config import get_settings
from backend.models import build_envelope
from backend.routes import api_router

settings = get_settings()
logger = logging.getLogger("hephaestus.backend")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Hephaestus backend API for predictive maintenance workflows.",
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Attach request id and emit structured request logs."""
    request_id = request.headers.get("x-request-id", f"req-{uuid4().hex[:10]}")
    request.state.request_id = request_id

    started = perf_counter()
    response = await call_next(request)
    duration_ms = round((perf_counter() - started) * 1000, 2)

    response.headers["x-request-id"] = request_id
    logger.info(
        "request_completed method=%s path=%s status=%s request_id=%s duration_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        request_id,
        duration_ms,
    )
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return standardized envelope for HTTP exceptions."""
    request_id = getattr(request.state, "request_id", f"req-{uuid4().hex[:10]}")
    envelope = build_envelope(
        request_id=request_id,
        status="error",
        confidence=0.0,
        warnings=[],
        payload={"error": str(exc.detail)},
    )
    return JSONResponse(status_code=exc.status_code, content=envelope.model_dump(mode="json"))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return standardized envelope for unhandled exceptions."""
    logger.exception("unhandled_exception request_path=%s", request.url.path)
    request_id = getattr(request.state, "request_id", f"req-{uuid4().hex[:10]}")
    envelope = build_envelope(
        request_id=request_id,
        status="error",
        confidence=0.0,
        warnings=["unexpected backend exception"],
        payload={"error": "internal server error"},
    )
    return JSONResponse(status_code=500, content=envelope.model_dump(mode="json"))

app.include_router(api_router)
