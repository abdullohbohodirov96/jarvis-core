"""
Health check routes for JARVIS.

Prefix: /api/health

Endpoints
---------
GET /         — liveness probe
GET /ready    — readiness probe (checks DB)
GET /info     — full system information
"""

from __future__ import annotations

import platform
import sys
from typing import Any

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from config.settings import settings
from core.database import check_database
from core.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/health", tags=["health"])


# --------------------------------------------------------------------------- #
# GET /api/health/                                                               #
# --------------------------------------------------------------------------- #


@router.get(
    "/",
    summary="Liveness probe",
    response_description="Returns ok when the process is alive",
)
async def liveness() -> dict[str, Any]:
    """Simple liveness check — always returns 200 when the process is running."""
    from api.app import get_uptime  # local import to avoid circular at module load

    return {
        "status": "ok",
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "uptime": round(get_uptime(), 2),
    }


# --------------------------------------------------------------------------- #
# GET /api/health/ready                                                          #
# --------------------------------------------------------------------------- #


@router.get(
    "/ready",
    summary="Readiness probe",
    response_description="Returns component status — 200 if all healthy",
)
async def readiness() -> JSONResponse:
    """Readiness check — verifies that all critical subsystems are reachable."""
    db_ok = await check_database()

    components: dict[str, Any] = {
        "database": {
            "status": "ok" if db_ok else "degraded",
            "url": settings.DATABASE_URL,
        },
    }

    overall = "ok" if all(
        c["status"] == "ok" for c in components.values()
    ) else "degraded"

    http_status = (
        status.HTTP_200_OK if overall == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(
        status_code=http_status,
        content={
            "status": overall,
            "components": components,
        },
    )


# --------------------------------------------------------------------------- #
# GET /api/health/info                                                           #
# --------------------------------------------------------------------------- #

# Fields that must never appear in the /info response.
_SECRET_FIELDS = {
    "API_SECRET_KEY",
    "OPENAI_API_KEY",
    "TELEGRAM_API_HASH",
    "TELEGRAM_API_ID",
    "TELEGRAM_PHONE",
}


@router.get(
    "/info",
    summary="System information",
    response_description="Full system info with sanitised settings",
)
async def system_info() -> dict[str, Any]:
    """Return full system and configuration information, with secrets redacted."""
    from api.app import get_uptime

    # Build a sanitised settings snapshot.
    raw: dict[str, Any] = settings.model_dump()
    safe_settings: dict[str, Any] = {
        key: ("***REDACTED***" if key in _SECRET_FIELDS else value)
        for key, value in raw.items()
    }

    return {
        "app": {
            "name": settings.APP_NAME,
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "uptime_seconds": round(get_uptime(), 2),
        },
        "runtime": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "node": platform.node(),
        },
        "configuration": safe_settings,
    }
