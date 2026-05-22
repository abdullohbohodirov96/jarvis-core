"""
Health-check endpoints.

GET /health        – liveness probe  (is the process alive?)
GET /health/ready  – readiness probe (are dependencies reachable?)
GET /health/info   – deployment info (version, uptime, environment)
"""

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()

# Record process start time once at import so /info can report uptime.
_PROCESS_START = time.time()


# ---------------------------------------------------------------------------
# GET /health  — liveness
# ---------------------------------------------------------------------------


@router.get(
    "",
    summary="Liveness check",
    description="Returns 200 OK if the process is running and accepting traffic.",
    response_description="Basic alive status.",
    status_code=status.HTTP_200_OK,
)
async def liveness() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ok", "message": "JARVIS is alive."},
    )


# ---------------------------------------------------------------------------
# GET /health/ready  — readiness
# ---------------------------------------------------------------------------


async def _check_database(request: Request) -> Dict[str, Any]:
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        return {"status": "unavailable", "detail": "Engine not initialised."}
    try:
        async with engine.connect() as conn:
            from sqlalchemy import text

            await conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


async def _check_redis(request: Request) -> Dict[str, Any]:
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        return {"status": "unavailable", "detail": "Client not initialised."}
    try:
        pong = await redis.ping()
        return {"status": "ok", "pong": pong}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


async def _check_openai() -> Dict[str, Any]:
    if not settings.OPENAI_API_KEY:
        return {"status": "not_configured"}
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            )
        if resp.status_code == 200:
            return {"status": "ok"}
        return {"status": "error", "http_status": resp.status_code}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@router.get(
    "/ready",
    summary="Readiness check",
    description=(
        "Verifies that all critical downstream dependencies (database, Redis, OpenAI) "
        "are reachable.  Returns 200 if all mandatory checks pass, 503 otherwise."
    ),
    status_code=status.HTTP_200_OK,
)
async def readiness(request: Request) -> JSONResponse:
    db_result = await _check_database(request)
    redis_result = await _check_redis(request)
    openai_result = await _check_openai()

    checks: Dict[str, Any] = {
        "database": db_result,
        "redis": redis_result,
        "openai": openai_result,
    }

    # The service is "ready" when the mandatory infrastructure is reachable.
    # OpenAI is optional (can be degraded).
    mandatory_healthy = all(
        checks[k]["status"] in ("ok",)
        for k in ("database", "redis")
        if checks[k]["status"] != "unavailable"  # not configured == skip
    )

    overall = "ok" if mandatory_healthy else "degraded"
    http_code = (
        status.HTTP_200_OK if mandatory_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(
        status_code=http_code,
        content={"status": overall, "checks": checks},
    )


# ---------------------------------------------------------------------------
# GET /health/info  — system info
# ---------------------------------------------------------------------------


@router.get(
    "/info",
    summary="System information",
    description="Returns build/deployment metadata and process uptime.",
    status_code=status.HTTP_200_OK,
)
async def info() -> JSONResponse:
    uptime_seconds = time.time() - _PROCESS_START
    hours, remainder = divmod(int(uptime_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "uptime": {
                "seconds": round(uptime_seconds, 2),
                "human": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            },
        },
    )
