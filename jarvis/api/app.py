"""
JARVIS FastAPI application factory.

Usage
-----
    # Direct import (Uvicorn)
    uvicorn api.app:app --host 0.0.0.0 --port 8765

    # Programmatic
    from api.app import create_app
    app = create_app()
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings
from core.database import close_database, init_database
from core.logger import get_logger

log = get_logger(__name__)

# Track server start time for uptime calculation.
_START_TIME: float = time.time()


# --------------------------------------------------------------------------- #
# Lifespan                                                                       #
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Handle startup and shutdown lifecycle events."""
    global _START_TIME
    _START_TIME = time.time()

    # ---------------------------------------------------------------- startup
    log.info(
        "Starting {name} v{version} ({env})",
        name=settings.APP_NAME,
        version=settings.VERSION,
        env=settings.ENVIRONMENT,
    )
    await init_database()
    log.info("{name} is ready on {host}:{port}", name=settings.APP_NAME, host=settings.API_HOST, port=settings.API_PORT)

    yield  # Application runs here.

    # --------------------------------------------------------------- shutdown
    log.info("Shutting down {name}…", name=settings.APP_NAME)
    await close_database()
    log.info("{name} shutdown complete.", name=settings.APP_NAME)


# --------------------------------------------------------------------------- #
# Application factory                                                            #
# --------------------------------------------------------------------------- #


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="JARVIS — personal AI desktop assistant",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # -------------------------------------------------------- CORS middleware
    # Allow all origins for local desktop usage.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------- exception handlers
    @app.exception_handler(500)
    async def internal_server_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        log.error(
            "Unhandled exception on {method} {path}: {exc}",
            method=request.method,
            path=request.url.path,
            exc=exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "path": request.url.path,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        log.error(
            "Caught exception on {method} {path}: {exc}",
            method=request.method,
            path=request.url.path,
            exc=exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc) if settings.DEBUG else "Internal server error",
                "path": request.url.path,
            },
        )

    # --------------------------------------------------------- routers
    _register_routers(app)

    # ------------------------------------------------------------ root route
    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        return {
            "name": settings.APP_NAME,
            "version": settings.VERSION,
            "status": "running",
            "docs": "/docs" if settings.DEBUG else "disabled",
        }

    return app


def _register_routers(app: FastAPI) -> None:
    """Import and include all route modules."""
    from api.routes.health import router as health_router
    from api.routes.tasks import router as tasks_router
    from api.routes.commands import router as commands_router
    from api.routes.telegram import router as telegram_router

    app.include_router(health_router)
    app.include_router(tasks_router)
    app.include_router(commands_router)
    app.include_router(telegram_router)

    log.debug(
        "Registered routers: health, tasks, commands, telegram"
    )


def get_uptime() -> float:
    """Return seconds elapsed since the application started."""
    return time.time() - _START_TIME


# --------------------------------------------------------------------------- #
# Module-level app instance (used by Uvicorn)                                   #
# --------------------------------------------------------------------------- #

app: FastAPI = create_app()
