import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai.agent import NexusBrain
from core.config import settings
from database.connection import check_db_connection, init_db
from telegram.client import TelegramAutomator
from utils.logger import logger

# ── Module-level singletons ────────────────────────────────────────────────────
#
# These are created once at import time so that both the lifespan handler and
# any future routers can reference the same objects.

_brain: NexusBrain = NexusBrain()
_automator: TelegramAutomator = TelegramAutomator(brain=_brain)
_start_time: float = time.monotonic()


# ── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler (replaces deprecated ``@app.on_event``).

    Startup
    -------
    1. Initialise all database tables (idempotent).
    2. Spawn the Telegram listener as a non-blocking background task so it
       runs concurrently with the ASGI server event loop.

    Shutdown
    --------
    1. Gracefully disconnect the Telegram client.
    """
    # ── Startup ────────────────────────────────────────────────────────────
    logger.info("Starting {} (env={})", settings.PROJECT_NAME, settings.ENVIRONMENT)

    await init_db()

    telegram_task = asyncio.create_task(
        _automator.start(),
        name="telegram_daemon",
    )

    def _on_telegram_done(fut: asyncio.Future) -> None:
        if not fut.cancelled() and fut.exception():
            logger.error(
                "Telegram daemon exited with exception: {}", fut.exception()
            )

    telegram_task.add_done_callback(_on_telegram_done)
    logger.success("{} startup complete.", settings.PROJECT_NAME)

    yield  # ── Application is running ─────────────────────────────────────

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("Shutting down {}…", settings.PROJECT_NAME)
    await _automator.stop()
    telegram_task.cancel()
    try:
        await telegram_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutdown complete.")


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description=(
        "NEXUS — Elite personal AI assistant backend. "
        "Telegram automation · task tracking · AI-powered responses."
    ),
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get(
    "/health",
    summary="Health check",
    response_description="Service health status",
    tags=["System"],
)
async def health() -> JSONResponse:
    """
    Returns the operational status of NEXUS and its dependencies.

    Response shape::

        {
            "status":   "ok" | "degraded",
            "project":  "NEXUS-Core",
            "env":      "production",
            "uptime_s": 42.3,
            "database": "ok" | "unreachable"
        }

    HTTP 200 is returned even when ``status`` is ``"degraded"`` so that
    load-balancer health probes remain non-disruptive.  Inspect the body for
    the true component-level state.
    """
    db_ok: bool = await check_db_connection()
    uptime: float = round(time.monotonic() - _start_time, 2)

    payload = {
        "status": "ok" if db_ok else "degraded",
        "project": settings.PROJECT_NAME,
        "env": settings.ENVIRONMENT,
        "uptime_s": uptime,
        "database": "ok" if db_ok else "unreachable",
    }

    logger.debug("Health check | {}", payload)
    return JSONResponse(content=payload, status_code=200)
