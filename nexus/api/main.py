import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from utils.logger import logger

_start_time: float = time.monotonic()

# Track startup state so /health can report it honestly
_db_ready: bool = False
_telegram_ready: bool = False
_telegram_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _db_ready, _telegram_ready, _telegram_task

    logger.info("Starting {} (env={})", settings.PROJECT_NAME, settings.ENVIRONMENT)

    # ── 1. Database ────────────────────────────────────────────────────────
    try:
        from database.connection import init_db
        await init_db()
        _db_ready = True
        logger.success("Database ready.")
    except Exception as exc:
        # Log and continue — /health will report 'degraded' but the HTTP
        # server still starts so Render's health probe gets a 200 response.
        logger.error("Database init failed (will retry on next deploy): {}", exc)

    # ── 2. Telegram daemon ─────────────────────────────────────────────────
    try:
        from ai.agent import NexusBrain
        from telegram.client import TelegramAutomator

        brain = NexusBrain()
        automator = TelegramAutomator(brain=brain)

        _telegram_task = asyncio.create_task(
            automator.start(),
            name="telegram_daemon",
        )

        def _on_done(fut: asyncio.Future) -> None:
            global _telegram_ready
            if not fut.cancelled() and fut.exception():
                _telegram_ready = False
                logger.error("Telegram daemon crashed: {}", fut.exception())

        _telegram_task.add_done_callback(_on_done)
        _telegram_ready = True
        logger.success("Telegram daemon started.")
    except Exception as exc:
        logger.error("Telegram init failed: {}", exc)

    logger.success("{} startup complete.", settings.PROJECT_NAME)

    yield  # ── serving requests ───────────────────────────────────────────

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("Shutting down {}…", settings.PROJECT_NAME)
    if _telegram_task and not _telegram_task.done():
        _telegram_task.cancel()
        try:
            await _telegram_task
        except asyncio.CancelledError:
            pass
    logger.info("Shutdown complete.")


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


@app.get("/health", tags=["System"], summary="Health check")
async def health() -> JSONResponse:
    from database.connection import check_db_connection

    db_ok: bool = False
    try:
        db_ok = await check_db_connection()
    except Exception:
        pass

    uptime = round(time.monotonic() - _start_time, 2)
    overall = "ok" if (db_ok and _telegram_ready) else "degraded"

    return JSONResponse(
        content={
            "status": overall,
            "project": settings.PROJECT_NAME,
            "env": settings.ENVIRONMENT,
            "uptime_s": uptime,
            "database": "ok" if db_ok else "unreachable",
            "telegram": "ok" if _telegram_ready else "unavailable",
        },
        status_code=200,
    )
