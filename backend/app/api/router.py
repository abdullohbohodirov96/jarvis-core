"""
Master API router — aggregates all v1 sub-routers under /api/v1.

Sub-routers are imported individually so that a missing optional dependency
in one module does not prevent the rest of the API from loading.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.v1.endpoints.health import router as health_router
from backend.app.api.v1.endpoints.chat import router as chat_router
from backend.app.api.v1.endpoints.tasks import router as tasks_router
from backend.app.api.v1.endpoints.voice import router as voice_router
from backend.app.api.v1.endpoints.telegram import router as telegram_router
from backend.app.api.v1.endpoints.memory import router as memory_router

api_router = APIRouter()

# ── Health ────────────────────────────────────────────────────────────────────
api_router.include_router(
    health_router,
    prefix="/health",
    tags=["Health"],
)

# ── Chat / AI ─────────────────────────────────────────────────────────────────
api_router.include_router(
    chat_router,
    prefix="/chat",
    tags=["Chat"],
)

# ── Tasks & Reminders ─────────────────────────────────────────────────────────
api_router.include_router(
    tasks_router,
    prefix="/tasks",
    tags=["Tasks"],
)

# ── Voice (Whisper + ElevenLabs) ──────────────────────────────────────────────
api_router.include_router(
    voice_router,
    prefix="/voice",
    tags=["Voice"],
)

# ── Telegram integration ──────────────────────────────────────────────────────
api_router.include_router(
    telegram_router,
    prefix="/telegram",
    tags=["Telegram"],
)

# ── Memory / knowledge base ───────────────────────────────────────────────────
api_router.include_router(
    memory_router,
    prefix="/memory",
    tags=["Memory"],
)
