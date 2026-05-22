"""
JARVIS FastAPI application factory.

Entry point:  uvicorn backend.app.main:app --reload
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import get_settings
from backend.app.core.exceptions import register_exception_handlers
from backend.app.core.logging_config import get_logger, setup_logging
from backend.app.core.middleware import setup_middleware

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Application start time (for uptime reporting)
# ---------------------------------------------------------------------------
_APP_START_TIME = time.time()

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Manages a pool of active WebSocket connections for real-time chat.

    Each connection is identified by a ``client_id`` string.  Messages can be
    sent to a specific client or broadcast to all connected clients.
    """

    def __init__(self) -> None:
        self._connections: Dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, client_id: str, websocket: WebSocket) -> None:
        """Accept a new WebSocket handshake and register the connection."""
        await websocket.accept()
        async with self._lock:
            self._connections[client_id] = websocket
        logger.info("ws_client_connected", client_id=client_id)

    async def disconnect(self, client_id: str) -> None:
        """Remove a client from the pool (called when connection closes)."""
        async with self._lock:
            self._connections.pop(client_id, None)
        logger.info("ws_client_disconnected", client_id=client_id)

    async def send_text(self, client_id: str, message: str) -> bool:
        """Send a text frame to a specific client.

        Returns:
            ``True`` if the message was sent, ``False`` if the client is gone.
        """
        ws = self._connections.get(client_id)
        if ws is None:
            return False
        try:
            await ws.send_text(message)
            return True
        except Exception as exc:
            logger.warning("ws_send_error", client_id=client_id, error=str(exc))
            await self.disconnect(client_id)
            return False

    async def send_json(self, client_id: str, data: Dict[str, Any]) -> bool:
        """Send a JSON frame to a specific client."""
        ws = self._connections.get(client_id)
        if ws is None:
            return False
        try:
            await ws.send_json(data)
            return True
        except Exception as exc:
            logger.warning("ws_send_json_error", client_id=client_id, error=str(exc))
            await self.disconnect(client_id)
            return False

    async def broadcast(self, message: str) -> None:
        """Send a text frame to every connected client."""
        client_ids = list(self._connections.keys())
        await asyncio.gather(
            *(self.send_text(cid, message) for cid in client_ids),
            return_exceptions=True,
        )

    @property
    def active_clients(self) -> Set[str]:
        return set(self._connections.keys())

    def __len__(self) -> int:
        return len(self._connections)


# Module-level singleton so routes can import it directly.
ws_manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Lifespan context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown events.

    Startup sequence:
      1. Initialise logging
      2. Connect to the database (test pool)
      3. Ping Redis
      4. Register Celery (import workers so tasks are discoverable)

    Shutdown sequence (reverse):
      1. Close DB pool
      2. Close Redis client
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    setup_logging(
        log_level=settings.LOG_LEVEL,
        is_production=settings.is_production,
    )
    logger.info(
        "jarvis_starting",
        app=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )

    # Database
    db_engine = None
    try:
        from sqlalchemy.ext.asyncio import create_async_engine

        db_engine = create_async_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            echo=settings.DEBUG,
        )
        async with db_engine.connect() as conn:
            from sqlalchemy import text

            await conn.execute(text("SELECT 1"))
        logger.info("database_connected")
        app.state.db_engine = db_engine
    except Exception as exc:
        logger.warning("database_connection_failed", error=str(exc))
        app.state.db_engine = None

    # Redis
    redis_client = None
    try:
        import redis.asyncio as aioredis  # type: ignore

        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
        )
        await redis_client.ping()
        logger.info("redis_connected")
        app.state.redis = redis_client
    except Exception as exc:
        logger.warning("redis_connection_failed", error=str(exc))
        app.state.redis = None

    # Celery – just import to ensure tasks are registered
    try:
        import backend.app.workers  # noqa: F401  # type: ignore

        logger.info("celery_workers_imported")
    except ImportError:
        logger.warning("celery_workers_not_found")

    logger.info("jarvis_ready")

    # ── Hand control back to FastAPI ─────────────────────────────────────────
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("jarvis_shutting_down")

    if redis_client is not None:
        try:
            await redis_client.aclose()
            logger.info("redis_disconnected")
        except Exception:
            pass

    if db_engine is not None:
        try:
            await db_engine.dispose()
            logger.info("database_disconnected")
        except Exception:
            pass

    logger.info("jarvis_stopped")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Returns:
        A fully configured ``FastAPI`` instance ready to be served.
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "JARVIS — an intelligent personal assistant platform with "
            "AI reasoning, task management, voice processing, and Telegram integration."
        ),
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # Middleware (order matters — see middleware.py for details)
    setup_middleware(app)

    # Exception handlers
    register_exception_handlers(app)

    # API routers
    from backend.app.api.router import api_router

    app.include_router(api_router, prefix="/api/v1")

    # Static files (frontend build artefacts)
    import os
    static_dir = "static"
    if not os.path.isdir(static_dir) and os.path.isdir("backend/static"):
        static_dir = "backend/static"
    try:
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    except RuntimeError:
        # static directory doesn't exist yet — fine in development
        pass

    # ── Root health check ─────────────────────────────────────────────────────

    @app.get("/", tags=["Root"], summary="Root health check")
    async def root():
        import os
        from fastapi.responses import FileResponse
        # Check both potential paths for static files
        paths = ["backend/static/index.html", "static/index.html", "app/static/index.html"]
        for path in paths:
            if os.path.exists(path):
                return FileResponse(path)
        return JSONResponse(
            content={
                "status": "ok",
                "app": settings.APP_NAME,
                "version": settings.APP_VERSION,
            }
        )

    # ── WebSocket chat endpoint ───────────────────────────────────────────────

    @app.websocket("/ws/chat/{client_id}")
    async def websocket_chat(websocket: WebSocket, client_id: str) -> None:
        """Real-time bidirectional chat over WebSocket.

        Protocol:
          - Client sends: ``{"message": "...", "conversation_id": "..."}``
          - Server sends: ``{"type": "token", "content": "..."}`` (streaming)
          - Server sends: ``{"type": "done", "conversation_id": "..."}``
          - Server sends: ``{"type": "error", "message": "..."}``
        """
        await ws_manager.connect(client_id, websocket)
        try:
            while True:
                data = await websocket.receive_json()
                message: str = data.get("message", "").strip()
                conversation_id: str = data.get("conversation_id", "default")

                if not message:
                    await ws_manager.send_json(
                        client_id,
                        {"type": "error", "message": "Empty message."},
                    )
                    continue

                logger.info(
                    "ws_message_received",
                    client_id=client_id,
                    conversation_id=conversation_id,
                    message_len=len(message),
                )

                # Delegate to the AI service if available; echo otherwise.
                try:
                    from backend.app.ai.chat_service import stream_chat_response  # type: ignore

                    async for token in stream_chat_response(message, conversation_id):
                        await ws_manager.send_json(
                            client_id, {"type": "token", "content": token}
                        )
                except ImportError:
                    # AI service not yet implemented — simple echo
                    await ws_manager.send_json(
                        client_id,
                        {"type": "token", "content": f"Echo: {message}"},
                    )

                await ws_manager.send_json(
                    client_id,
                    {"type": "done", "conversation_id": conversation_id},
                )

        except WebSocketDisconnect:
            await ws_manager.disconnect(client_id)
        except Exception as exc:
            logger.error("ws_unhandled_error", client_id=client_id, error=str(exc))
            await ws_manager.disconnect(client_id)

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (used by uvicorn / gunicorn)
# ---------------------------------------------------------------------------
app = create_app()
