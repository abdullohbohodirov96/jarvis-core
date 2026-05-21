"""
FastAPI middleware for JARVIS.

Middleware stack (applied in registration order — outermost to innermost):
  1. CORSMiddleware        – sets CORS headers
  2. RequestIDMiddleware   – injects X-Request-ID
  3. TimingMiddleware      – logs request duration
  4. RateLimitMiddleware   – per-IP sliding-window rate limiting

Call `setup_middleware(app)` once inside `create_app()`.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from backend.app.core.config import get_settings
from backend.app.core.logging_config import (
    bind_request_id,
    clear_request_context,
    get_logger,
)

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Request-ID middleware
# ---------------------------------------------------------------------------

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every request and echo it in the response.

    The ID is either taken from the incoming ``X-Request-ID`` header (so that
    upstream proxies / API gateways can propagate their own trace IDs) or a
    fresh UUID4 is generated.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())

        # Make the request_id available on the request state so that route
        # handlers can retrieve it without re-parsing the header.
        request.state.request_id = request_id

        # Bind into structlog context (no-op when structlog is absent).
        bind_request_id(request_id)

        try:
            response = await call_next(request)
        finally:
            clear_request_context()

        response.headers[REQUEST_ID_HEADER] = request_id
        return response


# ---------------------------------------------------------------------------
# Timing middleware
# ---------------------------------------------------------------------------

class TimingMiddleware(BaseHTTPMiddleware):
    """Log the wall-clock duration of every HTTP request.

    Adds an ``X-Process-Time`` response header (seconds, 4 d.p.) and emits
    a structured INFO log line with method, path, status code, and duration.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        response.headers["X-Process-Time"] = f"{duration:.4f}"

        log_kwargs: Dict[str, Any] = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
        }

        if hasattr(request.state, "request_id"):
            log_kwargs["request_id"] = request.state.request_id

        if response.status_code >= 500:
            logger.error("request_completed", **log_kwargs)
        elif response.status_code >= 400:
            logger.warning("request_completed", **log_kwargs)
        else:
            logger.info("request_completed", **log_kwargs)

        return response


# ---------------------------------------------------------------------------
# Rate-limit middleware
# ---------------------------------------------------------------------------

# Simple token-bucket / sliding-window counter that lives in process memory.
# For multi-process deployments the Redis backend is preferred.
_InMemoryStore: Dict[str, Tuple[float, int]] = defaultdict(lambda: (0.0, 0))


def _get_client_ip(request: Request) -> str:
    """Extract the real client IP, honoring common proxy headers."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    if request.client:
        return request.client.host
    return "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter.

    Tries to use Redis for distributed counting; falls back to an in-process
    dict when Redis is unavailable so the service stays functional.

    Configuration is read from ``settings.RATE_LIMIT_REQUESTS`` and
    ``settings.RATE_LIMIT_WINDOW_SECONDS``.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._max_requests: int = settings.RATE_LIMIT_REQUESTS
        self._window: int = settings.RATE_LIMIT_WINDOW_SECONDS
        self._redis: Optional[Any] = None  # lazily initialised

    async def _get_redis(self) -> Optional[Any]:
        """Return a lazily-created Redis client, or None on failure."""
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis  # type: ignore

            client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=1,
            )
            await client.ping()
            self._redis = client
            return self._redis
        except Exception:
            return None

    async def _check_redis(self, ip: str, now: float) -> Tuple[bool, int]:
        """Return (is_allowed, remaining) using Redis INCR + EXPIRE."""
        redis = await self._get_redis()
        if redis is None:
            return True, self._max_requests  # allow when Redis unavailable

        key = f"ratelimit:{ip}:{int(now // self._window)}"
        try:
            count = await redis.incr(key)
            if count == 1:
                await redis.expire(key, self._window)
            remaining = max(0, self._max_requests - count)
            return count <= self._max_requests, remaining
        except Exception:
            return True, self._max_requests

    def _check_memory(self, ip: str, now: float) -> Tuple[bool, int]:
        """Return (is_allowed, remaining) using the in-process counter."""
        window_start, count = _InMemoryStore[ip]
        if now - window_start > self._window:
            _InMemoryStore[ip] = (now, 1)
            return True, self._max_requests - 1
        new_count = count + 1
        _InMemoryStore[ip] = (window_start, new_count)
        remaining = max(0, self._max_requests - new_count)
        return new_count <= self._max_requests, remaining

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Health checks are exempt from rate limiting.
        if request.url.path.startswith("/health"):
            return await call_next(request)

        ip = _get_client_ip(request)
        now = time.time()

        redis = await self._get_redis()
        if redis is not None:
            allowed, remaining = await self._check_redis(ip, now)
        else:
            allowed, remaining = self._check_memory(ip, now)

        if not allowed:
            logger.warning("rate_limit_exceeded", client_ip=ip, path=request.url.path)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Please slow down.",
                        "details": {
                            "limit": self._max_requests,
                            "window_seconds": self._window,
                        },
                    }
                },
                headers={
                    "Retry-After": str(self._window),
                    "X-RateLimit-Limit": str(self._max_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------

def setup_middleware(app: FastAPI) -> None:
    """Register all middleware on *app*.

    Starlette/FastAPI middleware is applied in *reverse* registration order,
    so the first-registered middleware is the outermost (executed first on
    ingress, last on egress).

    Registration order here:
        1. CORSMiddleware  (outermost – handles preflight before any logic)
        2. RequestIDMiddleware
        3. TimingMiddleware
        4. RateLimitMiddleware  (innermost – has access to Request-ID)
    """

    # 1. CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER, "X-Process-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )

    # 2. Request ID
    app.add_middleware(RequestIDMiddleware)

    # 3. Timing
    app.add_middleware(TimingMiddleware)

    # 4. Rate limit
    app.add_middleware(RateLimitMiddleware)
