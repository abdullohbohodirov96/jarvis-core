"""
Custom exception hierarchy for JARVIS.

All domain exceptions inherit from JarvisBaseException so callers can catch
the entire family with a single `except JarvisBaseException`.

FastAPI exception handlers are registered via `register_exception_handlers`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------


class JarvisBaseException(Exception):
    """Root exception for all JARVIS-specific errors.

    Attributes:
        message: Human-readable error description.
        code:    Machine-readable error code string (e.g. "AI_TIMEOUT").
        details: Optional dict of extra context for debugging.
        status_code: HTTP status code to return (default 500).
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(
        self,
        message: str = "An unexpected error occurred.",
        code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details: Dict[str, Any] = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(code={self.code!r}, message={self.message!r})"


# ---------------------------------------------------------------------------
# Domain-specific exceptions
# ---------------------------------------------------------------------------


class AIException(JarvisBaseException):
    """Raised when an AI / LLM operation fails."""

    status_code = status.HTTP_502_BAD_GATEWAY

    def __init__(
        self,
        message: str = "AI service error.",
        code: str = "AI_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class DatabaseException(JarvisBaseException):
    """Raised on database read/write failures."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    def __init__(
        self,
        message: str = "Database operation failed.",
        code: str = "DB_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class TelegramException(JarvisBaseException):
    """Raised when a Telegram API call fails."""

    status_code = status.HTTP_502_BAD_GATEWAY

    def __init__(
        self,
        message: str = "Telegram service error.",
        code: str = "TELEGRAM_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class VoiceException(JarvisBaseException):
    """Raised when voice transcription or synthesis fails."""

    status_code = status.HTTP_502_BAD_GATEWAY

    def __init__(
        self,
        message: str = "Voice pipeline error.",
        code: str = "VOICE_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class WorkerException(JarvisBaseException):
    """Raised when a Celery background task fails."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(
        self,
        message: str = "Background worker error.",
        code: str = "WORKER_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class AuthException(JarvisBaseException):
    """Raised on authentication / authorisation failures."""

    status_code = status.HTTP_401_UNAUTHORIZED

    def __init__(
        self,
        message: str = "Authentication failed.",
        code: str = "AUTH_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class ValidationException(JarvisBaseException):
    """Raised when request or domain data fails validation."""

    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

    def __init__(
        self,
        message: str = "Validation error.",
        code: str = "VALIDATION_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class RateLimitException(JarvisBaseException):
    """Raised when a client exceeds the request rate limit."""

    status_code = status.HTTP_429_TOO_MANY_REQUESTS

    def __init__(
        self,
        message: str = "Rate limit exceeded. Please slow down.",
        code: str = "RATE_LIMIT_EXCEEDED",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class NotFoundException(JarvisBaseException):
    """Raised when a requested resource does not exist."""

    status_code = status.HTTP_404_NOT_FOUND

    def __init__(
        self,
        message: str = "Resource not found.",
        code: str = "NOT_FOUND",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class ConflictException(JarvisBaseException):
    """Raised when an operation conflicts with existing state."""

    status_code = status.HTTP_409_CONFLICT

    def __init__(
        self,
        message: str = "Resource conflict.",
        code: str = "CONFLICT",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


# ---------------------------------------------------------------------------
# FastAPI exception handlers
# ---------------------------------------------------------------------------


def _make_error_response(
    status_code: int,
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
            }
        },
    )


async def _jarvis_exception_handler(
    request: Request, exc: JarvisBaseException
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    formatted = [
        {
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]
    return _make_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        code="VALIDATION_ERROR",
        message="Request validation failed.",
        details={"errors": formatted},
    )


async def _generic_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    return _make_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="INTERNAL_ERROR",
        message="An unexpected internal server error occurred.",
        details={"type": type(exc).__name__},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all custom exception handlers to a FastAPI application instance.

    Call this inside ``create_app()`` after the app is instantiated.

    Args:
        app: The FastAPI application instance.
    """
    app.add_exception_handler(JarvisBaseException, _jarvis_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _generic_exception_handler)  # type: ignore[arg-type]
