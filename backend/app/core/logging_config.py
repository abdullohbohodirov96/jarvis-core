"""
Structured logging configuration for JARVIS.

Attempts to use structlog for rich structured/JSON logging.
Falls back gracefully to stdlib logging if structlog is not installed.

Usage:
    from app.core.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Something happened", user_id=42, action="login")
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Detect optional structlog dependency
# ---------------------------------------------------------------------------
try:
    import structlog  # type: ignore

    _STRUCTLOG_AVAILABLE = True
except ImportError:
    _STRUCTLOG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_log_level(level_name: str) -> int:
    """Convert a level name string to a logging integer constant."""
    return getattr(logging, level_name.upper(), logging.INFO)


def _build_file_handler(log_dir: str = "logs") -> logging.Handler:
    """Create a rotating file handler that writes to *log_dir*/jarvis.log."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        filename=f"{log_dir}/jarvis.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    return handler


# ---------------------------------------------------------------------------
# structlog-based setup
# ---------------------------------------------------------------------------

def _setup_structlog(log_level: int, is_production: bool) -> None:
    """Configure structlog with JSON output (prod) or pretty console (dev)."""

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if is_production:
        # JSON renderer for log aggregation tools (Datadog, Loki, etc.)
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        # Human-friendly coloured output for local development
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    # Root stdlib handler (console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Rotating file handler
    file_handler = _build_file_handler()
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    # Avoid duplicate handlers on repeated calls (e.g. during testing)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# stdlib-only setup (fallback)
# ---------------------------------------------------------------------------

def _setup_stdlib(log_level: int, is_production: bool) -> None:
    """Configure stdlib logging when structlog is not available."""

    if is_production:
        fmt = (
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"name": "%(name)s", "message": "%(message)s"}'
        )
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    datefmt = "%Y-%m-%dT%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = _build_file_handler()
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logging(log_level: str = "INFO", is_production: bool = False) -> None:
    """Initialise the logging system.

    Call this once at application startup (e.g. inside the FastAPI lifespan).

    Args:
        log_level: String level name (DEBUG / INFO / WARNING / ERROR / CRITICAL).
        is_production: When True, emit JSON-formatted log lines.
    """
    level = _get_log_level(log_level)

    if _STRUCTLOG_AVAILABLE:
        _setup_structlog(level, is_production)
    else:
        _setup_stdlib(level, is_production)


class _StdlibLoggerWrapper:
    """Wraps stdlib Logger to accept structlog-style keyword arguments."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, event: str, **kwargs: Any) -> None:
        extra = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
        msg = f"{event} {extra}" if extra else event
        self._logger.log(level, msg)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, event, **kwargs)


def get_logger(name: str) -> Any:
    """Return a logger bound to *name*.

    Returns a structlog BoundLogger when available, otherwise a standard
    logging.Logger.  Both expose the same .debug / .info / .warning / .error
    / .critical interface.

    Args:
        name: Typically ``__name__`` of the calling module.
    """
    if _STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return _StdlibLoggerWrapper(logging.getLogger(name))


def bind_request_id(request_id: str) -> None:
    """Bind a request_id into the current structlog context.

    Has no effect when structlog is not installed (request_id will still be
    passed as a keyword arg to individual log calls if needed).

    Args:
        request_id: UUID or other unique identifier for the current request.
    """
    if _STRUCTLOG_AVAILABLE:
        structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request_context() -> None:
    """Clear all structlog context vars (call at end of each request)."""
    if _STRUCTLOG_AVAILABLE:
        structlog.contextvars.clear_contextvars()
