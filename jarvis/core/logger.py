"""
JARVIS structured logging.

Uses loguru with:
  - Colored stdout sink (always active, level driven by settings).
  - File sink at /tmp/jarvis.log active in production mode only
    (10 MB rotation, 7-day retention).

Usage
-----
    from core.logger import logger, get_logger

    log = get_logger(__name__)
    log.info("hello {name}", name="world")
"""

from __future__ import annotations

import sys
from typing import Union

from loguru import logger as _loguru_logger

from config.settings import settings

# Public re-export so callers can do `from core.logger import logger`
logger = _loguru_logger

_LOG_FORMAT_STDOUT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[module]}</cyan> | "
    "<level>{message}</level>"
)

_LOG_FORMAT_FILE = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{extra[module]} | "
    "{message}"
)


def setup_logging() -> None:
    """Configure all loguru sinks.  Safe to call multiple times."""
    _loguru_logger.remove()  # Remove the default sink.

    # -------------------------------------------------------- stdout sink
    _loguru_logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL.upper(),
        format=_LOG_FORMAT_STDOUT,
        colorize=True,
        diagnose=settings.is_development,
        backtrace=settings.is_development,
        enqueue=False,
    )

    # ------------------------------------------------- file sink (prod only)
    if settings.is_production:
        _loguru_logger.add(
            "/tmp/jarvis.log",
            level=settings.LOG_LEVEL.upper(),
            format=_LOG_FORMAT_FILE,
            rotation="10 MB",
            retention="7 days",
            compression="gz",
            encoding="utf-8",
            diagnose=False,
            backtrace=False,
            enqueue=True,  # thread-safe async write
        )


def get_logger(name: str) -> "loguru.Logger":  # noqa: F821
    """Return a loguru logger pre-bound with a `module` context key.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    """
    return _loguru_logger.bind(module=name)


# Run setup at import time so that every module that imports `logger`
# immediately gets a properly configured sink.
setup_logging()
