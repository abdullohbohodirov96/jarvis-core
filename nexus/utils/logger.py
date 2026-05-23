import sys
import os
from loguru import logger as _loguru_logger
from core.config import settings


def _build_format(record: dict) -> str:
    level_colors = {
        "TRACE": "<cyan>",
        "DEBUG": "<blue>",
        "INFO": "<green>",
        "SUCCESS": "<green>",
        "WARNING": "<yellow>",
        "ERROR": "<red>",
        "CRITICAL": "<red>",
    }
    level_name: str = record["level"].name
    color = level_colors.get(level_name, "<white>")
    fmt = (
        "<dim>{time:YYYY-MM-DD HH:mm:ss.SSS}</dim> | "
        f"{color}{{level: <8}}</> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>\n"
    )
    if record["exception"]:
        fmt += "{exception}\n"
    return fmt


def configure_logging() -> None:
    _loguru_logger.remove()

    _loguru_logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL.upper(),
        format=_build_format,
        colorize=True,
        backtrace=True,
        diagnose=not settings.is_production,
        enqueue=False,
    )

    if settings.is_production:
        # Use /tmp so it always exists on any host (Render, Railway, Fly, etc.)
        log_path = "/tmp/nexus.log"
        try:
            _loguru_logger.add(
                log_path,
                level="WARNING",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation="10 MB",
                retention="14 days",
                compression="gz",
                backtrace=False,
                diagnose=False,
                enqueue=True,
            )
        except Exception as exc:
            # Never crash the app over a logging misconfiguration
            _loguru_logger.warning("Could not add file log sink ({}): {}", log_path, exc)


configure_logging()

logger = _loguru_logger
