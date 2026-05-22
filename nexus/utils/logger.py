import sys
from loguru import logger as _loguru_logger
from core.config import settings


def _build_format(record: dict) -> str:
    """
    Build a colored, human-readable log format string.

    Format:
        2024-01-01 12:00:00.000 | INFO     | module:function:42 - message
    """
    level_colors = {
        "TRACE": "<cyan>",
        "DEBUG": "<blue>",
        "INFO": "<green>",
        "SUCCESS": "<bold><green>",
        "WARNING": "<yellow>",
        "ERROR": "<red>",
        "CRITICAL": "<bold><red>",
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
    """
    Remove default loguru sink and install a clean colored console sink
    whose minimum level is controlled by settings.LOG_LEVEL.
    """
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
        _loguru_logger.add(
            "logs/nexus.log",
            level="WARNING",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="14 days",
            compression="gz",
            backtrace=False,
            diagnose=False,
            enqueue=True,
        )


configure_logging()

logger = _loguru_logger
