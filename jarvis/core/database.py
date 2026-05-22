"""
JARVIS async database layer.

SQLAlchemy 2.0 + aiosqlite backend.

Public API
----------
    Base          — DeclarativeBase for all models
    AsyncSessionLocal — async sessionmaker
    get_db()      — FastAPI dependency yielding an AsyncSession
    init_database()   — create all tables (call at startup)
    close_database()  — dispose engine (call at shutdown)
    check_database()  — health probe returning True/False
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from config.settings import settings
from core.logger import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Engine & session factory                                                      #
# --------------------------------------------------------------------------- #

_engine: AsyncEngine | None = None


def _get_engine() -> AsyncEngine:
    """Return (or lazily create) the shared async engine."""
    global _engine
    if _engine is None:
        db_url = settings.database_url_async
        connect_args: dict = {}

        # aiosqlite-specific pragmas for better concurrency and durability.
        if "sqlite" in db_url:
            connect_args = {
                "timeout": 30,
                "check_same_thread": False,
            }

        _engine = create_async_engine(
            db_url,
            echo=settings.DEBUG,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
        log.debug("Async database engine created: {url}", url=db_url)
    return _engine


# Declarative base shared by all model files.
class Base(DeclarativeBase):
    pass


# Session factory — import this where you need manual session management.
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=_get_engine(),
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# --------------------------------------------------------------------------- #
# FastAPI dependency                                                             #
# --------------------------------------------------------------------------- #


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session and ensure it is closed afterwards.

    Intended for use as a FastAPI dependency::

        @router.get("/example")
        async def endpoint(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# --------------------------------------------------------------------------- #
# Lifecycle helpers                                                              #
# --------------------------------------------------------------------------- #


async def init_database() -> None:
    """Create all tables registered on ``Base``.

    Called once during application startup.  Importing models before calling
    this function ensures their tables are registered on ``Base.metadata``.
    """
    # Import all models so their tables are registered with Base.metadata
    # before we call create_all.  Each model file is imported lazily here
    # to avoid circular imports.
    try:
        from memory import models as _models  # noqa: F401  side-effect import
    except ImportError:
        log.warning("memory.models not importable yet — skipping model pre-load")

    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    log.info("Database initialised — all tables created / verified.")


async def close_database() -> None:
    """Dispose the async engine, releasing all pooled connections.

    Called once during application shutdown.
    """
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        log.info("Database engine disposed.")


async def check_database() -> bool:
    """Return ``True`` if the database is reachable, ``False`` otherwise."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("Database health check failed: {exc}", exc=exc)
        return False
