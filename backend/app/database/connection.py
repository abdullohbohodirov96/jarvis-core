"""
Async SQLAlchemy engine, session factory, and FastAPI dependency.

All database interaction should go through get_db(); direct engine access is
reserved for startup/shutdown lifecycle hooks and health checks.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_engine: AsyncEngine | None = None


def _create_engine() -> AsyncEngine:
    """Create and return the async engine singleton."""
    return create_async_engine(
        settings.DATABASE_URL,
        # Connection pool configuration
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,          # Verify connections before checkout
        pool_recycle=3600,           # Recycle connections after 1 hour
        pool_timeout=30,             # Wait up to 30 s for a free connection
        echo=settings.is_debug,      # Log SQL in debug/dev mode only
        echo_pool=settings.is_debug,
        future=True,
        # asyncpg-specific options
        connect_args={
            "server_settings": {
                "application_name": settings.APP_NAME,
                "jit": "off",        # Avoid JIT overhead for short queries
            }
        },
    )


def get_engine() -> AsyncEngine:
    """Return the process-level engine, creating it on first call."""
    global _engine
    if _engine is None:
        _engine = _create_engine()
    return _engine


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=get_engine(),
    class_=AsyncSession,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,   # Avoid implicit lazy-loads after commit
)


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    # Subclasses can override __table_args__ to add schema-level options.
    __abstract__ = True

    def __repr__(self) -> str:  # pragma: no cover
        cls = type(self).__name__
        pk = getattr(self, "id", None)
        return f"<{cls} id={pk!r}>"


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a transactional AsyncSession.

    Usage::

        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            ...

    The session is automatically rolled back on unhandled exceptions and
    closed when the request finishes.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except SQLAlchemyError:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """
    Create all tables defined in Base.metadata.

    Should be called once during application startup (e.g. in a FastAPI
    lifespan handler).  In production, prefer Alembic migrations instead.
    """
    # Import every model module so that their tables register on Base.metadata
    # before we call create_all.
    import app.models  # noqa: F401 – side-effect import

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialised.")


async def close_db() -> None:
    """Dispose the engine connection pool.

    Call during application shutdown to release all held connections cleanly.
    """
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        logger.info("Database engine disposed.")


async def check_db_connection() -> bool:
    """
    Lightweight health check: executes ``SELECT 1`` against the database.

    Returns ``True`` on success, ``False`` if the database is unreachable.
    """
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except (OperationalError, SQLAlchemyError) as exc:
        logger.warning("Database health check failed: %s", exc)
        return False
