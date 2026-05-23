import uuid
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from core.config import settings
from utils.logger import logger


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


def _build_engine() -> AsyncEngine:
    """
    Build the async SQLAlchemy engine.

    Connection-pool settings are tuned for a lightweight production service:
    - pool_size=5     : keep five persistent connections open
    - max_overflow=10 : allow up to ten extra connections under load burst
    - pool_pre_ping   : discard stale connections before lending them out
    - pool_recycle    : recycle connections older than 30 min to avoid
                        "gone away" errors from intermediate proxies
    
    Supabase (PgBouncer) transaction pooling fix:
    - prepared_statement_cache_size=0: Disables SQLAlchemy prepared statement caching.
    - connect_args={"statement_cache_size": 0}: Disables asyncpg native prepared statement caching.
    - connect_args={"prepared_statement_name_func": ...}: Generates unique names for anonymous 
      prepared statements so PgBouncer transaction mode never suffers from statement name collision.
    """
    return create_async_engine(
        settings.DATABASE_URL,
        echo=settings.is_development,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=1800,
        prepared_statement_cache_size=0,
        connect_args={
            "statement_cache_size": 0,
            "prepared_statement_name_func": lambda: f"__asyncpg_{uuid.uuid4().hex}__"
        }
    )


engine: AsyncEngine = _build_engine()

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a per-request async database session.

    Usage::

        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            ...

    The session is committed automatically on success and rolled back on any
    unhandled exception, then closed in the ``finally`` block.
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


async def init_db() -> None:
    """
    Create all tables that are registered on ``Base.metadata``.

    Called once at application startup.  Safe to call on every restart —
    SQLAlchemy uses ``CREATE TABLE IF NOT EXISTS`` semantics via
    ```checkfirst=True``` (the default for ```create_all```).
    """
    from database import models  # noqa: F401 — side-effect: registers models

    logger.info("Initialising database tables…")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.success("Database tables ready.")


async def check_db_connection() -> bool:
    """
    Probe the database with a lightweight ``SELECT 1``.

    Returns ``True`` if the database is reachable, ``False`` otherwise.
    Used by the ``/health`` endpoint.
    """
    from sqlalchemy import text

    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error(f"Database health-check failed: {exc}")
        return False
