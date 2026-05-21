"""
Alembic migration environment.

Supports both offline (SQL script generation) and online (async live database)
migration modes.  The application's DATABASE_URL from settings is used so that
environment-specific connection strings are resolved automatically.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from logging.config import fileConfig
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that backend.* imports resolve
# regardless of how alembic is invoked.
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# ---------------------------------------------------------------------------
# Application imports
# ---------------------------------------------------------------------------
# Import settings first (no side-effects)
from backend.app.core.config import get_settings

# Import ALL model modules so their tables are registered on Base.metadata
# before autogenerate compares against the live schema.
import backend.app.models  # noqa: F401 – registers User, Conversation, etc.

from backend.app.database.connection import Base

# ---------------------------------------------------------------------------
# Alembic Config object
# ---------------------------------------------------------------------------
config = context.config

# Inject the runtime DATABASE_URL, converting asyncpg → psycopg2 for the
# synchronous offline path.  The online path uses the async URL directly.
settings = get_settings()
_async_url = settings.DATABASE_URL  # e.g. postgresql+asyncpg://...

# Provide a sync-compatible URL for offline mode (SQL script generation).
# asyncpg cannot be used with the synchronous offline rendering path.
_sync_url = _async_url.replace(
    "postgresql+asyncpg://", "postgresql+psycopg2://"
).replace(
    "postgresql+asyncpg+srv://", "postgresql+psycopg2://"
)

# Override the ini-file URL with the one from settings.
config.set_main_option("sqlalchemy.url", _async_url)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

# ---------------------------------------------------------------------------
# Target metadata for autogenerate
# ---------------------------------------------------------------------------
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Helper: include object filter
# ---------------------------------------------------------------------------

def include_object(
    obj: object,
    name: str,
    type_: str,
    reflected: bool,
    compare_to: object,
) -> bool:
    """
    Fine-grained control over which database objects are included in
    autogenerate comparisons.

    Currently we:
    - Skip tables prefixed with "celery_" (managed by Celery Beat).
    - Skip the PostGIS geometry_columns / spatial_ref_sys views.
    """
    if type_ == "table" and name.startswith(("celery_", "spatial_ref_sys", "geometry_columns")):
        return False
    return True


# ---------------------------------------------------------------------------
# Offline migrations (generate SQL script, no DB connection needed)
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine; by skipping
    the Engine creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the script output.
    """
    url = _sync_url  # Use sync URL for offline SQL rendering
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migrations (run against a live database)
# ---------------------------------------------------------------------------

def do_run_migrations(connection: Connection) -> None:
    """Configure the migration context and run all pending migrations."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=False,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine from config and run migrations inside it."""
    # Build engine config from alembic.ini [alembic] section, overriding the
    # URL with the async one from settings.
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _async_url

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # No pooling for migration runs
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migration mode."""
    asyncio.run(run_async_migrations())


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    logger.info("Running migrations in offline mode")
    run_migrations_offline()
else:
    logger.info("Running migrations in online mode")
    run_migrations_online()
