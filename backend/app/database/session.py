"""
Transaction management utilities and a generic base repository.

The BaseRepository follows the Unit-of-Work pattern: callers own the session
and pass it in; the repository never commits or rolls back on its own.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Generic, TypeVar
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, NoResultFound, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Generic type variable bound to any SQLAlchemy mapped class
ModelT = TypeVar("ModelT")


# ---------------------------------------------------------------------------
# Transaction context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def managed_transaction(
    session: AsyncSession,
) -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager that wraps work in an explicit savepoint (nested
    transaction).  If the outer session already has an active transaction this
    creates a SAVEPOINT; otherwise it starts a full BEGIN/COMMIT block.

    Usage::

        async with managed_transaction(session) as txn:
            txn.add(some_object)
            # raises → automatic rollback to savepoint / full rollback
    """
    if session.in_transaction():
        async with session.begin_nested() as savepoint:
            try:
                yield session
                await savepoint.commit()
            except SQLAlchemyError:
                await savepoint.rollback()
                raise
    else:
        async with session.begin():
            try:
                yield session
            except SQLAlchemyError:
                await session.rollback()
                raise


# ---------------------------------------------------------------------------
# get_or_create helper
# ---------------------------------------------------------------------------

async def get_or_create(
    session: AsyncSession,
    model: type[ModelT],
    defaults: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[ModelT, bool]:
    """
    Look up a row matching *kwargs*; create it (with *defaults* merged in) if
    absent.

    Returns a ``(instance, created)`` 2-tuple where *created* is ``True`` when
    the row was newly inserted.

    The insert is guarded with a savepoint so that an IntegrityError from a
    concurrent insert (race condition) is handled gracefully: the function
    retries the SELECT after the conflict.
    """
    stmt = select(model).filter_by(**kwargs)
    result = await session.execute(stmt)
    instance = result.scalars().first()
    if instance is not None:
        return instance, False

    # Merge lookup kwargs with defaults for the new row
    create_kwargs = {**kwargs, **(defaults or {})}
    instance = model(**create_kwargs)  # type: ignore[call-arg]

    try:
        async with managed_transaction(session):
            session.add(instance)
    except IntegrityError:
        # Concurrent insert won the race; fall back to SELECT
        await session.rollback()
        result = await session.execute(stmt)
        instance = result.scalars().one()
        return instance, False

    return instance, True


# ---------------------------------------------------------------------------
# Base repository
# ---------------------------------------------------------------------------

class BaseRepository(Generic[ModelT]):
    """
    Generic CRUD repository.

    Subclass and set the ``model`` class attribute::

        class UserRepository(BaseRepository[User]):
            model = User

    All public methods receive the ``session`` as their first argument so that
    multiple repositories can participate in the same unit-of-work.
    """

    model: type[ModelT]

    # ── Create ────────────────────────────────────────────────────────────────

    async def create(
        self,
        session: AsyncSession,
        **data: Any,
    ) -> ModelT:
        """
        Insert a new row and return the persisted instance.

        The caller is responsible for committing the session.
        """
        instance = self.model(**data)  # type: ignore[call-arg]
        session.add(instance)
        await session.flush()          # Populate server-generated defaults (e.g. id)
        await session.refresh(instance)
        logger.debug("Created %s id=%s", type(instance).__name__, getattr(instance, "id", None))
        return instance

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get(
        self,
        session: AsyncSession,
        id: UUID | str | int,
    ) -> ModelT | None:
        """Return a single row by primary key, or ``None`` if not found."""
        return await session.get(self.model, id)

    async def get_or_raise(
        self,
        session: AsyncSession,
        id: UUID | str | int,
    ) -> ModelT:
        """Return a single row by primary key, raising ``NoResultFound`` if absent."""
        instance = await self.get(session, id)
        if instance is None:
            raise NoResultFound(f"{self.model.__name__} with id={id!r} not found.")
        return instance

    async def get_all(
        self,
        session: AsyncSession,
        skip: int = 0,
        limit: int = 20,
        **filters: Any,
    ) -> list[ModelT]:
        """
        Return a paginated list of rows, optionally filtered by equality on
        keyword arguments.

        Soft-deleted rows (``is_deleted=True``) are automatically excluded when
        the model defines that column.
        """
        stmt = select(self.model)

        # Exclude soft-deleted rows when the column exists
        if hasattr(self.model, "is_deleted"):
            stmt = stmt.where(self.model.is_deleted.is_(False))  # type: ignore[attr-defined]

        # Apply equality filters for any provided kwargs
        for attr, value in filters.items():
            column = getattr(self.model, attr, None)
            if column is None:
                raise AttributeError(
                    f"{self.model.__name__} has no attribute '{attr}'"
                )
            stmt = stmt.where(column == value)

        stmt = stmt.offset(skip).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    # ── Update ────────────────────────────────────────────────────────────────

    async def update(
        self,
        session: AsyncSession,
        id: UUID | str | int,
        **data: Any,
    ) -> ModelT | None:
        """
        Apply *data* fields to the row identified by *id*.

        Returns the updated instance, or ``None`` when the row does not exist.
        """
        instance = await self.get(session, id)
        if instance is None:
            return None
        for attr, value in data.items():
            setattr(instance, attr, value)
        session.add(instance)
        await session.flush()
        await session.refresh(instance)
        logger.debug(
            "Updated %s id=%s fields=%s",
            type(instance).__name__,
            id,
            list(data.keys()),
        )
        return instance

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete(
        self,
        session: AsyncSession,
        id: UUID | str | int,
    ) -> bool:
        """
        Soft-delete the row when the model has ``is_deleted``; otherwise hard-
        delete.

        Returns ``True`` on success, ``False`` when the row was not found.
        """
        instance = await self.get(session, id)
        if instance is None:
            return False

        if hasattr(instance, "is_deleted"):
            instance.is_deleted = True  # type: ignore[attr-defined]
            session.add(instance)
            await session.flush()
            logger.debug(
                "Soft-deleted %s id=%s", type(instance).__name__, id
            )
        else:
            await session.delete(instance)
            await session.flush()
            logger.debug(
                "Hard-deleted %s id=%s", type(instance).__name__, id
            )
        return True

    # ── Count ─────────────────────────────────────────────────────────────────

    async def count(
        self,
        session: AsyncSession,
        **filters: Any,
    ) -> int:
        """Return the number of rows matching *filters* (excludes soft-deleted)."""
        stmt = select(func.count()).select_from(self.model)  # type: ignore[arg-type]

        if hasattr(self.model, "is_deleted"):
            stmt = stmt.where(self.model.is_deleted.is_(False))  # type: ignore[attr-defined]

        for attr, value in filters.items():
            column = getattr(self.model, attr, None)
            if column is None:
                raise AttributeError(
                    f"{self.model.__name__} has no attribute '{attr}'"
                )
            stmt = stmt.where(column == value)

        result = await session.execute(stmt)
        return result.scalar_one()
