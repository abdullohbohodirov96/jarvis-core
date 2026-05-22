"""
JARVIS data-access layer.

Provides async data stores for:
  TaskStore           — CRUD + query for Task records
  ConversationStore   — chat-history management
  MemoryStore         — long-term memory management
  TelegramMessageStore — Telegram message persistence

All methods are async and use SQLAlchemy 2.0-style (select / update / delete
with explicit where clauses).  Every store class receives an
``async_sessionmaker`` at construction; callers should pass
``core.database.AsyncSessionLocal``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import delete, distinct, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.logger import get_logger
from memory.models import (
    ConversationTurn,
    Memory,
    Task,
    TelegramMessage,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# TaskStore
# ---------------------------------------------------------------------------


class TaskStore:
    """Async CRUD for :class:`~memory.models.Task`."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    # ------------------------------------------------------------------ Create

    async def create(
        self,
        title: str,
        description: Optional[str] = None,
        priority: str = "medium",
        due_date: Optional[datetime] = None,
        source: str = "manual",
        tags: Optional[list[Any]] = None,
    ) -> Task:
        """Persist a new Task and return it."""
        task = Task(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            priority=priority,
            due_date=due_date,
            source=source,
            tags=tags or [],
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
        async with self._sf() as session:
            session.add(task)
            await session.commit()
            await session.refresh(task)
        log.debug("Created task {id!r}: {title!r}", id=task.id, title=task.title)
        return task

    # ------------------------------------------------------------------ Read

    async def get(self, task_id: str) -> Optional[Task]:
        """Return the Task with *task_id*, or None."""
        async with self._sf() as session:
            result = await session.execute(
                select(Task).where(Task.id == task_id)
            )
            return result.scalar_one_or_none()

    async def get_all(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 50,
    ) -> list[Task]:
        """
        Return up to *limit* tasks, optionally filtered by *status* and/or
        *priority*.  Results are ordered newest-first.
        """
        stmt = select(Task).order_by(Task.created_at.desc()).limit(limit)
        if status is not None:
            stmt = stmt.where(Task.status == status)
        if priority is not None:
            stmt = stmt.where(Task.priority == priority)

        async with self._sf() as session:
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def search(self, query: str) -> list[Task]:
        """
        Return tasks whose *title* or *description* contains *query*
        (case-insensitive LIKE search).
        """
        like = f"%{query}%"
        stmt = (
            select(Task)
            .where(
                or_(
                    Task.title.ilike(like),
                    Task.description.ilike(like),
                )
            )
            .order_by(Task.created_at.desc())
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_due_soon(self, hours_ahead: int = 24) -> list[Task]:
        """Return pending/in-progress tasks due within *hours_ahead* hours."""
        now = _utcnow()
        cutoff = now + timedelta(hours=hours_ahead)
        stmt = (
            select(Task)
            .where(
                Task.due_date.is_not(None),
                Task.due_date <= cutoff,
                Task.due_date >= now,
                Task.status.in_(["pending", "in_progress"]),
            )
            .order_by(Task.due_date.asc())
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_overdue(self) -> list[Task]:
        """Return pending/in-progress tasks whose due_date is in the past."""
        now = _utcnow()
        stmt = (
            select(Task)
            .where(
                Task.due_date.is_not(None),
                Task.due_date < now,
                Task.status.in_(["pending", "in_progress"]),
            )
            .order_by(Task.due_date.asc())
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return list(result.scalars().all())

    # ------------------------------------------------------------------ Update

    async def update(self, task_id: str, **fields: Any) -> Optional[Task]:
        """
        Update arbitrary *fields* on a task and return the refreshed object,
        or None if the task does not exist.
        """
        # Always bump updated_at.
        fields["updated_at"] = _utcnow()
        async with self._sf() as session:
            await session.execute(
                update(Task).where(Task.id == task_id).values(**fields)
            )
            await session.commit()
            result = await session.execute(select(Task).where(Task.id == task_id))
            return result.scalar_one_or_none()

    async def complete(self, task_id: str) -> Optional[Task]:
        """Mark a task as completed and record *completed_at*."""
        return await self.update(
            task_id,
            status="completed",
            completed_at=_utcnow(),
        )

    # ------------------------------------------------------------------ Delete

    async def delete(self, task_id: str) -> bool:
        """Hard-delete a task.  Returns True if a row was deleted."""
        async with self._sf() as session:
            result = await session.execute(
                delete(Task).where(Task.id == task_id)
            )
            await session.commit()
            deleted = result.rowcount > 0
        log.debug("Deleted task {id!r}: {ok}", id=task_id, ok=deleted)
        return deleted


# ---------------------------------------------------------------------------
# ConversationStore
# ---------------------------------------------------------------------------


class ConversationStore:
    """Async CRUD for :class:`~memory.models.ConversationTurn`."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def save_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
        action_type: Optional[str] = None,
        action_params: Optional[dict[str, Any]] = None,
        tokens: int = 0,
    ) -> ConversationTurn:
        """Append a new turn to *conversation_id* and return it."""
        turn = ConversationTurn(
            conversation_id=conversation_id,
            role=role,
            content=content,
            action_type=action_type,
            action_params=action_params,
            tokens_used=tokens,
            created_at=_utcnow(),
        )
        async with self._sf() as session:
            session.add(turn)
            await session.commit()
            await session.refresh(turn)
        return turn

    async def get_history(
        self, conversation_id: str, limit: int = 20
    ) -> list[ConversationTurn]:
        """
        Return up to *limit* turns for *conversation_id*, ordered
        chronologically (oldest first).
        """
        stmt = (
            select(ConversationTurn)
            .where(ConversationTurn.conversation_id == conversation_id)
            .order_by(ConversationTurn.created_at.asc())
            .limit(limit)
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_all_conversations(self) -> list[str]:
        """Return a list of distinct conversation_ids."""
        stmt = select(
            distinct(ConversationTurn.conversation_id)
        ).order_by(ConversationTurn.conversation_id)
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [row[0] for row in result.all()]

    async def delete_conversation(self, conversation_id: str) -> int:
        """Delete all turns for *conversation_id*.  Returns the number of rows deleted."""
        async with self._sf() as session:
            result = await session.execute(
                delete(ConversationTurn).where(
                    ConversationTurn.conversation_id == conversation_id
                )
            )
            await session.commit()
            return result.rowcount


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------


class MemoryStore:
    """Async CRUD for :class:`~memory.models.Memory`."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def add(
        self,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
        tags: Optional[list[Any]] = None,
        source_conv_id: Optional[str] = None,
    ) -> Memory:
        """Persist a new memory item and return it."""
        mem = Memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or [],
            source_conversation_id=source_conv_id,
            created_at=_utcnow(),
        )
        async with self._sf() as session:
            session.add(mem)
            await session.commit()
            await session.refresh(mem)
        log.debug(
            "Stored memory id={id} type={t!r} importance={imp:.2f}",
            id=mem.id,
            t=memory_type,
            imp=importance,
        )
        return mem

    async def search(self, query: str, limit: int = 10) -> list[Memory]:
        """
        Return up to *limit* memories whose content contains *query*
        (case-insensitive LIKE).  Results are ordered by importance desc.
        """
        like = f"%{query}%"
        stmt = (
            select(Memory)
            .where(Memory.content.ilike(like))
            .order_by(Memory.importance.desc(), Memory.created_at.desc())
            .limit(limit)
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_by_type(self, memory_type: str) -> list[Memory]:
        """Return all memories of *memory_type*, ordered by importance desc."""
        stmt = (
            select(Memory)
            .where(Memory.memory_type == memory_type)
            .order_by(Memory.importance.desc(), Memory.created_at.desc())
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def increment_access(self, memory_id: int) -> None:
        """Increment *access_count* and update *last_accessed_at* for a memory."""
        now = _utcnow()
        async with self._sf() as session:
            await session.execute(
                update(Memory)
                .where(Memory.id == memory_id)
                .values(
                    access_count=Memory.access_count + 1,
                    last_accessed_at=now,
                )
            )
            await session.commit()

    async def get_important(self, min_importance: float = 0.7) -> list[Memory]:
        """Return all memories with importance ≥ *min_importance*."""
        stmt = (
            select(Memory)
            .where(Memory.importance >= min_importance)
            .order_by(Memory.importance.desc())
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return list(result.scalars().all())


# ---------------------------------------------------------------------------
# TelegramMessageStore
# ---------------------------------------------------------------------------


class TelegramMessageStore:
    """Async CRUD for :class:`~memory.models.TelegramMessage`."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def save(
        self,
        telegram_msg_id: int,
        chat_id: int,
        chat_name: Optional[str],
        sender: str,
        text: Optional[str],
        has_media: bool = False,
        is_outgoing: bool = False,
    ) -> TelegramMessage:
        """Persist an incoming or outgoing Telegram message and return it."""
        msg = TelegramMessage(
            telegram_msg_id=telegram_msg_id,
            chat_id=chat_id,
            chat_name=chat_name,
            sender=sender,
            text=text,
            has_media=has_media,
            is_outgoing=is_outgoing,
            processed=False,
            created_at=_utcnow(),
        )
        async with self._sf() as session:
            session.add(msg)
            await session.commit()
            await session.refresh(msg)
        return msg

    async def get_chat_messages(
        self, chat_id: int, limit: int = 50
    ) -> list[TelegramMessage]:
        """Return the latest *limit* messages for *chat_id*."""
        stmt = (
            select(TelegramMessage)
            .where(TelegramMessage.chat_id == chat_id)
            .order_by(TelegramMessage.created_at.desc())
            .limit(limit)
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def mark_processed(self, msg_id: int) -> None:
        """Set *processed=True* on the TelegramMessage with primary-key *msg_id*."""
        async with self._sf() as session:
            await session.execute(
                update(TelegramMessage)
                .where(TelegramMessage.id == msg_id)
                .values(processed=True)
            )
            await session.commit()

    async def get_unprocessed(self) -> list[TelegramMessage]:
        """Return all TelegramMessages that have not yet been processed."""
        stmt = (
            select(TelegramMessage)
            .where(TelegramMessage.processed == False)  # noqa: E712
            .order_by(TelegramMessage.created_at.asc())
        )
        async with self._sf() as session:
            result = await session.execute(stmt)
            return list(result.scalars().all())
