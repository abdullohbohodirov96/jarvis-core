"""
JARVIS SQLAlchemy 2.0 async ORM models.

Tables:
  tasks                — user/AI task management
  conversation_turns   — per-conversation message history
  telegram_messages    — raw inbound/outbound Telegram messages
  memories             — long-term agent memory store

Legacy tables kept for backward compatibility:
  conversations        — conversation session header
  conversation_messages — legacy per-conversation messages
  scheduled_messages   — Telegram scheduled-message tracking
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #


def _utcnow() -> datetime:
    """Return the current UTC time (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


# --------------------------------------------------------------------------- #
# Legacy enums (kept for backward compat)                                       #
# --------------------------------------------------------------------------- #


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DELETED = "deleted"


class TaskPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class MessageRole(str, enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# --------------------------------------------------------------------------- #
# Task                                                                          #
# --------------------------------------------------------------------------- #


class Task(Base):
    """A tracked task / to-do item created by voice, Telegram, or the UI."""

    __tablename__ = "tasks"

    # UUID stored as a 36-char string so it works with SQLite and Postgres.
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_new_uuid,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # low / medium / high / urgent
    priority: Mapped[str] = mapped_column(
        String(20), nullable=False, default="medium"
    )
    # pending / in_progress / completed / cancelled
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending", index=True
    )

    due_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    reminder_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    reminder_sent: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    # manual / voice / telegram / ai
    source: Mapped[str] = mapped_column(
        String(50), nullable=False, default="manual"
    )

    tags: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)

    is_deleted: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True
    )

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<Task id={self.id!r} title={self.title!r} "
            f"status={self.status!r} priority={self.priority!r}>"
        )


# --------------------------------------------------------------------------- #
# ConversationTurn                                                               #
# --------------------------------------------------------------------------- #


class ConversationTurn(Base):
    """A single turn (message) in a multi-turn conversation with the AI."""

    __tablename__ = "conversation_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True
    )
    # user / assistant / system
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    action_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    action_params: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )

    tokens_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<ConversationTurn id={self.id} "
            f"conv={self.conversation_id!r} role={self.role!r}>"
        )


# --------------------------------------------------------------------------- #
# TelegramMessage                                                                #
# --------------------------------------------------------------------------- #


class TelegramMessage(Base):
    """A raw Telegram message stored for analysis / history."""

    __tablename__ = "telegram_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_msg_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    chat_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    sender: Mapped[str] = mapped_column(String(255), nullable=False)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    has_media: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_outgoing: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    processed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<TelegramMessage id={self.id} "
            f"chat_id={self.chat_id} sender={self.sender!r}>"
        )


# --------------------------------------------------------------------------- #
# Memory                                                                         #
# --------------------------------------------------------------------------- #


class Memory(Base):
    """Long-term factual / preference memory for the JARVIS agent."""

    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # general / preference / fact / event
    memory_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="general"
    )

    # Importance score in [0, 1]; higher = surface more often.
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    tags: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)

    source_conversation_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True
    )

    access_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<Memory id={self.id} type={self.memory_type!r} "
            f"importance={self.importance:.2f}>"
        )


# --------------------------------------------------------------------------- #
# Legacy models (backward compat — do not use in new code)                      #
# --------------------------------------------------------------------------- #


class Conversation(Base):
    """Legacy conversation session header (kept for DB compat)."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    messages: Mapped[list[ConversationMessage]] = relationship(
        "ConversationMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.created_at",
    )

    def __repr__(self) -> str:
        return f"<Conversation id={self.id!r}>"


class ConversationMessage(Base):
    """Legacy single message within a Conversation (kept for DB compat)."""

    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(
        SAEnum(MessageRole, native_enum=False), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    conversation: Mapped[Conversation] = relationship(
        "Conversation", back_populates="messages"
    )

    def __repr__(self) -> str:
        return (
            f"<ConversationMessage id={self.id} role={self.role.value} "
            f"conversation_id={self.conversation_id!r}>"
        )


class ScheduledMessage(Base):
    """Legacy scheduled Telegram message (kept for DB compat)."""

    __tablename__ = "scheduled_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    send_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    sent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    sent_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<ScheduledMessage id={self.id} chat_id={self.chat_id!r} "
            f"send_at={self.send_at!r} sent={self.sent}>"
        )
