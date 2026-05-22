"""
JARVIS SQLAlchemy ORM models.

All models inherit from ``core.database.Base`` so that ``init_database()``
can discover and create their tables automatically.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


# --------------------------------------------------------------------------- #
# Enums                                                                         #
# --------------------------------------------------------------------------- #


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DELETED = "deleted"  # soft-delete sentinel


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
# Task                                                                           #
# --------------------------------------------------------------------------- #


class Task(Base):
    """A user-facing task tracked by JARVIS."""

    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[TaskStatus] = mapped_column(
        SAEnum(TaskStatus, native_enum=False),
        default=TaskStatus.PENDING,
        server_default=TaskStatus.PENDING.value,
        nullable=False,
        index=True,
    )
    priority: Mapped[TaskPriority] = mapped_column(
        SAEnum(TaskPriority, native_enum=False),
        default=TaskPriority.MEDIUM,
        server_default=TaskPriority.MEDIUM.value,
        nullable=False,
        index=True,
    )

    due_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Soft-delete flag (status == DELETED is the canonical check,
    # but this column lets us filter quickly).
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Optional source: "voice", "telegram", "api", "ui"
    source: Mapped[str | None] = mapped_column(String(64), nullable=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Task id={self.id} title={self.title!r} status={self.status.value}>"


# --------------------------------------------------------------------------- #
# Conversation                                                                   #
# --------------------------------------------------------------------------- #


class Conversation(Base):
    """A conversation session with the JARVIS AI assistant."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
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

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Conversation id={self.id!r}>"


class ConversationMessage(Base):
    """A single message within a Conversation."""

    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(
        SAEnum(MessageRole, native_enum=False),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    conversation: Mapped[Conversation] = relationship(
        "Conversation", back_populates="messages"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ConversationMessage id={self.id} role={self.role.value} "
            f"conversation_id={self.conversation_id!r}>"
        )


# --------------------------------------------------------------------------- #
# ScheduledMessage                                                               #
# --------------------------------------------------------------------------- #


class ScheduledMessage(Base):
    """A Telegram message scheduled for future delivery."""

    __tablename__ = "scheduled_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    send_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    sent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ScheduledMessage id={self.id} chat_id={self.chat_id!r} "
            f"send_at={self.send_at!r} sent={self.sent}>"
        )
