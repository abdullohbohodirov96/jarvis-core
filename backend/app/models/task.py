"""
Task ORM model.

Represents an actionable item created by or for a user.  Tasks may originate
from manual entry, AI extraction from a conversation, a Telegram message, or
voice input.  Recurrence rules follow iCalendar RRULE conventions stored as
JSON.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import TimestampedBase

if TYPE_CHECKING:
    from app.models.message import Message
    from app.models.user import User


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    OVERDUE = "overdue"


class TaskPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TaskSource(str, enum.Enum):
    MANUAL = "manual"
    AI_EXTRACTED = "ai_extracted"
    TELEGRAM = "telegram"
    VOICE = "voice"


class Task(TimestampedBase):
    """An actionable to-do item belonging to a user."""

    __tablename__ = "tasks"

    # ── Foreign keys ─────────────────────────────────────────────────────────

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who owns this task",
    )

    # ── Core fields ───────────────────────────────────────────────────────────

    title: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Short imperative description of the task",
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Optional long-form details or acceptance criteria",
    )

    # ── Status & priority ─────────────────────────────────────────────────────

    status: Mapped[TaskStatus] = mapped_column(
        Enum(
            TaskStatus,
            name="task_status_enum",
            create_type=True,
        ),
        nullable=False,
        default=TaskStatus.PENDING,
        server_default=TaskStatus.PENDING.value,
        index=True,
        comment="Current lifecycle state of the task",
    )
    priority: Mapped[TaskPriority] = mapped_column(
        Enum(
            TaskPriority,
            name="task_priority_enum",
            create_type=True,
        ),
        nullable=False,
        default=TaskPriority.MEDIUM,
        server_default=TaskPriority.MEDIUM.value,
        comment="Urgency ranking used for sorting and reminders",
    )

    # ── Scheduling ────────────────────────────────────────────────────────────

    due_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="UTC deadline; null means no fixed deadline",
    )
    reminder_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="UTC time to fire the reminder notification",
    )
    reminder_sent: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="Set to true once the reminder has been delivered",
    )

    # ── Provenance ────────────────────────────────────────────────────────────

    source: Mapped[TaskSource] = mapped_column(
        Enum(
            TaskSource,
            name="task_source_enum",
            create_type=True,
        ),
        nullable=False,
        default=TaskSource.MANUAL,
        server_default=TaskSource.MANUAL.value,
        comment="How the task was created",
    )
    source_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
        comment="Message that prompted AI extraction of this task",
    )

    # ── Classification ────────────────────────────────────────────────────────

    tags: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
        default=list,
        comment="Free-form labels for filtering and grouping",
    )
    recurrence: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="iCalendar-style RRULE stored as dict; null for one-off tasks",
    )

    # ── Completion ────────────────────────────────────────────────────────────

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="UTC timestamp when the task was marked completed",
    )

    # ── Relationships ─────────────────────────────────────────────────────────

    user: Mapped["User"] = relationship(
        "User",
        back_populates="tasks",
        lazy="select",
    )
    source_message: Mapped["Message | None"] = relationship(
        "Message",
        foreign_keys=[source_message_id],
        lazy="select",
    )

    def __repr__(self) -> str:
        return (
            f"<Task id={self.id!r} "
            f"title={self.title!r} "
            f"status={self.status!r} "
            f"priority={self.priority!r}>"
        )
