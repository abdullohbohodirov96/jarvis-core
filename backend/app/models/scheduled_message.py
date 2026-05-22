"""
ScheduledMessage ORM model.

Represents a Telegram message that is queued for delivery at a specific future
time.  A background Celery task polls the table, sends pending messages via the
Telegram client, and updates the status / timestamps accordingly.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import TimestampedBase

if TYPE_CHECKING:
    from app.models.user import User


class ScheduledMessageStatus(str, enum.Enum):
    """Delivery lifecycle state of a scheduled message."""

    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScheduledMessage(TimestampedBase):
    """A Telegram message queued for future delivery."""

    __tablename__ = "scheduled_messages"

    # ── Foreign keys ─────────────────────────────────────────────────────────

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who scheduled this message",
    )

    # ── Target ───────────────────────────────────────────────────────────────

    telegram_chat_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        index=True,
        comment="Telegram chat ID the message will be sent to",
    )

    # ── Content ───────────────────────────────────────────────────────────────

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Message text to be delivered (Markdown supported)",
    )

    # ── Schedule ─────────────────────────────────────────────────────────────

    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="UTC time at which the message should be dispatched",
    )
    sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="UTC timestamp when the message was actually sent; null if not yet sent",
    )

    # ── Delivery state ────────────────────────────────────────────────────────

    status: Mapped[ScheduledMessageStatus] = mapped_column(
        Enum(
            ScheduledMessageStatus,
            name="scheduled_message_status_enum",
            create_type=True,
        ),
        nullable=False,
        default=ScheduledMessageStatus.PENDING,
        server_default=ScheduledMessageStatus.PENDING.value,
        index=True,
        comment="Current delivery lifecycle state",
    )
    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of send attempts made so far",
    )
    error_message: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="Last error description from a failed send attempt",
    )

    # ── Relationships ─────────────────────────────────────────────────────────

    user: Mapped["User"] = relationship(
        "User",
        back_populates="scheduled_messages",
        lazy="select",
    )

    def __repr__(self) -> str:
        return (
            f"<ScheduledMessage id={self.id!r} "
            f"telegram_chat_id={self.telegram_chat_id!r} "
            f"scheduled_at={self.scheduled_at!r} "
            f"status={self.status!r}>"
        )
