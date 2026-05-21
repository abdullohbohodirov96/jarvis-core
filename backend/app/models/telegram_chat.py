"""
TelegramChat ORM model.

Tracks a Telegram chat (private DM, group, channel, or supergroup) that a user
has linked to their JARVIS account.  When ``is_monitored`` is True, incoming
messages are ingested and processed by the AI.  ``auto_reply_enabled`` allows
JARVIS to send autonomous replies.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, Boolean, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import UniqueConstraint

from backend.app.models.base import TimestampedBase

if TYPE_CHECKING:
    from backend.app.models.user import User


class TelegramChatType(str, enum.Enum):
    """Telegram chat type categories."""

    PRIVATE = "private"
    GROUP = "group"
    CHANNEL = "channel"
    SUPERGROUP = "supergroup"


class TelegramChat(TimestampedBase):
    """A Telegram chat tracked for a specific JARVIS user."""

    __tablename__ = "telegram_chats"

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "telegram_chat_id",
            name="uq_telegram_chats_user_chat",
        ),
    )

    # ── Foreign keys ─────────────────────────────────────────────────────────

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="JARVIS user who owns this chat link",
    )

    # ── Telegram identifiers ──────────────────────────────────────────────────

    telegram_chat_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        index=True,
        comment="Telegram's numeric chat identifier (may be negative for groups)",
    )
    chat_title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Human-readable name of the chat as reported by Telegram",
    )
    chat_type: Mapped[TelegramChatType] = mapped_column(
        Enum(
            TelegramChatType,
            name="telegram_chat_type_enum",
            create_type=True,
        ),
        nullable=False,
        default=TelegramChatType.PRIVATE,
        server_default=TelegramChatType.PRIVATE.value,
        comment="Telegram chat category",
    )

    # ── Behaviour flags ───────────────────────────────────────────────────────

    is_monitored: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="Ingest messages from this chat into JARVIS",
    )
    auto_reply_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="Allow JARVIS to send autonomous replies in this chat",
    )

    # ── Sync state ────────────────────────────────────────────────────────────

    last_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="UTC timestamp of the last successful message sync",
    )
    unread_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of unread messages as reported at last sync",
    )

    # ── Extra metadata ────────────────────────────────────────────────────────

    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSON,
        nullable=True,
        default=dict,
        comment="Additional Telegram chat attributes (description, member count, etc.)",
    )

    # ── Relationships ─────────────────────────────────────────────────────────

    user: Mapped["User"] = relationship(
        "User",
        back_populates="telegram_chats",
        lazy="select",
    )

    def __repr__(self) -> str:
        return (
            f"<TelegramChat id={self.id!r} "
            f"telegram_chat_id={self.telegram_chat_id!r} "
            f"type={self.chat_type!r} "
            f"monitored={self.is_monitored}>"
        )
