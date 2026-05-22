"""
Conversation ORM model.

A conversation groups an ordered series of messages between a user and the AI.
It can originate from multiple surfaces (web chat, Telegram, voice, API).
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, Boolean, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import TimestampedBase

if TYPE_CHECKING:
    from app.models.message import Message
    from app.models.user import User


class ConversationSource(str, enum.Enum):
    """Where the conversation was initiated."""

    CHAT = "chat"
    TELEGRAM = "telegram"
    VOICE = "voice"
    API = "api"


class Conversation(TimestampedBase):
    """A thread of messages between a user and the AI assistant."""

    __tablename__ = "conversations"

    # ── Foreign keys ─────────────────────────────────────────────────────────

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Owner of this conversation",
    )

    # ── Metadata ─────────────────────────────────────────────────────────────

    title: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Auto-generated or user-supplied conversation title",
    )
    summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="AI-generated summary of the conversation so far",
    )
    context: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        default=dict,
        comment="Arbitrary context injected into the system prompt (persona, tools, etc.)",
    )
    source: Mapped[ConversationSource] = mapped_column(
        Enum(
            ConversationSource,
            name="conversation_source_enum",
            create_type=True,
        ),
        nullable=False,
        default=ConversationSource.CHAT,
        server_default=ConversationSource.CHAT.value,
        comment="Surface that originated the conversation",
    )

    # ── State ─────────────────────────────────────────────────────────────────

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        comment="Inactive conversations are archived and excluded from suggestions",
    )
    message_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Denormalised count of messages; updated on each insert",
    )
    last_message_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="UTC timestamp of the most recent message in this conversation",
    )

    # ── Relationships ─────────────────────────────────────────────────────────

    user: Mapped["User"] = relationship(
        "User",
        back_populates="conversations",
        lazy="select",
    )
    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="Message.created_at",
    )

    def __repr__(self) -> str:
        return (
            f"<Conversation id={self.id!r} "
            f"user_id={self.user_id!r} "
            f"source={self.source!r} "
            f"messages={self.message_count}>"
        )
