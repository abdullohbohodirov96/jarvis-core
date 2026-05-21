"""
Message ORM model.

Stores individual chat messages within a conversation.  Each row captures the
role (user / assistant / system / tool), the text content, token usage, and
optional Telegram metadata for cross-platform traceability.
"""

from __future__ import annotations

import enum
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import TimestampedBase

if TYPE_CHECKING:
    from backend.app.models.conversation import Conversation
    from backend.app.models.user import User


class MessageRole(str, enum.Enum):
    """OpenAI-compatible message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessageSource(str, enum.Enum):
    """Surface that produced this message."""

    CHAT = "chat"
    TELEGRAM = "telegram"
    VOICE = "voice"


class Message(TimestampedBase):
    """A single turn in a conversation."""

    __tablename__ = "messages"

    # ── Foreign keys ─────────────────────────────────────────────────────────

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent conversation",
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Null for AI-generated (assistant / system / tool) messages",
    )

    # ── Content ───────────────────────────────────────────────────────────────

    role: Mapped[MessageRole] = mapped_column(
        Enum(
            MessageRole,
            name="message_role_enum",
            create_type=True,
        ),
        nullable=False,
        comment="Conversation role of this message",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Raw text content of the message",
    )

    # ── Model / token metadata ────────────────────────────────────────────────

    tokens_used: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Total tokens consumed by this message (prompt + completion)",
    )
    model_used: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Model identifier used to generate this message (e.g. gpt-4o)",
    )

    # ── Surface-specific fields ───────────────────────────────────────────────

    source: Mapped[MessageSource] = mapped_column(
        Enum(
            MessageSource,
            name="message_source_enum",
            create_type=True,
        ),
        nullable=False,
        default=MessageSource.CHAT,
        server_default=MessageSource.CHAT.value,
        comment="Interface that produced this message",
    )
    telegram_message_id: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        comment="Telegram message ID for messages mirrored from Telegram",
    )
    telegram_chat_id: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        comment="Telegram chat ID for messages mirrored from Telegram",
    )

    # ── Extra data ────────────────────────────────────────────────────────────

    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSON,
        nullable=True,
        default=dict,
        comment="Arbitrary extra data (tool call results, attachments, etc.)",
    )

    # ── Relationships ─────────────────────────────────────────────────────────

    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages",
        lazy="select",
    )
    user: Mapped["User | None"] = relationship(
        "User",
        foreign_keys=[user_id],
        lazy="select",
    )

    def __repr__(self) -> str:
        return (
            f"<Message id={self.id!r} "
            f"role={self.role!r} "
            f"conversation_id={self.conversation_id!r}>"
        )
