"""
User ORM model.

Represents an authenticated human user of the JARVIS platform.  A user can
interact via the web chat, Telegram bot, or voice interface.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, Boolean, DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import TimestampedBase

if TYPE_CHECKING:
    from backend.app.models.conversation import Conversation
    from backend.app.models.memory import Memory
    from backend.app.models.task import Task
    from backend.app.models.telegram_chat import TelegramChat
    from backend.app.models.scheduled_message import ScheduledMessage


class User(TimestampedBase):
    """Platform user account."""

    __tablename__ = "users"

    # ── Identity ──────────────────────────────────────────────────────────────

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="Primary e-mail address used for login",
    )
    username: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        comment="Public display handle (no spaces)",
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Bcrypt-hashed password; never store plaintext",
    )
    full_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Optional human-readable display name",
    )

    # ── Authorisation flags ───────────────────────────────────────────────────

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        comment="Inactive users cannot log in",
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="Grants unrestricted access to all platform resources",
    )

    # ── Telegram integration ──────────────────────────────────────────────────

    telegram_user_id: Mapped[int | None] = mapped_column(
        BigInteger,
        unique=True,
        nullable=True,
        index=True,
        comment="Telegram numeric user ID (null if not linked)",
    )
    telegram_username: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Telegram @username without the at-sign",
    )

    # ── Feature flags ─────────────────────────────────────────────────────────

    voice_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="Whether voice input/output is enabled for this user",
    )

    # ── Preferences & state ───────────────────────────────────────────────────

    preferences: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        default=dict,
        comment="Arbitrary user preferences (timezone, language, theme, etc.)",
    )
    last_seen: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="UTC timestamp of the user's most recent activity",
    )

    # ── Relationships ─────────────────────────────────────────────────────────

    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="desc(Conversation.last_message_at)",
    )
    tasks: Mapped[list["Task"]] = relationship(
        "Task",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
    )
    memories: Mapped[list["Memory"]] = relationship(
        "Memory",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
    )
    telegram_chats: Mapped[list["TelegramChat"]] = relationship(
        "TelegramChat",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
    )
    scheduled_messages: Mapped[list["ScheduledMessage"]] = relationship(
        "ScheduledMessage",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
    )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"<User id={self.id!r} "
            f"username={self.username!r} "
            f"email={self.email!r}>"
        )
