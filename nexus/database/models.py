from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Integer, String, Text, BigInteger, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.connection import Base

def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)

class User(Base):
    """
    Represents a whitelisted/registered user of the NEXUS platform.
    """
    __tablename__ = "users"

    tg_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        index=True,
        autoincrement=False,
        doc="Telegram User ID of the user."
    )

    username: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        default=None,
        doc="Telegram username without leading @."
    )

    is_allowed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        doc="Whether this user has access to the Mini App."
    )

    is_admin: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        doc="Whether this user is an admin who can approve/deny others."
    )

    analyzed_chats: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default="'[]'",
        doc="List of Telegram chat IDs/usernames to monitor and analyze."
    )

    pinned_msg_id: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        default=None,
        doc="Message ID of the pinned TODO board in the bot chat."
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        doc="UTC timestamp of when this user registered."
    )

    integrations: Mapped[list["UserIntegration"]] = relationship(
        "UserIntegration",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    todos: Mapped[list["Todo"]] = relationship(
        "Todo",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        status = "admin" if self.is_admin else ("allowed" if self.is_allowed else "pending")
        return f"<User tg_id={self.tg_id} username={self.username!r} status={status}>"

class UserIntegration(Base):
    """
    Stores credentials and state for marketplace integrations connected by a user.
    Credentials are encrypted before saving.
    """
    __tablename__ = "user_integrations"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        index=True
    )

    tg_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.tg_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    app_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Identifier for the integration: e.g. todoist, telegram_bot, instagram, google_sheets"
    )

    credentials: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default="{}",
        doc="Encrypted API credentials in JSON format."
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow
    )

    user: Mapped["User"] = relationship("User", back_populates="integrations")

    def __repr__(self) -> str:
        status = "active" if self.is_active else "inactive"
        return f"<UserIntegration user={self.tg_id} app={self.app_id!r} status={status}>"

class Todo(Base):
    """
    Represents a task associated with a user in the SaaS NEXUS platform.
    """
    __tablename__ = "todos"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        index=True
    )

    tg_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.tg_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    title: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        index=True
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None
    )

    is_done: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false"
    )

    priority: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="medium",
        server_default="medium"
    )

    due_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None
    )

    section: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="vazifalar",
        server_default="vazifalar",
        doc="Section of the kanban board: 'vazifalar' | 'kutilmoqda' | 'keraklilar' | 'bajarildi'"
    )

    source: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="manual",
        server_default="manual",
        doc="How the todo was created: 'manual' | 'promise' | 'request'"
    )

    from_chat_id: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        default=None,
        doc="Telegram chat ID where the promise/request was detected."
    )

    from_chat_name: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        default=None,
        doc="Name of the Telegram chat where the promise/request was detected."
    )

    original_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
        doc="Original message text that triggered todo creation."
    )

    follow_up_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        doc="UTC datetime when a follow-up reminder should be sent."
    )

    follow_up_sent: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        doc="Whether the follow-up reminder has been sent."
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow
    )

    user: Mapped["User"] = relationship("User", back_populates="todos")

    def __repr__(self) -> str:
        status = "done" if self.is_done else "pending"
        return f"<Todo id={self.id} user={self.tg_id} title={self.title!r} status={status}>"
