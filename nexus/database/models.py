from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from database.connection import Base


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


class Task(Base):
    """
    Represents a tracked task extracted from a user message or created
    explicitly through the API.

    Columns
    -------
    id           : auto-incrementing integer primary key
    title        : short, mandatory task name (max 512 chars)
    description  : optional longer description / context
    is_completed : completion flag; defaults to False
    created_at   : UTC timestamp set at INSERT time, never updated
    """

    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        index=True,
        doc="Short, human-readable task title.",
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
        doc="Optional expanded description or background context.",
    )

    is_completed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        doc="Set to True once the task is finished.",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        doc="UTC timestamp of when this task record was created.",
    )

    def __repr__(self) -> str:
        status = "✓" if self.is_completed else "○"
        return f"<Task id={self.id} [{status}] {self.title!r}>"
