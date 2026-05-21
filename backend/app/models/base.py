"""
Shared abstract base class for all SQLAlchemy ORM models.

Every concrete model inherits TimestampedBase and gets:
  - UUID primary key
  - Timezone-aware created_at / updated_at timestamps
  - Soft-delete flag (is_deleted)
  - to_dict() and __repr__() helpers
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.connection import Base


def _utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


class TimestampedBase(Base):
    """
    Abstract base model providing common columns and helpers.

    All subclasses must define ``__tablename__``.
    """

    __abstract__ = True

    # ── Primary key ───────────────────────────────────────────────────────────

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
        comment="Universally unique row identifier",
    )

    # ── Audit timestamps ──────────────────────────────────────────────────────

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
        comment="UTC timestamp when the row was first created",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
        onupdate=_utcnow,
        comment="UTC timestamp of the most recent modification",
    )

    # ── Soft-delete ───────────────────────────────────────────────────────────

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        index=True,
        comment="Soft-delete flag; rows with is_deleted=true are hidden from queries",
    )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def to_dict(self, *, exclude: set[str] | None = None) -> dict[str, Any]:
        """
        Serialise the model to a plain dictionary.

        *exclude* is an optional set of column names to omit.  UUID values are
        converted to strings so the result is JSON-serialisable by default.
        """
        omit = exclude or set()
        result: dict[str, Any] = {}
        for col in self.__table__.columns:  # type: ignore[attr-defined]
            if col.name in omit:
                continue
            value = getattr(self, col.name)
            if isinstance(value, uuid.UUID):
                value = str(value)
            elif isinstance(value, datetime):
                value = value.isoformat()
            result[col.name] = value
        return result

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} "
            f"id={self.id!r} "
            f"created_at={self.created_at!r}>"
        )
