"""
Memory ORM model.

Represents a single item in the AI's long-term memory store.  Items are
retrieved via semantic search (embedding similarity) and a recency/importance
ranking.  The decay_factor is reduced by a background job to deprioritise stale
memories; an optional expires_at lets transient facts be cleaned up.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import TimestampedBase

if TYPE_CHECKING:
    from app.models.conversation import Conversation
    from app.models.user import User


class MemoryType(str, enum.Enum):
    """Semantic category of a memory item."""

    FACT = "fact"               # Objective fact about the user or world
    PREFERENCE = "preference"   # User likes/dislikes, habits
    EVENT = "event"             # Something that happened at a point in time
    RELATIONSHIP = "relationship"  # Information about people the user knows
    SKILL = "skill"             # Capabilities the user wants JARVIS to remember
    CONTEXT = "context"         # Short-lived situational context


class Memory(TimestampedBase):
    """A single unit of long-term memory associated with a user."""

    __tablename__ = "memories"

    # ── Foreign keys ─────────────────────────────────────────────────────────

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User whose memory this item belongs to",
    )
    source_conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        comment="Conversation from which this memory was extracted",
    )

    # ── Content ───────────────────────────────────────────────────────────────

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Natural-language statement of the memory",
    )
    memory_type: Mapped[MemoryType] = mapped_column(
        Enum(
            MemoryType,
            name="memory_type_enum",
            create_type=True,
        ),
        nullable=False,
        default=MemoryType.FACT,
        server_default=MemoryType.FACT.value,
        index=True,
        comment="Semantic category used for targeted retrieval",
    )

    # ── Ranking signals ───────────────────────────────────────────────────────

    importance: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.5,
        server_default="0.5",
        comment="User/AI assigned importance score in the range [0.0, 1.0]",
    )
    decay_factor: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
        server_default="1.0",
        comment="Multiplier reduced over time; effective_score = importance * decay_factor",
    )

    # ── Embedding ─────────────────────────────────────────────────────────────

    embedding: Mapped[list[float] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Dense vector embedding of *content* for similarity search",
    )

    # ── Classification ────────────────────────────────────────────────────────

    tags: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
        default=list,
        comment="Free-form labels for filtering",
    )

    # ── Access tracking ───────────────────────────────────────────────────────

    access_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of times this memory has been retrieved",
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="UTC timestamp of the most recent retrieval",
    )

    # ── Expiry ────────────────────────────────────────────────────────────────

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="UTC timestamp after which this memory should be ignored or deleted",
    )

    # ── Relationships ─────────────────────────────────────────────────────────

    user: Mapped["User"] = relationship(
        "User",
        back_populates="memories",
        lazy="select",
    )
    source_conversation: Mapped["Conversation | None"] = relationship(
        "Conversation",
        foreign_keys=[source_conversation_id],
        lazy="select",
    )

    def __repr__(self) -> str:
        snippet = (self.content[:40] + "…") if len(self.content) > 40 else self.content
        return (
            f"<Memory id={self.id!r} "
            f"type={self.memory_type!r} "
            f"importance={self.importance} "
            f"content={snippet!r}>"
        )
