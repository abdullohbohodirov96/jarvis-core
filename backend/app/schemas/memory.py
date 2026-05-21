"""
Memory-related Pydantic v2 schemas.

Covers long-term memory CRUD and semantic search.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.models.memory import MemoryType


# ---------------------------------------------------------------------------
# Memory CRUD schemas
# ---------------------------------------------------------------------------

class MemoryBase(BaseModel):
    """Common optional fields shared by create and update schemas."""

    memory_type: MemoryType | None = None
    importance: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Importance score in [0.0, 1.0]; higher = retrieved more often",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Free-form labels for targeted retrieval",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="UTC expiry timestamp; null = never expires",
    )


class MemoryCreate(MemoryBase):
    """Schema for POST /memories."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(
        min_length=1,
        max_length=8_000,
        description="Natural-language statement to store as a memory",
    )
    memory_type: MemoryType = Field(
        default=MemoryType.FACT,
        description="Semantic category of this memory",
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    source_conversation_id: UUID | None = Field(
        default=None,
        description="Conversation from which this memory was extracted",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Pre-computed dense vector; computed server-side if omitted",
    )

    @field_validator("embedding")
    @classmethod
    def validate_embedding_length(cls, v: list[float] | None) -> list[float] | None:
        """Guard against obviously wrong embedding dimensions."""
        if v is not None and len(v) == 0:
            raise ValueError("embedding must be a non-empty list of floats")
        return v


class MemoryUpdate(MemoryBase):
    """Schema for PATCH /memories/{id} — all fields optional."""

    model_config = ConfigDict(extra="forbid")

    content: str | None = Field(default=None, min_length=1, max_length=8_000)
    embedding: list[float] | None = None
    decay_factor: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override the decay multiplier (normally managed by background job)",
    )


class MemoryResponse(BaseModel):
    """Serialised memory item returned to the client."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    content: str
    memory_type: MemoryType
    importance: float
    decay_factor: float
    # Embedding is intentionally excluded from default responses (large payload)
    tags: list[str] | None
    source_conversation_id: UUID | None
    access_count: int
    last_accessed_at: datetime | None
    expires_at: datetime | None
    created_at: datetime
    updated_at: datetime

    @property
    def effective_score(self) -> float:
        """Composite retrieval score = importance × decay_factor."""
        return self.importance * self.decay_factor


# ---------------------------------------------------------------------------
# Search schemas
# ---------------------------------------------------------------------------

class MemorySearchRequest(BaseModel):
    """
    Request payload for semantic memory search.

    The service embeds *query*, performs cosine-similarity search over stored
    embeddings, and returns the top *limit* results optionally filtered by
    *memory_type*.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=2_000,
        description="Natural-language query used for semantic search",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of memories to return",
    )
    memory_type: MemoryType | None = Field(
        default=None,
        description="Restrict results to a specific memory type",
    )
    min_importance: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Exclude memories below this importance threshold",
    )
    include_expired: bool = Field(
        default=False,
        description="If False (default), exclude memories past their expires_at",
    )


class MemorySearchResult(BaseModel):
    """A single search hit with its similarity score."""

    memory: MemoryResponse
    similarity: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine similarity between the query embedding and this memory",
    )


class MemorySearchResponse(BaseModel):
    """Search result envelope."""

    results: list[MemorySearchResult] = Field(
        description="Ranked list of matching memories (most similar first)"
    )
    query: str = Field(description="The original search query")
    total_searched: int = Field(
        ge=0,
        description="Number of candidate memories examined before ranking",
    )
