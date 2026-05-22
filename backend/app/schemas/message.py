"""
Message and Conversation Pydantic v2 schemas.

Covers conversation CRUD, individual message I/O, the chat request/response
cycle, and the Server-Sent Events (SSE) streaming chunk format.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.conversation import ConversationSource
from app.models.message import MessageRole, MessageSource


# ---------------------------------------------------------------------------
# Conversation schemas
# ---------------------------------------------------------------------------

class ConversationCreate(BaseModel):
    """Schema for creating a new conversation thread."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=512)
    source: ConversationSource = Field(default=ConversationSource.CHAT)
    context: dict[str, Any] | None = Field(
        default=None,
        description="Initial context dict injected into the system prompt",
    )


class ConversationUpdate(BaseModel):
    """Schema for PATCH /conversations/{id}."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=512)
    summary: str | None = None
    context: dict[str, Any] | None = None
    is_active: bool | None = None


class ConversationResponse(BaseModel):
    """Serialised conversation returned to the client."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    title: str | None
    summary: str | None
    context: dict[str, Any] | None
    source: ConversationSource
    is_active: bool
    message_count: int
    last_message_at: datetime | None
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Message schemas
# ---------------------------------------------------------------------------

class MessageCreate(BaseModel):
    """Schema for inserting a new message (internal use / API ingestion)."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    role: MessageRole
    content: str = Field(min_length=1)
    tokens_used: int | None = Field(default=None, ge=0)
    model_used: str | None = Field(default=None, max_length=128)
    source: MessageSource = Field(default=MessageSource.CHAT)
    telegram_message_id: int | None = None
    telegram_chat_id: int | None = None
    metadata: dict[str, Any] | None = None


class MessageResponse(BaseModel):
    """Serialised message returned to the client."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    user_id: UUID | None
    role: MessageRole
    content: str
    tokens_used: int | None
    model_used: str | None
    source: MessageSource
    telegram_message_id: int | None
    telegram_chat_id: int | None
    metadata: dict[str, Any] | None = Field(None, alias="metadata_")
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# ---------------------------------------------------------------------------
# Chat request / response
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """
    Payload for POST /chat/message — the primary human turn.

    If *conversation_id* is None a new conversation is created automatically.
    Set *stream* to True to receive a text/event-stream (SSE) response instead
    of a single JSON object.
    """

    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        min_length=1,
        max_length=32_000,
        description="The user's natural-language input",
    )
    conversation_id: UUID | None = Field(
        default=None,
        description="Continue an existing conversation; omit to start a new one",
    )
    stream: bool = Field(
        default=False,
        description="If True, stream the response as SSE chunks",
    )
    context_override: dict[str, Any] | None = Field(
        default=None,
        description="One-off context keys to merge into the system prompt for this turn",
    )


class ChatResponse(BaseModel):
    """Non-streaming chat completion response."""

    message: str = Field(description="Full AI reply text")
    conversation_id: UUID = Field(description="ID of the conversation this turn belongs to")
    message_id: UUID = Field(description="ID of the newly created assistant message")
    tokens_used: int = Field(ge=0, description="Total tokens consumed (prompt + completion)")
    model_used: str = Field(description="Model identifier that generated the response")


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------

class StreamChunk(BaseModel):
    """
    A single Server-Sent Event chunk for streaming chat responses.

    Event wire format::

        data: {"type": "delta", "content": "Hello", "conversation_id": "..."}\\n\\n

    *type* values:

    - ``start``   — sent once at the beginning; carries ``conversation_id``
    - ``delta``   — incremental text token(s)
    - ``end``     — sent once when the stream is complete; carries summary stats
    - ``error``   — sent if an unrecoverable error occurs mid-stream
    """

    type: str = Field(description="Event type: start | delta | end | error")
    content: str | None = Field(
        default=None,
        description="Text delta (present for 'delta' events only)",
    )
    conversation_id: UUID | None = Field(
        default=None,
        description="Conversation ID (present in 'start' and 'end' events)",
    )
    message_id: UUID | None = Field(
        default=None,
        description="Persisted message ID (present in 'end' event)",
    )
    tokens_used: int | None = Field(
        default=None,
        description="Total tokens (present in 'end' event)",
    )
    error: str | None = Field(
        default=None,
        description="Error message (present in 'error' event)",
    )
