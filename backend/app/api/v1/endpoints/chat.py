"""
Chat endpoints for JARVIS.

POST   /chat/message                     – send a message, stream AI response via SSE
GET    /chat/history/{conversation_id}   – retrieve conversation history
DELETE /chat/conversation/{conversation_id} – clear / delete a conversation
POST   /chat/summary/{conversation_id}   – generate a summary of the conversation
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.app.core.exceptions import AIException, NotFoundException
from backend.app.core.logging_config import get_logger
from backend.app.utils.helpers import generate_id, now_utc

logger = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class MessageRequest(BaseModel):
    """Incoming chat message from the user."""

    message: str = Field(..., min_length=1, max_length=32_000, description="User message text")
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID; a new one is created if omitted"
    )
    stream: bool = Field(default=True, description="Whether to stream the response via SSE")
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional extra context passed to the AI"
    )


class MessageResponse(BaseModel):
    """Non-streaming AI response."""

    conversation_id: str
    message_id: str
    role: str = "assistant"
    content: str
    created_at: datetime
    tokens_used: Optional[int] = None


class ChatMessage(BaseModel):
    """A single message in the conversation history."""

    message_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    created_at: datetime
    tokens_used: Optional[int] = None


class ConversationHistory(BaseModel):
    """Full conversation history response."""

    conversation_id: str
    messages: List[ChatMessage]
    total: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class SummaryResponse(BaseModel):
    """Conversation summary response."""

    conversation_id: str
    summary: str
    message_count: int
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-process conversation store (replace with DB in production)
# ---------------------------------------------------------------------------

# Structure: { conversation_id: [{"role": ..., "content": ..., ...}, ...] }
_conversations: Dict[str, List[Dict[str, Any]]] = {}


def _get_conversation(conversation_id: str) -> List[Dict[str, Any]]:
    return _conversations.get(conversation_id, [])


def _append_message(
    conversation_id: str,
    role: str,
    content: str,
    message_id: Optional[str] = None,
) -> Dict[str, Any]:
    if conversation_id not in _conversations:
        _conversations[conversation_id] = []
    msg: Dict[str, Any] = {
        "message_id": message_id or generate_id(),
        "role": role,
        "content": content,
        "created_at": now_utc().isoformat(),
    }
    _conversations[conversation_id].append(msg)
    return msg


# ---------------------------------------------------------------------------
# AI streaming helper
# ---------------------------------------------------------------------------


async def _stream_ai_response(
    message: str,
    conversation_id: str,
    history: List[Dict[str, Any]],
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted tokens from the AI.

    Attempts to use the OpenAI client.  Falls back to a mock stream when the
    client is unavailable so the endpoint works without credentials in dev.
    """
    try:
        from openai import AsyncOpenAI  # type: ignore
        from backend.app.core.config import get_settings

        settings = get_settings()
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        messages: List[Dict[str, str]] = [
            {"role": m["role"], "content": m["content"]}
            for m in history[-20:]  # limit context window
        ]
        messages.append({"role": "user", "content": message})

        stream = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=settings.OPENAI_MAX_TOKENS,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                payload = json.dumps({"type": "token", "content": delta.content})
                yield f"data: {payload}\n\n"

    except ImportError:
        # openai package not installed — echo mock tokens
        words = f"[Mock response to: {message[:50]}]".split()
        for word in words:
            payload = json.dumps({"type": "token", "content": word + " "})
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.05)

    except Exception as exc:
        logger.error("ai_stream_error", error=str(exc), conversation_id=conversation_id)
        error_payload = json.dumps({"type": "error", "message": str(exc)})
        yield f"data: {error_payload}\n\n"
        return

    done_payload = json.dumps({"type": "done", "conversation_id": conversation_id})
    yield f"data: {done_payload}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/message",
    summary="Send a message and receive an AI response",
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
)
async def send_message(request: MessageRequest) -> StreamingResponse:
    """Send a user message and receive the AI response.

    When ``stream=True`` (default) the response is delivered as Server-Sent
    Events (SSE).  Each event is a JSON object::

        {"type": "token", "content": "..."}   – incremental token
        {"type": "done", "conversation_id": "..."} – end of stream
        {"type": "error", "message": "..."}   – error frame

    When ``stream=False`` the full response is returned as a JSON body.
    """
    conversation_id = request.conversation_id or generate_id()
    history = _get_conversation(conversation_id)

    # Record the user message
    _append_message(conversation_id, "user", request.message)

    if request.stream:
        async def _sse_generator() -> AsyncGenerator[str, None]:
            full_response: list[str] = []
            async for chunk in _stream_ai_response(
                request.message, conversation_id, history
            ):
                # Collect assistant tokens to store in history
                try:
                    event_data = chunk.replace("data: ", "").strip()
                    if event_data:
                        parsed = json.loads(event_data)
                        if parsed.get("type") == "token":
                            full_response.append(parsed.get("content", ""))
                except (json.JSONDecodeError, KeyError):
                    pass
                yield chunk

            # Persist assistant reply
            if full_response:
                _append_message(
                    conversation_id, "assistant", "".join(full_response)
                )

        return StreamingResponse(
            _sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ── Non-streaming path ─────────────────────────────────────────────────
    tokens: List[str] = []
    async for chunk in _stream_ai_response(request.message, conversation_id, history):
        try:
            event_data = chunk.replace("data: ", "").strip()
            if event_data:
                parsed = json.loads(event_data)
                if parsed.get("type") == "token":
                    tokens.append(parsed.get("content", ""))
        except (json.JSONDecodeError, KeyError):
            pass

    content = "".join(tokens)
    msg = _append_message(conversation_id, "assistant", content)

    return MessageResponse(  # type: ignore[return-value]
        conversation_id=conversation_id,
        message_id=msg["message_id"],
        content=content,
        created_at=datetime.fromisoformat(msg["created_at"]),
    )


@router.get(
    "/history/{conversation_id}",
    response_model=ConversationHistory,
    summary="Get conversation history",
    status_code=status.HTTP_200_OK,
)
async def get_history(conversation_id: str) -> ConversationHistory:
    """Return all messages in a conversation, oldest first."""
    messages = _get_conversation(conversation_id)
    if not messages:
        raise NotFoundException(
            message=f"Conversation '{conversation_id}' not found.",
            details={"conversation_id": conversation_id},
        )

    chat_messages = [
        ChatMessage(
            message_id=m["message_id"],
            role=m["role"],
            content=m["content"],
            created_at=datetime.fromisoformat(m["created_at"]),
        )
        for m in messages
    ]

    return ConversationHistory(
        conversation_id=conversation_id,
        messages=chat_messages,
        total=len(chat_messages),
    )


@router.delete(
    "/conversation/{conversation_id}",
    summary="Delete a conversation",
    status_code=status.HTTP_200_OK,
)
async def delete_conversation(conversation_id: str) -> Dict[str, Any]:
    """Permanently delete a conversation and all its messages."""
    if conversation_id not in _conversations:
        raise NotFoundException(
            message=f"Conversation '{conversation_id}' not found.",
            details={"conversation_id": conversation_id},
        )
    del _conversations[conversation_id]
    logger.info("conversation_deleted", conversation_id=conversation_id)
    return {"status": "deleted", "conversation_id": conversation_id}


@router.post(
    "/summary/{conversation_id}",
    response_model=SummaryResponse,
    summary="Generate a conversation summary",
    status_code=status.HTTP_200_OK,
)
async def summarize_conversation(conversation_id: str) -> SummaryResponse:
    """Ask the AI to produce a brief summary of the conversation."""
    messages = _get_conversation(conversation_id)
    if not messages:
        raise NotFoundException(
            message=f"Conversation '{conversation_id}' not found.",
            details={"conversation_id": conversation_id},
        )

    # Build a compact transcript for the summarisation prompt
    transcript_lines = [
        f"{m['role'].upper()}: {m['content'][:500]}"
        for m in messages
    ]
    transcript = "\n".join(transcript_lines)
    prompt = (
        f"Summarise the following conversation in 2-4 sentences:\n\n{transcript}"
    )

    summary_text = ""
    try:
        from openai import AsyncOpenAI  # type: ignore
        from backend.app.core.config import get_settings

        cfg = get_settings()
        client = AsyncOpenAI(api_key=cfg.OPENAI_API_KEY)
        resp = await client.chat.completions.create(
            model=cfg.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            stream=False,
        )
        summary_text = resp.choices[0].message.content or ""
    except ImportError:
        summary_text = f"[Mock summary of {len(messages)} messages in conversation {conversation_id}]"
    except Exception as exc:
        raise AIException(
            message="Failed to generate summary.",
            details={"error": str(exc)},
        ) from exc

    return SummaryResponse(
        conversation_id=conversation_id,
        summary=summary_text.strip(),
        message_count=len(messages),
        generated_at=now_utc(),
    )
