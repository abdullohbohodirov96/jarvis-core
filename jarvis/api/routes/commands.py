"""
AI command routes for JARVIS.

Prefix: /api/commands

Endpoints
---------
POST /chat                — send a message, receive a full response
POST /chat/stream         — SSE streaming of tokens
POST /voice/transcribe    — upload audio, get back transcription text
POST /voice/speak         — text to speech, returns audio bytes
GET  /status              — assistant status flags
"""

from __future__ import annotations

import asyncio
import io
import json
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from core.database import get_db
from core.events import EventType, event_bus
from core.logger import get_logger
from memory.models import Conversation, ConversationMessage, MessageRole

log = get_logger(__name__)

router = APIRouter(prefix="/api/commands", tags=["commands"])

# --------------------------------------------------------------------------- #
# In-memory state flags (replaced by a real state machine in production)        #
# --------------------------------------------------------------------------- #

_assistant_state: dict[str, bool] = {
    "listening": False,
    "speaking": False,
    "active": True,
}


# --------------------------------------------------------------------------- #
# Pydantic schemas                                                               #
# --------------------------------------------------------------------------- #


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8192)
    conversation_id: str | None = Field(
        None, description="Existing conversation ID.  Omit to start a new one."
    )


class ChatResponse(BaseModel):
    response: str
    actions: list[dict[str, Any]] = Field(default_factory=list)
    conversation_id: str
    tokens: int


class TranscriptionResponse(BaseModel):
    transcription: str
    duration_seconds: float | None = None
    language: str | None = None


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)
    voice: str | None = Field(None, description="Override TTS voice")
    speed: str | None = Field(None, description="Override playback speed")


class AssistantStatus(BaseModel):
    listening: bool
    speaking: bool
    active: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# --------------------------------------------------------------------------- #
# Helpers                                                                        #
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = (
    "You are JARVIS, a highly capable personal AI desktop assistant. "
    "You are helpful, concise, and proactive. "
    "When the user asks you to perform actions (open apps, manage files, "
    "send messages, create tasks) you describe the action clearly. "
    "Always respond in plain text unless asked for code or structured output."
)


async def _get_or_create_conversation(
    conversation_id: str | None, db: AsyncSession
) -> Conversation:
    """Retrieve an existing conversation or create a fresh one."""
    if conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.is_active.is_(True),
            )
        )
        conv = result.scalar_one_or_none()
        if conv:
            return conv
        # Treat an unknown conversation_id as a new conversation start.
        log.warning(
            "Conversation {id!r} not found — starting new one", id=conversation_id
        )

    conv_id = str(uuid.uuid4())
    conv = Conversation(id=conv_id, is_active=True)
    db.add(conv)
    await db.flush()
    return conv


async def _load_history(
    conversation_id: str, db: AsyncSession
) -> list[dict[str, str]]:
    """Return the recent message history for the AI context window."""
    result = await db.execute(
        select(ConversationMessage)
        .where(ConversationMessage.conversation_id == conversation_id)
        .order_by(ConversationMessage.created_at.desc())
        .limit(settings.MAX_CONVERSATION_HISTORY)
    )
    messages = list(reversed(result.scalars().all()))
    return [{"role": m.role.value, "content": m.content} for m in messages]


async def _save_message(
    conversation_id: str,
    role: MessageRole,
    content: str,
    tokens: int | None,
    db: AsyncSession,
) -> ConversationMessage:
    msg = ConversationMessage(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tokens_used=tokens,
    )
    db.add(msg)
    await db.flush()
    return msg


async def _call_openai(
    history: list[dict[str, str]], user_message: str
) -> tuple[str, int]:
    """Call the OpenAI Chat Completions API.

    Returns (response_text, total_tokens).
    Falls back gracefully when the API key is missing.
    """
    if not settings.OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY not set — returning mock response")
        return (
            f"[JARVIS mock] You said: {user_message!r}. "
            "Set OPENAI_API_KEY to enable the real AI.",
            0,
        )

    from openai import AsyncOpenAI  # lazy import

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    resp = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=settings.OPENAI_MAX_TOKENS,
        temperature=settings.OPENAI_TEMPERATURE,
    )

    text = resp.choices[0].message.content or ""
    tokens = resp.usage.total_tokens if resp.usage else 0
    return text, tokens


async def _stream_openai(
    history: list[dict[str, str]], user_message: str
) -> AsyncGenerator[str, None]:
    """Stream tokens from OpenAI using SSE format."""
    if not settings.OPENAI_API_KEY:
        mock = (
            f"[JARVIS mock] You said: {user_message!r}. "
            "Set OPENAI_API_KEY to enable the real AI."
        )
        for word in mock.split():
            yield f"data: {json.dumps({'delta': word + ' '})}\n\n"
            await asyncio.sleep(0.05)
        yield "data: [DONE]\n\n"
        return

    from openai import AsyncOpenAI  # lazy import

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    stream = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=settings.OPENAI_MAX_TOKENS,
        temperature=settings.OPENAI_TEMPERATURE,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield f"data: {json.dumps({'delta': delta})}\n\n"

    yield "data: [DONE]\n\n"


# --------------------------------------------------------------------------- #
# POST /api/commands/chat                                                        #
# --------------------------------------------------------------------------- #


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to JARVIS",
)
async def chat(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Send a user message and receive the full AI response."""
    conv = await _get_or_create_conversation(payload.conversation_id, db)

    history = await _load_history(conv.id, db)
    response_text, tokens = await _call_openai(history, payload.message)

    # Persist both turns.
    await _save_message(conv.id, MessageRole.USER, payload.message, None, db)
    await _save_message(conv.id, MessageRole.ASSISTANT, response_text, tokens, db)

    log.info(
        "Chat: conversation={cid} tokens={t}", cid=conv.id, t=tokens
    )

    await event_bus.publish(
        EventType.AI_RESPONSE_READY,
        {"conversation_id": conv.id, "response": response_text, "tokens": tokens},
    )

    return ChatResponse(
        response=response_text,
        actions=[],
        conversation_id=conv.id,
        tokens=tokens,
    )


# --------------------------------------------------------------------------- #
# POST /api/commands/chat/stream                                                 #
# --------------------------------------------------------------------------- #


@router.post(
    "/chat/stream",
    summary="Streaming chat with JARVIS (SSE)",
    response_class=StreamingResponse,
)
async def chat_stream(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream AI tokens as Server-Sent Events.

    Each event has the shape: ``data: {"delta": "<token>"}``
    The final event is: ``data: [DONE]``
    """
    conv = await _get_or_create_conversation(payload.conversation_id, db)
    history = await _load_history(conv.id, db)

    # Save user message immediately so it's in the DB.
    await _save_message(conv.id, MessageRole.USER, payload.message, None, db)
    await db.commit()

    async def event_generator() -> AsyncGenerator[str, None]:
        # Send conversation_id as first event so the client can reference it.
        yield f"data: {json.dumps({'conversation_id': conv.id})}\n\n"

        full_response: list[str] = []
        async for chunk in _stream_openai(history, payload.message):
            full_response.append(chunk)
            yield chunk

        # Persist assistant message after streaming completes.
        assistant_text = "".join(
            json.loads(c.removeprefix("data: ").strip()).get("delta", "")
            for c in full_response
            if c.startswith("data: ") and "[DONE]" not in c
        )
        try:
            async with db.begin_nested():
                await _save_message(
                    conv.id, MessageRole.ASSISTANT, assistant_text, None, db
                )
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to persist streamed assistant message: {exc}", exc=exc)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------------------------------- #
# POST /api/commands/voice/transcribe                                            #
# --------------------------------------------------------------------------- #


@router.post(
    "/voice/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe audio to text",
)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file (wav, mp3, ogg, webm)"),
) -> Any:
    """Transcribe an uploaded audio file using OpenAI Whisper."""
    if not settings.OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY not set — returning mock transcription")
        return TranscriptionResponse(
            transcription="[mock] Set OPENAI_API_KEY to enable transcription.",
            duration_seconds=None,
            language="en",
        )

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty",
        )

    from openai import AsyncOpenAI  # lazy import

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    filename = audio.filename or "audio.wav"
    content_type = audio.content_type or "audio/wav"

    resp = await client.audio.transcriptions.create(
        model="whisper-1",
        file=(filename, io.BytesIO(audio_bytes), content_type),
        language="en",
    )

    log.info("Transcription: {text!r}", text=resp.text[:80])

    await event_bus.publish(
        EventType.VOICE_COMMAND_RECEIVED, {"transcription": resp.text}
    )

    return TranscriptionResponse(
        transcription=resp.text,
        duration_seconds=None,
        language="en",
    )


# --------------------------------------------------------------------------- #
# POST /api/commands/voice/speak                                                 #
# --------------------------------------------------------------------------- #


@router.post(
    "/voice/speak",
    summary="Text to speech",
    response_class=Response,
    responses={200: {"content": {"audio/mpeg": {}}}},
)
async def speak(payload: SpeakRequest) -> Response:
    """Convert text to speech and return audio/mpeg bytes.

    Uses OpenAI TTS when the API key is available, otherwise returns a
    structured error in the response body.
    """
    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY not configured — TTS unavailable",
        )

    from openai import AsyncOpenAI  # lazy import

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # Map edge-tts voice names to OpenAI voice IDs where possible.
    voice_map = {
        "en-US-GuyNeural": "onyx",
        "en-US-AriaNeural": "nova",
        "en-GB-RyanNeural": "echo",
    }
    tts_voice = payload.voice or settings.TTS_VOICE
    openai_voice = voice_map.get(tts_voice, "onyx")

    resp = await client.audio.speech.create(
        model="tts-1",
        voice=openai_voice,  # type: ignore[arg-type]
        input=payload.text,
        response_format="mp3",
        speed=1.0,  # OpenAI accepts 0.25–4.0; ignore edge-tts "+0%" syntax
    )

    audio_bytes = resp.content

    log.info("TTS generated: {n} bytes for {chars} chars", n=len(audio_bytes), chars=len(payload.text))

    return Response(content=audio_bytes, media_type="audio/mpeg")


# --------------------------------------------------------------------------- #
# GET /api/commands/status                                                       #
# --------------------------------------------------------------------------- #


@router.get(
    "/status",
    response_model=AssistantStatus,
    summary="Assistant status",
)
async def get_status() -> AssistantStatus:
    """Return current listening/speaking/active flags for the assistant."""
    return AssistantStatus(**_assistant_state)
