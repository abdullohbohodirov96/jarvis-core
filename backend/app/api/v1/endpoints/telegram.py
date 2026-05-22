"""
Telegram integration endpoints for JARVIS.

GET  /telegram/chats                   – list recent chats
GET  /telegram/messages/{chat_id}      – get messages from a chat
POST /telegram/send                    – send a message
POST /telegram/schedule                – schedule a future message
GET  /telegram/status                  – Telegram connection status
POST /telegram/connect                 – initiate Telegram connection / auth
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, status
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.exceptions import TelegramException
from app.core.logging_config import get_logger
from app.utils.helpers import generate_id, now_utc

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TelegramChat(BaseModel):
    chat_id: int
    name: str
    type: str  # "private" | "group" | "supergroup" | "channel"
    unread_count: Optional[int] = None
    last_message: Optional[str] = None
    last_message_at: Optional[datetime] = None


class TelegramMessage(BaseModel):
    message_id: int
    chat_id: int
    sender: Optional[str] = None
    text: Optional[str] = None
    date: datetime
    is_outgoing: bool = False
    media_type: Optional[str] = None


class SendMessageRequest(BaseModel):
    chat_id: int = Field(..., description="Target chat / user ID")
    text: str = Field(..., min_length=1, max_length=4096, description="Message text")
    parse_mode: Optional[str] = Field(
        default=None, description="'html' | 'markdown' | None"
    )
    reply_to_message_id: Optional[int] = None


class SendMessageResponse(BaseModel):
    message_id: int
    chat_id: int
    text: str
    sent_at: datetime


class ScheduleMessageRequest(BaseModel):
    chat_id: int
    text: str = Field(..., min_length=1, max_length=4096)
    send_at: datetime = Field(..., description="When to send the message (UTC)")
    parse_mode: Optional[str] = None


class ScheduleMessageResponse(BaseModel):
    schedule_id: str
    chat_id: int
    text: str
    send_at: datetime
    created_at: datetime


class TelegramStatusResponse(BaseModel):
    connected: bool
    phone: Optional[str]
    session_name: str
    detail: Optional[str] = None


class ConnectRequest(BaseModel):
    phone_code: Optional[str] = Field(
        default=None,
        description="Verification code from Telegram SMS (required for second step)",
    )
    password: Optional[str] = Field(
        default=None, description="2FA password if enabled"
    )


class ConnectResponse(BaseModel):
    status: str  # "awaiting_code" | "awaiting_password" | "connected" | "error"
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# In-process scheduled messages store
# ---------------------------------------------------------------------------

_scheduled: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Telethon client helper
# ---------------------------------------------------------------------------


async def _get_telegram_client():  # type: ignore
    """Return a connected Telethon client.

    Raises TelegramException if the client cannot be created or the
    configuration is incomplete.
    """
    if not settings.TELEGRAM_API_ID or not settings.TELEGRAM_API_HASH:
        raise TelegramException(
            message="Telegram is not configured (missing API_ID or API_HASH).",
            code="TELEGRAM_NOT_CONFIGURED",
        )
    try:
        from telethon import TelegramClient  # type: ignore

        client = TelegramClient(
            settings.TELEGRAM_SESSION_NAME,
            settings.TELEGRAM_API_ID,
            settings.TELEGRAM_API_HASH,
        )
        await client.connect()
        return client
    except ImportError:
        raise TelegramException(
            message="Telethon is not installed.",
            code="TELETHON_NOT_INSTALLED",
        )
    except Exception as exc:
        raise TelegramException(
            message="Failed to connect to Telegram.",
            code="TELEGRAM_CONNECT_FAILED",
            details={"error": str(exc)},
        ) from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/status",
    response_model=TelegramStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Telegram connection status",
)
async def telegram_status() -> TelegramStatusResponse:
    """Report whether JARVIS is currently connected to Telegram."""
    if not settings.TELEGRAM_API_ID or not settings.TELEGRAM_API_HASH:
        return TelegramStatusResponse(
            connected=False,
            phone=settings.TELEGRAM_PHONE or None,
            session_name=settings.TELEGRAM_SESSION_NAME,
            detail="Not configured — set TELEGRAM_API_ID and TELEGRAM_API_HASH.",
        )

    try:
        from telethon import TelegramClient  # type: ignore

        client = TelegramClient(
            settings.TELEGRAM_SESSION_NAME,
            settings.TELEGRAM_API_ID,
            settings.TELEGRAM_API_HASH,
        )
        await client.connect()
        authorised = await client.is_user_authorized()
        await client.disconnect()
        return TelegramStatusResponse(
            connected=authorised,
            phone=settings.TELEGRAM_PHONE or None,
            session_name=settings.TELEGRAM_SESSION_NAME,
            detail="Authorised." if authorised else "Session exists but not authorised.",
        )
    except ImportError:
        return TelegramStatusResponse(
            connected=False,
            phone=settings.TELEGRAM_PHONE or None,
            session_name=settings.TELEGRAM_SESSION_NAME,
            detail="Telethon not installed.",
        )
    except Exception as exc:
        return TelegramStatusResponse(
            connected=False,
            phone=settings.TELEGRAM_PHONE or None,
            session_name=settings.TELEGRAM_SESSION_NAME,
            detail=str(exc),
        )


@router.post(
    "/connect",
    response_model=ConnectResponse,
    status_code=status.HTTP_200_OK,
    summary="Initiate or complete Telegram authentication",
)
async def connect_telegram(payload: ConnectRequest) -> ConnectResponse:
    """Two-step Telegram authentication flow.

    1. First call (no ``phone_code``): sends SMS code to the configured phone number.
    2. Second call (with ``phone_code``): completes sign-in.
    """
    if not settings.TELEGRAM_API_ID or not settings.TELEGRAM_API_HASH:
        raise TelegramException(
            message="Telegram API credentials are not configured.",
            code="TELEGRAM_NOT_CONFIGURED",
        )

    try:
        from telethon import TelegramClient  # type: ignore
        from telethon.errors import SessionPasswordNeededError  # type: ignore

        client = TelegramClient(
            settings.TELEGRAM_SESSION_NAME,
            settings.TELEGRAM_API_ID,
            settings.TELEGRAM_API_HASH,
        )
        await client.connect()

        if await client.is_user_authorized():
            await client.disconnect()
            return ConnectResponse(status="connected", detail="Already authenticated.")

        if payload.phone_code is None:
            # Step 1: request the verification code
            await client.send_code_request(settings.TELEGRAM_PHONE)
            await client.disconnect()
            return ConnectResponse(
                status="awaiting_code",
                detail=f"Verification code sent to {settings.TELEGRAM_PHONE}.",
            )

        # Step 2: sign in with the code
        try:
            await client.sign_in(
                phone=settings.TELEGRAM_PHONE,
                code=payload.phone_code,
            )
        except SessionPasswordNeededError:
            if payload.password:
                await client.sign_in(password=payload.password)
            else:
                await client.disconnect()
                return ConnectResponse(
                    status="awaiting_password",
                    detail="2FA password required.",
                )

        await client.disconnect()
        return ConnectResponse(status="connected", detail="Successfully authenticated.")

    except ImportError:
        raise TelegramException(
            message="Telethon is not installed.",
            code="TELETHON_NOT_INSTALLED",
        )
    except TelegramException:
        raise
    except Exception as exc:
        raise TelegramException(
            message="Authentication failed.",
            code="TELEGRAM_AUTH_FAILED",
            details={"error": str(exc)},
        ) from exc


@router.get(
    "/chats",
    response_model=List[TelegramChat],
    status_code=status.HTTP_200_OK,
    summary="List recent Telegram chats",
)
async def list_chats(limit: int = 20) -> List[TelegramChat]:
    """Return the most recently active chats."""
    try:
        from telethon.tl.types import User, Chat, Channel  # type: ignore

        client = await _get_telegram_client()
        await client.get_dialogs(limit=limit)  # warm the entity cache
        dialogs = await client.get_dialogs(limit=limit)
        await client.disconnect()

        chats: List[TelegramChat] = []
        for dialog in dialogs:
            entity = dialog.entity
            if isinstance(entity, User):
                chat_type = "private"
                name = f"{entity.first_name or ''} {entity.last_name or ''}".strip()
            elif isinstance(entity, Channel):
                chat_type = "channel" if entity.broadcast else "supergroup"
                name = entity.title or str(entity.id)
            elif isinstance(entity, Chat):
                chat_type = "group"
                name = entity.title or str(entity.id)
            else:
                chat_type = "unknown"
                name = str(getattr(entity, "id", "?"))

            last_msg = None
            last_dt = None
            if dialog.message:
                last_msg = (
                    getattr(dialog.message, "message", None) or
                    getattr(dialog.message, "text", None)
                )
                raw_date = getattr(dialog.message, "date", None)
                if raw_date:
                    last_dt = raw_date if raw_date.tzinfo else raw_date.replace(tzinfo=timezone.utc)

            chats.append(
                TelegramChat(
                    chat_id=dialog.id,
                    name=name,
                    type=chat_type,
                    unread_count=dialog.unread_count,
                    last_message=last_msg,
                    last_message_at=last_dt,
                )
            )
        return chats

    except TelegramException:
        raise
    except Exception as exc:
        raise TelegramException(
            message="Failed to list chats.",
            code="TELEGRAM_LIST_CHATS_FAILED",
            details={"error": str(exc)},
        ) from exc


@router.get(
    "/messages/{chat_id}",
    response_model=List[TelegramMessage],
    status_code=status.HTTP_200_OK,
    summary="Get messages from a specific chat",
)
async def get_messages(chat_id: int, limit: int = 50) -> List[TelegramMessage]:
    """Fetch the most recent *limit* messages from *chat_id*."""
    try:
        client = await _get_telegram_client()
        msgs = await client.get_messages(chat_id, limit=limit)
        await client.disconnect()

        result: List[TelegramMessage] = []
        for m in msgs:
            sender_name: Optional[str] = None
            if hasattr(m, "sender") and m.sender:
                s = m.sender
                fn = getattr(s, "first_name", "")
                ln = getattr(s, "last_name", "")
                un = getattr(s, "username", "")
                sender_name = f"{fn} {ln}".strip() or un or str(getattr(s, "id", "?"))

            dt = m.date
            if dt and not dt.tzinfo:
                dt = dt.replace(tzinfo=timezone.utc)

            result.append(
                TelegramMessage(
                    message_id=m.id,
                    chat_id=chat_id,
                    sender=sender_name,
                    text=getattr(m, "message", None) or getattr(m, "text", None),
                    date=dt or now_utc(),
                    is_outgoing=getattr(m, "out", False),
                    media_type=type(m.media).__name__ if m.media else None,
                )
            )
        return result

    except TelegramException:
        raise
    except Exception as exc:
        raise TelegramException(
            message=f"Failed to fetch messages from chat {chat_id}.",
            code="TELEGRAM_GET_MESSAGES_FAILED",
            details={"chat_id": chat_id, "error": str(exc)},
        ) from exc


@router.post(
    "/send",
    response_model=SendMessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a Telegram message",
)
async def send_message(payload: SendMessageRequest) -> SendMessageResponse:
    """Send a text message to a Telegram chat."""
    try:
        client = await _get_telegram_client()
        sent = await client.send_message(
            entity=payload.chat_id,
            message=payload.text,
            parse_mode=payload.parse_mode,
            reply_to=payload.reply_to_message_id,
        )
        await client.disconnect()

        dt = sent.date
        if dt and not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)

        return SendMessageResponse(
            message_id=sent.id,
            chat_id=payload.chat_id,
            text=payload.text,
            sent_at=dt or now_utc(),
        )

    except TelegramException:
        raise
    except Exception as exc:
        raise TelegramException(
            message="Failed to send message.",
            code="TELEGRAM_SEND_FAILED",
            details={"error": str(exc)},
        ) from exc


@router.post(
    "/schedule",
    response_model=ScheduleMessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Schedule a Telegram message",
)
async def schedule_message(
    payload: ScheduleMessageRequest,
    background_tasks: BackgroundTasks,
) -> ScheduleMessageResponse:
    """Schedule a message to be sent at *send_at* (UTC datetime)."""
    now = now_utc()
    send_at = payload.send_at
    if send_at.tzinfo is None:
        send_at = send_at.replace(tzinfo=timezone.utc)

    if send_at <= now:
        raise TelegramException(
            message="send_at must be in the future.",
            code="TELEGRAM_SCHEDULE_IN_PAST",
            details={"send_at": send_at.isoformat(), "now": now.isoformat()},
        )

    schedule_id = generate_id()
    entry: Dict[str, Any] = {
        "schedule_id": schedule_id,
        "chat_id": payload.chat_id,
        "text": payload.text,
        "send_at": send_at.isoformat(),
        "created_at": now.isoformat(),
        "parse_mode": payload.parse_mode,
        "status": "pending",
    }
    _scheduled[schedule_id] = entry

    async def _send_later() -> None:
        delay = (send_at - now_utc()).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)
        try:
            client = await _get_telegram_client()
            await client.send_message(
                entity=payload.chat_id,
                message=payload.text,
                parse_mode=payload.parse_mode,
            )
            await client.disconnect()
            _scheduled[schedule_id]["status"] = "sent"
            logger.info("scheduled_message_sent", schedule_id=schedule_id)
        except Exception as exc:
            _scheduled[schedule_id]["status"] = "failed"
            logger.error("scheduled_message_failed", schedule_id=schedule_id, error=str(exc))

    background_tasks.add_task(_send_later)

    return ScheduleMessageResponse(
        schedule_id=schedule_id,
        chat_id=payload.chat_id,
        text=payload.text,
        send_at=send_at,
        created_at=now,
    )
