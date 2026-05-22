"""
Telegram control routes for JARVIS.

Prefix: /api/telegram

Endpoints
---------
GET  /status              — connection / session status
GET  /chats               — list recent dialogs
GET  /messages/{chat_id}  — messages from a chat
POST /send                — send a message
POST /schedule            — schedule a message for later
POST /connect             — initiate connection (triggers code send)

Note: The Telethon client is instantiated lazily. When TELEGRAM_API_ID / HASH
are not configured the endpoints return a graceful "not configured" response
instead of raising.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from core.database import get_db
from core.logger import get_logger
from memory.models import ScheduledMessage

log = get_logger(__name__)

router = APIRouter(prefix="/api/telegram", tags=["telegram"])


# --------------------------------------------------------------------------- #
# Pydantic schemas                                                               #
# --------------------------------------------------------------------------- #


class SendMessageRequest(BaseModel):
    chat_id: str | int = Field(..., description="Chat/channel/user identifier")
    message: str = Field(..., min_length=1, max_length=4096)


class ScheduleMessageRequest(BaseModel):
    chat_id: str | int = Field(..., description="Chat/channel/user identifier")
    message: str = Field(..., min_length=1, max_length=4096)
    send_at: datetime = Field(..., description="UTC datetime to send the message")


class ConnectionRequest(BaseModel):
    phone: str | None = Field(
        None,
        description="Phone number (overrides TELEGRAM_PHONE setting)",
    )


class TelegramStatusResponse(BaseModel):
    configured: bool
    connected: bool
    phone: str | None
    session_file: str
    details: str


class ChatInfo(BaseModel):
    id: int
    name: str
    unread_count: int
    last_message: str | None
    is_group: bool
    is_channel: bool


class TelegramMessage(BaseModel):
    id: int
    sender_id: int | None
    sender_name: str | None
    text: str
    date: datetime
    is_outgoing: bool


class SendResponse(BaseModel):
    success: bool
    message_id: int | None
    error: str | None = None


class ScheduleResponse(BaseModel):
    success: bool
    scheduled_id: int | None
    send_at: datetime
    error: str | None = None


# --------------------------------------------------------------------------- #
# Telethon client wrapper                                                        #
# --------------------------------------------------------------------------- #


class TelegramClientWrapper:
    """Thin wrapper around a Telethon TelegramClient with lazy initialization."""

    def __init__(self) -> None:
        self._client: Any | None = None
        self._connected: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def is_configured(self) -> bool:
        return bool(settings.TELEGRAM_API_ID and settings.TELEGRAM_API_HASH)

    @property
    def is_connected(self) -> bool:
        if self._client is None:
            return False
        try:
            return bool(self._client.is_connected())
        except Exception:  # noqa: BLE001
            return False

    async def get_client(self) -> Any:
        """Return the connected TelegramClient, raising if not configured."""
        if not self.is_configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Telegram not configured — set TELEGRAM_API_ID and TELEGRAM_API_HASH",
            )
        async with self._lock:
            if self._client is None or not self.is_connected:
                await self._connect()
        return self._client

    async def _connect(self) -> None:
        try:
            from telethon import TelegramClient  # lazy import  # type: ignore[import-not-found]

            client = TelegramClient(
                settings.TELEGRAM_SESSION,
                settings.TELEGRAM_API_ID,
                settings.TELEGRAM_API_HASH,
            )
            await client.connect()
            self._client = client
            log.info(
                "Telegram client connected (session={s})",
                s=settings.TELEGRAM_SESSION,
            )
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="telethon package is not installed",
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Telegram connect failed: {exc}", exc=exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Telegram connection failed: {exc}",
            )

    async def initiate_auth(self, phone: str) -> dict[str, str]:
        """Send the code to *phone* and return a dict with status info."""
        if not self.is_configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Telegram not configured",
            )
        try:
            from telethon import TelegramClient  # type: ignore[import-not-found]

            async with self._lock:
                if self._client is None:
                    client = TelegramClient(
                        settings.TELEGRAM_SESSION,
                        settings.TELEGRAM_API_ID,
                        settings.TELEGRAM_API_HASH,
                    )
                    await client.connect()
                    self._client = client

            await self._client.send_code_request(phone)
            return {
                "status": "code_sent",
                "phone": phone,
                "message": "Authorization code sent to the phone number.",
            }
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="telethon package is not installed",
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Telegram auth initiation failed: {exc}", exc=exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initiate Telegram auth: {exc}",
            )


# Module-level Telethon wrapper singleton.
_tg = TelegramClientWrapper()


# --------------------------------------------------------------------------- #
# GET /api/telegram/status                                                       #
# --------------------------------------------------------------------------- #


@router.get(
    "/status",
    response_model=TelegramStatusResponse,
    summary="Telegram connection status",
)
async def telegram_status() -> TelegramStatusResponse:
    """Return current Telegram client configuration and connection state."""
    return TelegramStatusResponse(
        configured=_tg.is_configured,
        connected=_tg.is_connected,
        phone=settings.TELEGRAM_PHONE or None,
        session_file=settings.TELEGRAM_SESSION,
        details=(
            "Ready"
            if _tg.is_connected
            else ("Not configured" if not _tg.is_configured else "Disconnected")
        ),
    )


# --------------------------------------------------------------------------- #
# GET /api/telegram/chats                                                        #
# --------------------------------------------------------------------------- #


@router.get(
    "/chats",
    response_model=list[ChatInfo],
    summary="List recent Telegram chats",
)
async def list_chats(
    limit: int = Query(20, ge=1, le=100),
) -> list[ChatInfo]:
    """Return the most recent dialogs from the Telegram account."""
    client = await _tg.get_client()

    try:
        dialogs = await client.get_dialogs(limit=limit)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to fetch Telegram dialogs: {exc}", exc=exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Telegram error: {exc}",
        )

    result: list[ChatInfo] = []
    for dialog in dialogs:
        entity = dialog.entity
        try:
            from telethon.tl.types import Channel, Chat, User  # type: ignore[import-not-found]

            is_group = isinstance(entity, Chat)
            is_channel = isinstance(entity, Channel) and getattr(entity, "broadcast", False)
            name = getattr(entity, "title", None) or (
                f"{getattr(entity, 'first_name', '')} "
                f"{getattr(entity, 'last_name', '')}".strip()
                if isinstance(entity, User)
                else str(dialog.name or "Unknown")
            )
        except ImportError:
            is_group = False
            is_channel = False
            name = str(dialog.name or "Unknown")

        last_msg: str | None = None
        if dialog.message:
            last_msg = getattr(dialog.message, "text", None) or ""
            if len(last_msg) > 100:
                last_msg = last_msg[:97] + "…"

        result.append(
            ChatInfo(
                id=dialog.id,
                name=name,
                unread_count=dialog.unread_count,
                last_message=last_msg,
                is_group=is_group,
                is_channel=is_channel,
            )
        )

    return result


# --------------------------------------------------------------------------- #
# GET /api/telegram/messages/{chat_id}                                           #
# --------------------------------------------------------------------------- #


@router.get(
    "/messages/{chat_id}",
    response_model=list[TelegramMessage],
    summary="Get messages from a chat",
)
async def get_messages(
    chat_id: str,
    limit: int = Query(50, ge=1, le=200),
) -> list[TelegramMessage]:
    """Return the most recent messages from a Telegram chat."""
    client = await _tg.get_client()

    # chat_id may arrive as a string (username) or numeric string.
    entity_key: str | int
    try:
        entity_key = int(chat_id)
    except ValueError:
        entity_key = chat_id

    try:
        messages = await client.get_messages(entity_key, limit=limit)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to fetch messages from chat {c}: {exc}", c=chat_id, exc=exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Telegram error: {exc}",
        )

    result: list[TelegramMessage] = []
    for msg in messages:
        sender_id: int | None = None
        sender_name: str | None = None

        if msg.sender:
            sender_id = getattr(msg.sender, "id", None)
            first = getattr(msg.sender, "first_name", "") or ""
            last = getattr(msg.sender, "last_name", "") or ""
            title = getattr(msg.sender, "title", "") or ""
            sender_name = (f"{first} {last}".strip()) or title or str(sender_id)

        result.append(
            TelegramMessage(
                id=msg.id,
                sender_id=sender_id,
                sender_name=sender_name,
                text=msg.text or "",
                date=msg.date,
                is_outgoing=msg.out,
            )
        )

    return result


# --------------------------------------------------------------------------- #
# POST /api/telegram/send                                                        #
# --------------------------------------------------------------------------- #


@router.post(
    "/send",
    response_model=SendResponse,
    summary="Send a Telegram message",
)
async def send_message(payload: SendMessageRequest) -> SendResponse:
    """Send a message to a Telegram chat, channel, or user."""
    client = await _tg.get_client()

    entity_key: str | int
    if isinstance(payload.chat_id, int):
        entity_key = payload.chat_id
    else:
        try:
            entity_key = int(payload.chat_id)
        except ValueError:
            entity_key = payload.chat_id

    try:
        sent = await client.send_message(entity_key, payload.message)
        log.info(
            "Telegram message sent to {c}: id={id}",
            c=payload.chat_id,
            id=sent.id,
        )
        return SendResponse(success=True, message_id=sent.id)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to send Telegram message: {exc}", exc=exc)
        return SendResponse(success=False, message_id=None, error=str(exc))


# --------------------------------------------------------------------------- #
# POST /api/telegram/schedule                                                    #
# --------------------------------------------------------------------------- #


@router.post(
    "/schedule",
    response_model=ScheduleResponse,
    summary="Schedule a Telegram message",
)
async def schedule_message(
    payload: ScheduleMessageRequest,
    db: AsyncSession = Depends(get_db),
) -> ScheduleResponse:
    """Persist a scheduled message to the database for future delivery."""
    scheduled = ScheduledMessage(
        chat_id=str(payload.chat_id),
        message=payload.message,
        send_at=payload.send_at,
        sent=False,
    )
    db.add(scheduled)
    await db.flush()
    await db.refresh(scheduled)

    log.info(
        "Scheduled Telegram message id={id} for {t}", id=scheduled.id, t=payload.send_at
    )

    return ScheduleResponse(
        success=True,
        scheduled_id=scheduled.id,
        send_at=payload.send_at,
    )


# --------------------------------------------------------------------------- #
# POST /api/telegram/connect                                                     #
# --------------------------------------------------------------------------- #


@router.post(
    "/connect",
    summary="Initiate Telegram connection / send auth code",
)
async def connect_telegram(payload: ConnectionRequest) -> dict[str, str]:
    """Initiate a Telegram session.

    If the account is already authorized the endpoint returns immediately.
    Otherwise it sends a verification code to the phone number.
    """
    phone = payload.phone or settings.TELEGRAM_PHONE
    if not phone:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Phone number required — provide it in the request body or set TELEGRAM_PHONE",
        )

    return await _tg.initiate_auth(phone)
