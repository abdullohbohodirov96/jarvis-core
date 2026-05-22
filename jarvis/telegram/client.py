"""
JARVIS Telegram userbot client.

Wraps Telethon's TelegramClient with a clean async API for:
  - Connecting / disconnecting
  - Listing dialogs and messages
  - Sending / scheduling messages
  - Searching contacts
  - Downloading media
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Optional

from telethon import TelegramClient
from telethon.errors import (
    FloodWaitError,
    SessionPasswordNeededError,
)
from telethon.tl.functions.messages import SendScheduledMessagesRequest
from telethon.tl.types import (
    Channel,
    Chat,
    InputPeerUser,
    Message,
    User,
)

from config.settings import settings
from core.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------


@dataclass
class DialogInfo:
    id: int
    name: str
    type: str  # private / group / channel
    unread_count: int
    last_message: Optional[str]
    last_message_date: Optional[datetime]


@dataclass
class MessageInfo:
    id: int
    sender: str
    text: Optional[str]
    date: datetime
    is_outgoing: bool
    has_media: bool
    media_type: Optional[str]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class JarvisTelegram:
    """Async Telethon userbot client for JARVIS."""

    def __init__(self) -> None:
        self._client: TelegramClient = TelegramClient(
            session=settings.TELEGRAM_SESSION,
            api_id=settings.TELEGRAM_API_ID,
            api_hash=settings.TELEGRAM_API_HASH,
            # Give Telethon its own event loop awareness
            loop=None,
        )
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Return True when the client has an active connection."""
        return self._connected and self._client.is_connected()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Start the Telethon client, signing in interactively if needed."""
        if self.is_connected:
            log.debug("Telegram client already connected — skipping connect()")
            return

        log.info("Connecting Telegram client (session={s})", s=settings.TELEGRAM_SESSION)
        try:
            await self._client.connect()

            if not await self._client.is_user_authorized():
                phone = settings.TELEGRAM_PHONE
                if not phone:
                    raise RuntimeError(
                        "TELEGRAM_PHONE not set — cannot sign in interactively."
                    )
                log.info("Sending sign-in code to {phone}", phone=phone)
                await self._client.send_code_request(phone)

                # In a headless environment the code must come from stdin or
                # be injected externally.  We do a best-effort interactive read.
                try:
                    code = input("Enter Telegram sign-in code: ").strip()
                    try:
                        await self._client.sign_in(phone, code)
                    except SessionPasswordNeededError:
                        password = input("Enter 2FA password: ").strip()
                        await self._client.sign_in(password=password)
                except EOFError:
                    log.warning(
                        "No stdin available for Telegram sign-in — "
                        "session file must already exist."
                    )

            self._connected = True
            me = await self._client.get_me()
            log.info(
                "Telegram connected as {name} (@{username})",
                name=getattr(me, "first_name", "?"),
                username=getattr(me, "username", "?"),
            )

        except FloodWaitError as exc:
            log.error("Telegram FloodWait: retry after {s}s", s=exc.seconds)
            raise
        except Exception as exc:
            log.error("Telegram connect error: {exc}", exc=exc)
            raise

    async def disconnect(self) -> None:
        """Gracefully disconnect the Telethon client."""
        if self._client.is_connected():
            await self._client.disconnect()
            log.info("Telegram client disconnected.")
        self._connected = False

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------

    async def get_dialogs(self, limit: int = 20) -> list[DialogInfo]:
        """Return up to *limit* dialogs (chats, groups, channels)."""
        if not self.is_connected:
            log.warning("get_dialogs called while disconnected")
            return []

        results: list[DialogInfo] = []
        try:
            async for dialog in self._client.iter_dialogs(limit=limit):
                entity = dialog.entity
                if isinstance(entity, User):
                    dtype = "private"
                    name = " ".join(
                        filter(None, [entity.first_name, entity.last_name])
                    ) or entity.username or str(entity.id)
                elif isinstance(entity, Channel) and entity.megagroup:
                    dtype = "group"
                    name = entity.title or str(entity.id)
                elif isinstance(entity, Channel):
                    dtype = "channel"
                    name = entity.title or str(entity.id)
                elif isinstance(entity, Chat):
                    dtype = "group"
                    name = entity.title or str(entity.id)
                else:
                    dtype = "unknown"
                    name = str(getattr(entity, "id", "?"))

                last_msg: Optional[str] = None
                last_msg_date: Optional[datetime] = None
                if dialog.message:
                    last_msg = getattr(dialog.message, "message", None) or ""
                    last_msg_date = getattr(dialog.message, "date", None)

                results.append(
                    DialogInfo(
                        id=dialog.id,
                        name=name,
                        type=dtype,
                        unread_count=dialog.unread_count,
                        last_message=last_msg,
                        last_message_date=last_msg_date,
                    )
                )
        except Exception as exc:
            log.error("get_dialogs error: {exc}", exc=exc)

        return results

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    async def get_messages(
        self, chat_id: int | str, limit: int = 50
    ) -> list[MessageInfo]:
        """Fetch the latest *limit* messages from *chat_id*."""
        if not self.is_connected:
            return []

        results: list[MessageInfo] = []
        try:
            entity = await self._client.get_entity(chat_id)
            async for msg in self._client.iter_messages(entity, limit=limit):
                if not isinstance(msg, Message):
                    continue

                sender_name = "unknown"
                if msg.sender:
                    s = msg.sender
                    if isinstance(s, User):
                        sender_name = (
                            " ".join(filter(None, [s.first_name, s.last_name]))
                            or s.username
                            or str(s.id)
                        )
                    else:
                        sender_name = getattr(s, "title", None) or str(s.id)

                media_type: Optional[str] = None
                has_media = msg.media is not None
                if has_media:
                    media_type = type(msg.media).__name__

                results.append(
                    MessageInfo(
                        id=msg.id,
                        sender=sender_name,
                        text=msg.message,
                        date=msg.date,
                        is_outgoing=msg.out,
                        has_media=has_media,
                        media_type=media_type,
                    )
                )
        except Exception as exc:
            log.error("get_messages error for {chat}: {exc}", chat=chat_id, exc=exc)

        return results

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send_message(self, chat_id: int | str, text: str) -> bool:
        """Send *text* to *chat_id*.  Returns True on success."""
        if not self.is_connected:
            log.warning("send_message: not connected")
            return False

        try:
            entity = await self._client.get_entity(chat_id)
            await self._client.send_message(entity, text)
            log.info("Message sent to {chat}", chat=chat_id)
            return True
        except FloodWaitError as exc:
            log.error("FloodWait: retry after {s}s", s=exc.seconds)
            return False
        except Exception as exc:
            log.error("send_message error: {exc}", exc=exc)
            return False

    async def send_message_by_name(self, contact_name: str, text: str) -> bool:
        """Search dialogs by name and send *text* to the first match."""
        dialogs = await self.get_dialogs(limit=100)
        name_lower = contact_name.lower()
        for dialog in dialogs:
            if name_lower in dialog.name.lower():
                return await self.send_message(dialog.id, text)

        log.warning("send_message_by_name: no dialog matching '{name}'", name=contact_name)
        return False

    async def schedule_message(
        self, chat_id: int | str, text: str, send_at: datetime
    ) -> bool:
        """Send *text* as a Telegram scheduled message at *send_at* (UTC)."""
        if not self.is_connected:
            return False

        try:
            entity = await self._client.get_entity(chat_id)
            await self._client.send_message(entity, text, schedule=send_at)
            log.info(
                "Scheduled message for {chat} at {dt}", chat=chat_id, dt=send_at
            )
            return True
        except Exception as exc:
            log.error("schedule_message error: {exc}", exc=exc)
            return False

    # ------------------------------------------------------------------
    # Unread / mark-read
    # ------------------------------------------------------------------

    async def get_unread_count(self) -> int:
        """Return the total unread message count across all dialogs."""
        dialogs = await self.get_dialogs(limit=200)
        return sum(d.unread_count for d in dialogs)

    async def mark_as_read(self, chat_id: int | str) -> None:
        """Mark all messages in *chat_id* as read."""
        if not self.is_connected:
            return
        try:
            entity = await self._client.get_entity(chat_id)
            await self._client.send_read_acknowledge(entity)
            log.debug("Marked {chat} as read", chat=chat_id)
        except Exception as exc:
            log.error("mark_as_read error: {exc}", exc=exc)

    # ------------------------------------------------------------------
    # Contacts / search
    # ------------------------------------------------------------------

    async def search_contacts(self, query: str) -> list[DialogInfo]:
        """Return dialogs whose name contains *query* (case-insensitive)."""
        dialogs = await self.get_dialogs(limit=200)
        q = query.lower()
        return [d for d in dialogs if q in d.name.lower()]

    async def get_contact_id(self, name: str) -> Optional[int]:
        """Return the Telegram ID of the first dialog matching *name*, or None."""
        matches = await self.search_contacts(name)
        return matches[0].id if matches else None

    # ------------------------------------------------------------------
    # Media download
    # ------------------------------------------------------------------

    async def download_media(self, message_id: int, chat_id: int) -> Optional[bytes]:
        """Download and return the media bytes from a specific message, or None."""
        if not self.is_connected:
            return None
        try:
            entity = await self._client.get_entity(chat_id)
            msg = await self._client.get_messages(entity, ids=message_id)
            if msg is None or not msg.media:
                return None
            data: bytes = await self._client.download_media(msg, file=bytes)
            return data
        except Exception as exc:
            log.error(
                "download_media error (msg={m}, chat={c}): {exc}",
                m=message_id,
                c=chat_id,
                exc=exc,
            )
            return None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_telegram_instance: Optional[JarvisTelegram] = None


def get_telegram() -> JarvisTelegram:
    """Return the process-level JarvisTelegram singleton."""
    global _telegram_instance
    if _telegram_instance is None:
        _telegram_instance = JarvisTelegram()
    return _telegram_instance
