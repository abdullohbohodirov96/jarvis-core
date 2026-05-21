"""
Telethon async client manager for JARVIS.

Provides TelegramClientManager: a high-level wrapper around the Telethon
TelegramClient that handles connection lifecycle, message I/O, dialog listing,
contact search, media download, and more.

A module-level singleton is exposed via ``get_telegram_client()``.
Session files are stored under ``data/sessions/``.
"""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from telethon import TelegramClient, errors, functions, types
from telethon.sessions import SQLiteSession
from telethon.tl.types import (
    Channel,
    Chat,
    Message,
    PeerChannel,
    PeerChat,
    PeerUser,
    User,
)

from backend.app.core.config import get_settings
from backend.app.core.exceptions import TelegramException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session directory
# ---------------------------------------------------------------------------

_SESSION_DIR = Path(__file__).resolve().parents[4] / "data" / "sessions"
_SESSION_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _entity_to_dict(entity: Any) -> dict[str, Any]:
    """Serialise a Telethon entity (User, Chat, Channel) to a plain dict."""
    if isinstance(entity, User):
        return {
            "type": "user",
            "id": entity.id,
            "first_name": entity.first_name or "",
            "last_name": entity.last_name or "",
            "username": entity.username or "",
            "phone": entity.phone or "",
            "is_bot": entity.bot,
            "is_verified": entity.verified,
            "is_restricted": entity.restricted,
        }
    if isinstance(entity, Chat):
        return {
            "type": "group",
            "id": entity.id,
            "title": entity.title or "",
            "participants_count": getattr(entity, "participants_count", 0),
        }
    if isinstance(entity, Channel):
        return {
            "type": "channel" if entity.broadcast else "supergroup",
            "id": entity.id,
            "title": entity.title or "",
            "username": entity.username or "",
            "participants_count": getattr(entity, "participants_count", 0),
            "is_broadcast": entity.broadcast,
            "is_megagroup": entity.megagroup,
        }
    return {"type": "unknown", "id": getattr(entity, "id", None)}


def _message_to_dict(message: Message) -> dict[str, Any]:
    """Serialise a Telethon Message to a plain dict."""
    sender_id: int | None = None
    if message.sender_id is not None:
        sender_id = message.sender_id
    elif message.from_id is not None:
        from_id = message.from_id
        if isinstance(from_id, PeerUser):
            sender_id = from_id.user_id
        elif isinstance(from_id, PeerChat):
            sender_id = from_id.chat_id
        elif isinstance(from_id, PeerChannel):
            sender_id = from_id.channel_id

    peer_id: int | None = None
    if message.peer_id is not None:
        if isinstance(message.peer_id, PeerUser):
            peer_id = message.peer_id.user_id
        elif isinstance(message.peer_id, PeerChat):
            peer_id = message.peer_id.chat_id
        elif isinstance(message.peer_id, PeerChannel):
            peer_id = message.peer_id.channel_id

    return {
        "id": message.id,
        "chat_id": peer_id,
        "sender_id": sender_id,
        "text": message.text or "",
        "date": message.date.isoformat() if message.date else None,
        "is_out": message.out,
        "is_reply": message.is_reply,
        "reply_to_msg_id": getattr(message.reply_to, "reply_to_msg_id", None)
        if message.reply_to
        else None,
        "has_media": message.media is not None,
        "media_type": type(message.media).__name__ if message.media else None,
        "views": getattr(message, "views", None),
        "forwards": getattr(message, "forwards", None),
        "edit_date": message.edit_date.isoformat() if message.edit_date else None,
        "grouped_id": message.grouped_id,
        "mentioned": message.mentioned,
    }


# ---------------------------------------------------------------------------
# TelegramClientManager
# ---------------------------------------------------------------------------


class TelegramClientManager:
    """
    Manages the Telethon TelegramClient lifecycle and exposes high-level
    async helpers for interacting with the Telegram API as a userbot.

    Args:
        session_name: SQLite session file name (without extension).
        api_id:       Telegram API ID from https://my.telegram.org.
        api_hash:     Telegram API hash.
        phone:        Phone number in international format (e.g. "+79001234567").
    """

    def __init__(
        self,
        session_name: str,
        api_id: int,
        api_hash: str,
        phone: str,
    ) -> None:
        self._session_name = session_name
        self._api_id = api_id
        self._api_hash = api_hash
        self._phone = phone

        session_path = str(_SESSION_DIR / session_name)
        self._client: TelegramClient = TelegramClient(
            session_path,
            api_id,
            api_hash,
            # Connection settings tuned for reliability
            connection_retries=5,
            retry_delay=1,
            auto_reconnect=True,
            request_retries=5,
            flood_sleep_threshold=60,
        )
        self._connected: bool = False
        logger.info(
            "TelegramClientManager created: session=%s phone=%s", session_name, phone
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Connect to Telegram.  If not yet authorised the client will ask for a
        code via the interactive flow (handled separately by SessionManager).
        """
        try:
            await self._client.connect()
            if not await self._client.is_user_authorized():
                logger.warning(
                    "Telegram client connected but not yet authorized. "
                    "Call SessionManager.request_code() to complete auth."
                )
            else:
                self._connected = True
                me: User = await self._client.get_me()  # type: ignore[assignment]
                logger.info(
                    "Telegram connected: user=%s id=%s",
                    getattr(me, "first_name", "?"),
                    getattr(me, "id", "?"),
                )
        except errors.RPCError as exc:
            raise TelegramException(
                message=f"Failed to connect to Telegram: {exc}",
                code="TELEGRAM_CONNECT_ERROR",
                details={"rpc_error": str(exc)},
            ) from exc

    async def disconnect(self) -> None:
        """Cleanly disconnect the Telethon client."""
        try:
            await self._client.disconnect()
            self._connected = False
            logger.info("Telegram client disconnected.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error disconnecting Telegram client: %s", exc)

    async def is_connected(self) -> bool:
        """Return True if the client is connected and authorised."""
        try:
            return self._client.is_connected() and await self._client.is_user_authorized()
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    async def send_message(
        self,
        chat_id_or_username: str | int,
        text: str,
        reply_to: int | None = None,
        parse_mode: str | None = "md",
    ) -> bool:
        """
        Send a text message to a chat.

        Args:
            chat_id_or_username: Numeric chat ID, ``@username``, or phone number.
            text:                Message body (Markdown by default).
            reply_to:            Optional message ID to reply to.
            parse_mode:          "md", "html", or None for plain text.

        Returns:
            True on success, False on failure.
        """
        try:
            await self._client.send_message(
                entity=chat_id_or_username,
                message=text,
                reply_to=reply_to,
                parse_mode=parse_mode,
                link_preview=False,
            )
            logger.debug("send_message: chat=%s len=%d", chat_id_or_username, len(text))
            return True
        except errors.FloodWaitError as exc:
            logger.warning(
                "send_message flood wait %ds for chat=%s", exc.seconds, chat_id_or_username
            )
            await asyncio.sleep(exc.seconds)
            # Retry once after flood wait
            try:
                await self._client.send_message(
                    entity=chat_id_or_username,
                    message=text,
                    reply_to=reply_to,
                    parse_mode=parse_mode,
                    link_preview=False,
                )
                return True
            except Exception as retry_exc:
                logger.error("send_message retry failed: %s", retry_exc)
                return False
        except errors.RPCError as exc:
            logger.error("send_message RPC error: %s", exc)
            raise TelegramException(
                message=f"Failed to send message: {exc}",
                code="SEND_MESSAGE_ERROR",
                details={"chat": str(chat_id_or_username), "rpc_error": str(exc)},
            ) from exc

    async def get_messages(
        self,
        chat_id: str | int,
        limit: int = 50,
        offset_id: int = 0,
        min_id: int = 0,
        max_id: int = 0,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve recent messages from a chat.

        Args:
            chat_id:   Numeric chat ID or ``@username``.
            limit:     Maximum number of messages to fetch (default 50).
            offset_id: Fetch messages before this message ID.
            min_id:    Only messages with ID >= min_id.
            max_id:    Only messages with ID <= max_id.
            search:    Optional substring to search within messages.

        Returns:
            List of message dicts (newest-first order from Telethon).
        """
        try:
            messages: list[dict[str, Any]] = []
            async for msg in self._client.iter_messages(
                entity=chat_id,
                limit=limit,
                offset_id=offset_id,
                min_id=min_id,
                max_id=max_id,
                search=search,
            ):
                if isinstance(msg, Message):
                    messages.append(_message_to_dict(msg))
            logger.debug(
                "get_messages: chat=%s fetched=%d", chat_id, len(messages)
            )
            return messages
        except errors.RPCError as exc:
            raise TelegramException(
                message=f"Failed to get messages: {exc}",
                code="GET_MESSAGES_ERROR",
                details={"chat": str(chat_id), "rpc_error": str(exc)},
            ) from exc

    async def get_dialogs(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Retrieve a list of recent dialogs (chats, groups, channels).

        Args:
            limit: Maximum number of dialogs.

        Returns:
            List of dialog dicts with entity info and last-message excerpt.
        """
        try:
            dialogs: list[dict[str, Any]] = []
            async for dialog in self._client.iter_dialogs(limit=limit):
                entity_dict = _entity_to_dict(dialog.entity)
                last_msg_text: str = ""
                if dialog.message and isinstance(dialog.message, Message):
                    last_msg_text = dialog.message.text or ""
                dialogs.append(
                    {
                        **entity_dict,
                        "dialog_id": dialog.id,
                        "name": dialog.name or "",
                        "unread_count": dialog.unread_count,
                        "last_message": last_msg_text[:200],
                        "last_message_date": dialog.message.date.isoformat()
                        if dialog.message and dialog.message.date
                        else None,
                        "pinned": dialog.pinned,
                        "archived": dialog.archived,
                    }
                )
            logger.debug("get_dialogs: fetched=%d", len(dialogs))
            return dialogs
        except errors.RPCError as exc:
            raise TelegramException(
                message=f"Failed to get dialogs: {exc}",
                code="GET_DIALOGS_ERROR",
                details={"rpc_error": str(exc)},
            ) from exc

    async def get_chat_info(self, chat_id: str | int) -> dict[str, Any]:
        """
        Get metadata for a specific chat, group, or channel.

        Args:
            chat_id: Numeric ID, ``@username``, or phone.

        Returns:
            Entity dict with type-specific fields.
        """
        try:
            entity = await self._client.get_entity(chat_id)
            info = _entity_to_dict(entity)
            # Fetch full entity for participant count / description
            try:
                full = await self._client(
                    functions.users.GetFullUserRequest(entity)
                    if isinstance(entity, User)
                    else functions.channels.GetFullChannelRequest(entity)
                    if isinstance(entity, Channel)
                    else functions.messages.GetFullChatRequest(entity.id)
                )
                if hasattr(full, "full_chat"):
                    info["about"] = getattr(full.full_chat, "about", "") or ""
                    info["participants_count"] = getattr(
                        full.full_chat, "participants_count", 0
                    )
                elif hasattr(full, "full_user"):
                    info["bio"] = getattr(full.full_user, "about", "") or ""
            except Exception:  # noqa: BLE001
                pass  # Full info is best-effort
            logger.debug("get_chat_info: chat=%s type=%s", chat_id, info.get("type"))
            return info
        except errors.RPCError as exc:
            raise TelegramException(
                message=f"Failed to get chat info: {exc}",
                code="GET_CHAT_INFO_ERROR",
                details={"chat": str(chat_id), "rpc_error": str(exc)},
            ) from exc

    async def search_contacts(self, query: str) -> list[dict[str, Any]]:
        """
        Search contacts and global users by name or username.

        Args:
            query: Search string.

        Returns:
            List of user dicts matching the query.
        """
        try:
            result = await self._client(
                functions.contacts.SearchRequest(q=query, limit=50)
            )
            contacts: list[dict[str, Any]] = []
            for user in result.users:
                if isinstance(user, User):
                    contacts.append(_entity_to_dict(user))
            logger.debug("search_contacts: query=%r found=%d", query, len(contacts))
            return contacts
        except errors.RPCError as exc:
            raise TelegramException(
                message=f"Failed to search contacts: {exc}",
                code="SEARCH_CONTACTS_ERROR",
                details={"query": query, "rpc_error": str(exc)},
            ) from exc

    async def download_media(self, message: Message) -> bytes | None:
        """
        Download media attached to a message.

        Args:
            message: Telethon Message object with a non-None ``media`` field.

        Returns:
            Raw bytes of the downloaded file, or None if the message has no media.
        """
        if message.media is None:
            return None
        try:
            buf = io.BytesIO()
            await self._client.download_media(message, file=buf)
            buf.seek(0)
            data = buf.read()
            logger.debug(
                "download_media: msg_id=%d size=%d bytes", message.id, len(data)
            )
            return data
        except errors.RPCError as exc:
            raise TelegramException(
                message=f"Failed to download media: {exc}",
                code="DOWNLOAD_MEDIA_ERROR",
                details={"msg_id": message.id, "rpc_error": str(exc)},
            ) from exc

    async def forward_message(
        self,
        from_chat: str | int,
        to_chat: str | int,
        message_id: int,
    ) -> bool:
        """
        Forward a message from one chat to another.

        Args:
            from_chat:  Source chat ID or username.
            to_chat:    Destination chat ID or username.
            message_id: ID of the message to forward.

        Returns:
            True on success.
        """
        try:
            await self._client.forward_messages(
                entity=to_chat,
                messages=message_id,
                from_peer=from_chat,
            )
            logger.debug(
                "forward_message: from=%s to=%s msg_id=%d",
                from_chat, to_chat, message_id,
            )
            return True
        except errors.FloodWaitError as exc:
            await asyncio.sleep(exc.seconds)
            await self._client.forward_messages(
                entity=to_chat, messages=message_id, from_peer=from_chat
            )
            return True
        except errors.RPCError as exc:
            raise TelegramException(
                message=f"Failed to forward message: {exc}",
                code="FORWARD_MESSAGE_ERROR",
                details={
                    "from": str(from_chat),
                    "to": str(to_chat),
                    "msg_id": message_id,
                    "rpc_error": str(exc),
                },
            ) from exc

    async def mark_as_read(self, chat_id: str | int) -> bool:
        """
        Mark all messages in a chat as read.

        Args:
            chat_id: Chat ID or username.

        Returns:
            True on success.
        """
        try:
            await self._client.send_read_acknowledge(chat_id)
            logger.debug("mark_as_read: chat=%s", chat_id)
            return True
        except errors.RPCError as exc:
            logger.warning("mark_as_read failed for %s: %s", chat_id, exc)
            return False

    # ------------------------------------------------------------------
    # Raw client access
    # ------------------------------------------------------------------

    @property
    def raw(self) -> TelegramClient:
        """Expose the underlying Telethon TelegramClient for advanced use."""
        return self._client


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_telegram_client_instance: TelegramClientManager | None = None
_instance_lock: asyncio.Lock = asyncio.Lock()


async def get_telegram_client() -> TelegramClientManager:
    """
    Return (and lazily initialise) the process-level TelegramClientManager.

    Settings are read from the application config.  The singleton is
    created once and reused across all callers within the same process.
    """
    global _telegram_client_instance
    async with _instance_lock:
        if _telegram_client_instance is None:
            settings = get_settings()
            _telegram_client_instance = TelegramClientManager(
                session_name=settings.TELEGRAM_SESSION_NAME,
                api_id=settings.TELEGRAM_API_ID,
                api_hash=settings.TELEGRAM_API_HASH,
                phone=settings.TELEGRAM_PHONE,
            )
            logger.info("TelegramClientManager singleton created.")
    return _telegram_client_instance
