"""
JARVIS Telegram message event handler.

Registers Telethon event listeners that:
  - Store incoming messages to the database.
  - Publish TELEGRAM_MESSAGE_RECEIVED events.
  - Optionally generate AI replies for monitored chats.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

from telethon import events
from telethon.tl.types import User

from core.logger import get_logger
from memory.store import TelegramMessageStore
from telegram.client import JarvisTelegram

if TYPE_CHECKING:
    # Imported lazily to avoid circular deps; the assistant module may not
    # exist yet when this module is first imported.
    from assistant.agent import JarvisAgent  # type: ignore[import]

log = get_logger(__name__)

# Internal event-bus placeholder.  Replace with a real pub/sub system when
# the event module is ready — for now we use a simple in-process asyncio
# Queue per registered listener.
_EVENT_TELEGRAM_MESSAGE_RECEIVED = "TELEGRAM_MESSAGE_RECEIVED"


class TelegramHandler:
    """Wire Telethon events into JARVIS business logic."""

    def __init__(
        self,
        client: JarvisTelegram,
        agent: Optional["JarvisAgent"] = None,
    ) -> None:
        self._client = client
        self._agent = agent
        self._monitored_chats: set[int] = set()
        # chat_id -> True means auto-reply is enabled for that chat
        self._auto_reply_chats: set[int] = set()
        # Registered event listeners (for testing / teardown)
        self._msg_store: Optional[TelegramMessageStore] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Register Telethon event handlers on the underlying client."""
        from core.database import AsyncSessionLocal  # local import avoids cycle

        self._msg_store = TelegramMessageStore(AsyncSessionLocal)

        # Telethon requires a raw TelegramClient, so we access the private attr.
        raw = self._client._client  # type: ignore[attr-defined]

        @raw.on(events.NewMessage(incoming=True))
        async def _on_new_message(event: events.NewMessage.Event) -> None:
            await self.on_new_message(event)

        log.info("TelegramHandler: event handlers registered.")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_new_message(self, event: events.NewMessage.Event) -> None:
        """
        Process an incoming Telegram message:

        1. Resolve sender name and chat metadata.
        2. Persist the message to the database.
        3. Publish the TELEGRAM_MESSAGE_RECEIVED event.
        4. Auto-reply if this chat is in the monitored + auto-reply sets.
        """
        msg = event.message
        chat_id: int = event.chat_id

        # --- Resolve sender ---
        sender_name = "unknown"
        try:
            sender = await event.get_sender()
            if isinstance(sender, User):
                sender_name = (
                    " ".join(filter(None, [sender.first_name, sender.last_name]))
                    or sender.username
                    or str(sender.id)
                )
            else:
                sender_name = getattr(sender, "title", None) or str(
                    getattr(sender, "id", "?")
                )
        except Exception as exc:
            log.warning("Could not resolve sender: {exc}", exc=exc)

        # --- Resolve chat name ---
        chat_name: Optional[str] = None
        try:
            chat = await event.get_chat()
            chat_name = (
                getattr(chat, "title", None)
                or getattr(chat, "first_name", None)
                or str(chat_id)
            )
        except Exception:
            chat_name = str(chat_id)

        has_media = msg.media is not None
        text: Optional[str] = msg.message or None

        log.debug(
            "New Telegram message | chat={chat} | from={sender} | text={text!r}",
            chat=chat_id,
            sender=sender_name,
            text=(text[:60] if text else None),
        )

        # --- Persist to DB ---
        if self._msg_store is not None:
            try:
                await self._msg_store.save(
                    telegram_msg_id=msg.id,
                    chat_id=chat_id,
                    chat_name=chat_name,
                    sender=sender_name,
                    text=text,
                    has_media=has_media,
                    is_outgoing=False,
                )
            except Exception as exc:
                log.error("Failed to save TelegramMessage: {exc}", exc=exc)

        # --- Publish event (simple log; swap for real bus as needed) ---
        log.info(
            "{event} | chat={chat} | sender={sender}",
            event=_EVENT_TELEGRAM_MESSAGE_RECEIVED,
            chat=chat_id,
            sender=sender_name,
        )

        # --- Auto-reply for monitored chats ---
        if chat_id in self._monitored_chats and chat_id in self._auto_reply_chats:
            if text:
                asyncio.create_task(self._auto_reply(chat_id, text))

    async def _auto_reply(self, chat_id: int, user_text: str) -> None:
        """Generate an AI reply and send it back to the chat."""
        if self._agent is None:
            log.warning(
                "Auto-reply requested for chat {chat} but no agent set.",
                chat=chat_id,
            )
            return
        try:
            reply = await self.process_command_from_message(user_text, chat_id)
            if reply:
                await self._client.send_message(chat_id, reply)
        except Exception as exc:
            log.error("Auto-reply error for chat {chat}: {exc}", chat=chat_id, exc=exc)

    # ------------------------------------------------------------------
    # Command processing
    # ------------------------------------------------------------------

    async def process_command_from_message(
        self, text: str, chat_id: int
    ) -> str:
        """
        Parse *text* as a JARVIS command (or general prompt) and return the
        agent's reply.  Falls back to a placeholder string when no agent is
        configured.
        """
        if self._agent is None:
            return "JARVIS agent is not configured."

        try:
            # Attempt to call the agent's chat / run method.
            # The exact method name depends on the agent implementation;
            # we try common signatures in order.
            if hasattr(self._agent, "chat"):
                result = await self._agent.chat(text)  # type: ignore[attr-defined]
            elif hasattr(self._agent, "run"):
                result = await self._agent.run(text)  # type: ignore[attr-defined]
            elif hasattr(self._agent, "process"):
                result = await self._agent.process(text)  # type: ignore[attr-defined]
            else:
                return "JARVIS agent does not expose a callable interface."

            if isinstance(result, str):
                return result
            # Some agents return a dict with a "response" / "text" key
            if isinstance(result, dict):
                return result.get("response") or result.get("text") or str(result)
            return str(result)

        except Exception as exc:
            log.error(
                "process_command_from_message error: {exc}", exc=exc
            )
            return f"Error processing command: {exc}"

    # ------------------------------------------------------------------
    # Monitored chat management
    # ------------------------------------------------------------------

    @property
    def monitored_chats(self) -> set[int]:
        return self._monitored_chats

    def add_monitored_chat(self, chat_id: int, auto_reply: bool = False) -> None:
        """Start monitoring *chat_id*.  Optionally enable auto-reply."""
        self._monitored_chats.add(chat_id)
        if auto_reply:
            self._auto_reply_chats.add(chat_id)
        log.info(
            "Monitoring chat {chat} (auto_reply={ar})", chat=chat_id, ar=auto_reply
        )

    def remove_monitored_chat(self, chat_id: int) -> None:
        """Stop monitoring *chat_id*."""
        self._monitored_chats.discard(chat_id)
        self._auto_reply_chats.discard(chat_id)
        log.info("Stopped monitoring chat {chat}", chat=chat_id)

    def enable_auto_reply(self, chat_id: int) -> None:
        """Enable auto-reply for an already-monitored chat."""
        self._auto_reply_chats.add(chat_id)

    def disable_auto_reply(self, chat_id: int) -> None:
        """Disable auto-reply for a chat (monitoring stays active)."""
        self._auto_reply_chats.discard(chat_id)
