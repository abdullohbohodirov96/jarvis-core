"""
Telegram message event handler for JARVIS.

Registers Telethon event handlers on the connected client, processes incoming
messages (stores to DB, detects tasks, triggers auto-reply), handles edits
and read-acknowledgements, and runs an async message processing queue.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

from telethon import TelegramClient, errors, events
from telethon.tl.types import Message, PeerChannel, PeerChat, PeerUser

from backend.app.core.exceptions import TelegramException

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.app.ai.client import OpenAIClient
    from backend.app.telegram.auto_reply import TelegramAutoReply
    from backend.app.telegram.client import TelegramClientManager
    from backend.app.telegram.task_detector import TelegramTaskDetector

logger = logging.getLogger(__name__)

# Maximum messages to buffer before dropping the oldest
_QUEUE_MAX_SIZE = 500
# Seconds to sleep between processing queue batches
_QUEUE_POLL_INTERVAL = 0.5


def _extract_chat_id(peer: Any) -> int | None:
    """Resolve a Telethon Peer object to its numeric ID."""
    if isinstance(peer, PeerUser):
        return peer.user_id
    if isinstance(peer, PeerChat):
        return peer.chat_id
    if isinstance(peer, PeerChannel):
        return peer.channel_id
    return None


class TelegramMessageHandler:
    """
    Registers Telethon event handlers and orchestrates incoming message
    processing: DB persistence, task extraction, and AI auto-reply.

    Args:
        client:        The connected TelegramClientManager instance.
        db_session:    An async SQLAlchemy session (request-scoped or shared).
        ai_agent:      Optional AI agent for generating replies (reserved for
                       future expansion; pass None if not available yet).
        task_detector: TelegramTaskDetector instance for extracting tasks.
    """

    def __init__(
        self,
        client: TelegramClientManager,
        db_session: AsyncSession,
        ai_agent: Any | None,
        task_detector: TelegramTaskDetector,
    ) -> None:
        self._client = client
        self._db = db_session
        self._ai_agent = ai_agent
        self._task_detector = task_detector
        # Deque used as a bounded async queue for incoming messages
        self._message_queue: deque[dict[str, Any]] = deque(maxlen=_QUEUE_MAX_SIZE)
        self._queue_lock: asyncio.Lock = asyncio.Lock()
        self._processing: bool = False
        self._handlers_registered: bool = False

    # ------------------------------------------------------------------
    # Handler setup
    # ------------------------------------------------------------------

    async def setup_handlers(self) -> None:
        """
        Register all Telethon event handlers on the raw client.
        Safe to call multiple times — registers only once.
        """
        if self._handlers_registered:
            logger.debug("Handlers already registered; skipping.")
            return

        raw: TelegramClient = self._client.raw

        @raw.on(events.NewMessage())
        async def _new_message_wrapper(event: events.NewMessage.Event) -> None:
            await self.on_new_message(event)

        @raw.on(events.MessageEdited())
        async def _edited_message_wrapper(event: events.MessageEdited.Event) -> None:
            await self.on_edited_message(event)

        @raw.on(events.MessageRead())
        async def _read_ack_wrapper(event: events.MessageRead.Event) -> None:
            await self.on_read_acknowledgement(event)

        self._handlers_registered = True
        logger.info("Telethon event handlers registered.")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_new_message(self, event: events.NewMessage.Event) -> None:
        """
        Handle a new incoming (or outgoing) message.

        Pipeline:
        1. Gate: only process messages that pass ``_should_process_message``.
        2. Serialise to dict and enqueue for background processing.
        3. Immediately attempt task detection and auto-reply if configured.
        """
        try:
            if not self._should_process_message(event):
                return

            msg_dict = self._message_to_dict(event.message)
            logger.debug(
                "on_new_message: chat=%s msg_id=%d", msg_dict.get("chat_id"), msg_dict.get("id")
            )

            # Enqueue for persistent storage
            async with self._queue_lock:
                self._message_queue.append(msg_dict)

            # Fire-and-forget: detect tasks & auto-reply in background
            asyncio.ensure_future(self._handle_message_pipeline(event, msg_dict))

        except Exception as exc:  # noqa: BLE001
            logger.exception("on_new_message unhandled error: %s", exc)

    async def on_edited_message(self, event: events.MessageEdited.Event) -> None:
        """
        Handle a message edit event.  Updates the stored message text if
        it exists in the DB; otherwise stores the edited version as new.
        """
        try:
            if not hasattr(event, "message") or not isinstance(event.message, Message):
                return

            msg_dict = self._message_to_dict(event.message)
            msg_dict["is_edited"] = True
            logger.debug(
                "on_edited_message: chat=%s msg_id=%d",
                msg_dict.get("chat_id"),
                msg_dict.get("id"),
            )

            # Persist edit to DB (best-effort; no crash on failure)
            try:
                await self._upsert_message_in_db(msg_dict)
            except Exception as db_exc:  # noqa: BLE001
                logger.warning("Failed to persist edited message: %s", db_exc)

        except Exception as exc:  # noqa: BLE001
            logger.exception("on_edited_message unhandled error: %s", exc)

    async def on_read_acknowledgement(self, event: events.MessageRead.Event) -> None:
        """
        Handle a read acknowledgement from Telegram.  Logs the event;
        can be extended to update DB read-status columns.
        """
        try:
            chat_id = _extract_chat_id(getattr(event, "peer", None))
            max_id = getattr(event, "max_id", None)
            outbox = getattr(event, "inbox", True)  # True = their messages read by us
            logger.debug(
                "on_read_acknowledgement: chat=%s max_id=%s outbox=%s",
                chat_id, max_id, outbox,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("on_read_acknowledgement error (non-critical): %s", exc)

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------

    async def _handle_message_pipeline(
        self,
        event: events.NewMessage.Event,
        msg_dict: dict[str, Any],
    ) -> None:
        """
        Background pipeline: store → detect tasks → maybe auto-reply.
        Errors in any stage are logged but do not propagate.
        """
        # 1. Persist to DB
        try:
            await self._upsert_message_in_db(msg_dict)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DB persistence failed for msg_id=%s: %s", msg_dict.get("id"), exc)

        # 2. Task detection (only for inbound messages with text)
        text = msg_dict.get("text", "")
        if text and not msg_dict.get("is_out"):
            try:
                tasks = await self._task_detector.detect_single_message(
                    message=text,
                    context=[],
                )
                if tasks:
                    logger.info(
                        "Detected %d task(s) in msg_id=%s", len(tasks), msg_dict.get("id")
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Task detection failed: %s", exc)

        # 3. Auto-reply (handled by TelegramAutoReply; hook point here)
        # This is intentionally left as a hook; TelegramAutoReply.should_reply
        # and generate_reply are invoked by an external orchestrator that wires
        # all services together, keeping the handler decoupled.

    async def _upsert_message_in_db(self, msg_dict: dict[str, Any]) -> None:
        """
        Persist or update a serialised message in the database.

        Uses raw SQL via the SQLAlchemy session's ``execute`` method so that
        the handler remains independent of any specific ORM model definition.
        The insert is a no-op if the message already exists (idempotent).
        """
        from sqlalchemy import text as sa_text

        stmt = sa_text(
            """
            INSERT INTO telegram_messages (
                message_id, chat_id, sender_id, text, date,
                is_out, is_reply, reply_to_msg_id, has_media,
                media_type, is_edited, created_at
            )
            VALUES (
                :message_id, :chat_id, :sender_id, :text, :date,
                :is_out, :is_reply, :reply_to_msg_id, :has_media,
                :media_type, :is_edited, :created_at
            )
            ON CONFLICT (message_id, chat_id) DO UPDATE
                SET text       = EXCLUDED.text,
                    is_edited  = EXCLUDED.is_edited,
                    updated_at = NOW()
            """
        )
        params = {
            "message_id": msg_dict.get("id"),
            "chat_id": msg_dict.get("chat_id"),
            "sender_id": msg_dict.get("sender_id"),
            "text": msg_dict.get("text", ""),
            "date": msg_dict.get("date"),
            "is_out": msg_dict.get("is_out", False),
            "is_reply": msg_dict.get("is_reply", False),
            "reply_to_msg_id": msg_dict.get("reply_to_msg_id"),
            "has_media": msg_dict.get("has_media", False),
            "media_type": msg_dict.get("media_type"),
            "is_edited": msg_dict.get("is_edited", False),
            "created_at": datetime.now(timezone.utc),
        }
        await self._db.execute(stmt, params)
        await self._db.flush()

    # ------------------------------------------------------------------
    # Queue processor
    # ------------------------------------------------------------------

    async def process_message_queue(self) -> None:
        """
        Continuously drain the internal message queue, persisting each
        buffered message to the database.

        Intended to run as a long-lived background task:
        ``asyncio.ensure_future(handler.process_message_queue())``
        """
        self._processing = True
        logger.info("Message queue processor started.")
        while self._processing:
            batch: list[dict[str, Any]] = []
            async with self._queue_lock:
                while self._message_queue:
                    batch.append(self._message_queue.popleft())

            for msg_dict in batch:
                try:
                    await self._upsert_message_in_db(msg_dict)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Queue processor: DB write failed for msg_id=%s: %s",
                        msg_dict.get("id"),
                        exc,
                    )

            await asyncio.sleep(_QUEUE_POLL_INTERVAL)

    def stop_queue_processor(self) -> None:
        """Signal the queue processor loop to stop after its current batch."""
        self._processing = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_process_message(self, event: events.NewMessage.Event) -> bool:
        """
        Return True if the message should be processed.

        Filtering rules:
        - Must be a genuine Message object (not a service/system message).
        - Skip empty messages (no text and no media).
        - Skip messages from bots (to avoid echo loops).
        - Skip channel posts (broadcast channels) to keep focus on 1-1 and groups.
        """
        message: Message | None = getattr(event, "message", None)
        if not isinstance(message, Message):
            return False
        if not message.text and message.media is None:
            return False
        if getattr(message, "post", False):
            # Channel broadcast post — skip
            return False
        return True

    @staticmethod
    def _message_to_dict(message: Message) -> dict[str, Any]:
        """Serialise a Telethon Message to a plain dict."""
        sender_id: int | None = None
        if message.sender_id is not None:
            sender_id = message.sender_id
        elif message.from_id is not None:
            sender_id = _extract_chat_id(message.from_id)

        peer_id: int | None = _extract_chat_id(message.peer_id) if message.peer_id else None

        return {
            "id": message.id,
            "chat_id": peer_id,
            "sender_id": sender_id,
            "text": message.text or "",
            "date": message.date.isoformat() if message.date else None,
            "is_out": bool(message.out),
            "is_reply": bool(message.is_reply),
            "reply_to_msg_id": getattr(message.reply_to, "reply_to_msg_id", None)
            if message.reply_to
            else None,
            "has_media": message.media is not None,
            "media_type": type(message.media).__name__ if message.media else None,
            "is_edited": False,
            "mentioned": bool(getattr(message, "mentioned", False)),
            "grouped_id": message.grouped_id,
            "views": getattr(message, "views", None),
        }
