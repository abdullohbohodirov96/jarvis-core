"""
Telegram message scheduler for JARVIS.

TelegramScheduler stores, retrieves, and dispatches scheduled messages.
It is designed to be driven by a Celery periodic task that calls
``process_due_messages()`` on a configurable cadence.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import and_, select, text as sa_text

from backend.app.core.exceptions import NotFoundException, TelegramException

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.app.telegram.client import TelegramClientManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain model (lightweight dataclass; mirrors the DB table)
# ---------------------------------------------------------------------------


class ScheduledMessageStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledMessage:
    """
    In-memory representation of a scheduled Telegram message.

    All DB writes go through raw SQL so the scheduler remains independent
    of any ORM model file that may not exist yet.
    """

    id: UUID = field(default_factory=uuid.uuid4)
    user_id: UUID = field(default_factory=uuid.uuid4)
    chat_id: str = ""
    text: str = ""
    scheduled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: ScheduledMessageStatus = ScheduledMessageStatus.PENDING
    retry_count: int = 0
    last_error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sent_at: datetime | None = None


# ---------------------------------------------------------------------------
# TelegramScheduler
# ---------------------------------------------------------------------------


class TelegramScheduler:
    """
    Manages scheduled Telegram messages: creation, cancellation, retrieval,
    and dispatch.

    Args:
        client:     Connected TelegramClientManager for sending messages.
        db_session: Async SQLAlchemy session.
    """

    _MAX_RETRIES: int = 3

    def __init__(
        self,
        client: TelegramClientManager,
        db_session: AsyncSession,
    ) -> None:
        self._client = client
        self._db = db_session

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def schedule_message(
        self,
        user_id: UUID,
        chat_id: str | int,
        text: str,
        send_at: datetime,
    ) -> ScheduledMessage:
        """
        Persist a new scheduled message.

        Args:
            user_id:  Owner user UUID.
            chat_id:  Telegram chat ID or username.
            text:     Message body to send.
            send_at:  UTC datetime when the message should be sent.

        Returns:
            The newly created ScheduledMessage dataclass.
        """
        if not text.strip():
            raise ValueError("Scheduled message text must not be empty.")

        # Ensure send_at is timezone-aware
        if send_at.tzinfo is None:
            send_at = send_at.replace(tzinfo=timezone.utc)

        msg = ScheduledMessage(
            id=uuid.uuid4(),
            user_id=user_id,
            chat_id=str(chat_id),
            text=text,
            scheduled_at=send_at,
            status=ScheduledMessageStatus.PENDING,
        )

        await self._db.execute(
            sa_text(
                """
                INSERT INTO scheduled_messages
                    (id, user_id, chat_id, text, scheduled_at, status,
                     retry_count, created_at)
                VALUES
                    (:id, :user_id, :chat_id, :text, :scheduled_at, :status,
                     :retry_count, :created_at)
                """
            ),
            {
                "id": str(msg.id),
                "user_id": str(msg.user_id),
                "chat_id": msg.chat_id,
                "text": msg.text,
                "scheduled_at": msg.scheduled_at,
                "status": msg.status.value,
                "retry_count": 0,
                "created_at": msg.created_at,
            },
        )
        await self._db.flush()

        logger.info(
            "schedule_message: id=%s user=%s chat=%s send_at=%s",
            msg.id, user_id, chat_id, send_at.isoformat(),
        )
        return msg

    async def cancel_scheduled(self, message_id: UUID) -> None:
        """
        Cancel a pending scheduled message.

        Args:
            message_id: UUID of the scheduled message to cancel.

        Raises:
            NotFoundException: If the message doesn't exist or isn't pending.
        """
        result = await self._db.execute(
            sa_text(
                """
                UPDATE scheduled_messages
                   SET status = :cancelled
                 WHERE id = :id
                   AND status = :pending
                RETURNING id
                """
            ),
            {
                "id": str(message_id),
                "cancelled": ScheduledMessageStatus.CANCELLED.value,
                "pending": ScheduledMessageStatus.PENDING.value,
            },
        )
        row = result.fetchone()
        if not row:
            raise NotFoundException(
                message=f"Scheduled message {message_id} not found or already processed.",
                code="SCHEDULED_MSG_NOT_FOUND",
            )
        await self._db.flush()
        logger.info("cancel_scheduled: id=%s", message_id)

    async def get_pending_messages(self, user_id: UUID) -> list[ScheduledMessage]:
        """
        Retrieve all pending scheduled messages for a user.

        Args:
            user_id: The owning user's UUID.

        Returns:
            List of ScheduledMessage objects ordered by scheduled_at ascending.
        """
        result = await self._db.execute(
            sa_text(
                """
                SELECT id, user_id, chat_id, text, scheduled_at,
                       status, retry_count, last_error, created_at, sent_at
                  FROM scheduled_messages
                 WHERE user_id = :user_id
                   AND status  = :pending
                 ORDER BY scheduled_at ASC
                """
            ),
            {
                "user_id": str(user_id),
                "pending": ScheduledMessageStatus.PENDING.value,
            },
        )
        rows = result.fetchall()
        return [self._row_to_msg(row) for row in rows]

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def process_due_messages(self) -> dict[str, int]:
        """
        Find all pending messages with ``scheduled_at <= now`` and send them.

        Intended to be called by a Celery periodic task every minute.

        Returns:
            Stats dict: {"sent": int, "failed": int}.
        """
        now = datetime.now(timezone.utc)
        result = await self._db.execute(
            sa_text(
                """
                SELECT id, user_id, chat_id, text, scheduled_at,
                       status, retry_count, last_error, created_at, sent_at
                  FROM scheduled_messages
                 WHERE status      = :pending
                   AND scheduled_at <= :now
                 ORDER BY scheduled_at ASC
                 LIMIT 100
                """
            ),
            {"pending": ScheduledMessageStatus.PENDING.value, "now": now},
        )
        rows = result.fetchall()
        if not rows:
            return {"sent": 0, "failed": 0}

        sent_count = 0
        failed_count = 0

        for row in rows:
            msg = self._row_to_msg(row)
            success = await self.send_scheduled(msg)
            if success:
                sent_count += 1
            else:
                failed_count += 1

        logger.info(
            "process_due_messages: total=%d sent=%d failed=%d",
            len(rows), sent_count, failed_count,
        )
        return {"sent": sent_count, "failed": failed_count}

    async def send_scheduled(self, scheduled_msg: ScheduledMessage) -> bool:
        """
        Attempt to send a single scheduled message via Telegram.

        On success updates status to SENT.  On failure increments retry_count
        and marks as FAILED when retries are exhausted.

        Args:
            scheduled_msg: The ScheduledMessage to dispatch.

        Returns:
            True if the message was sent successfully.
        """
        try:
            success = await self._client.send_message(
                chat_id_or_username=scheduled_msg.chat_id,
                text=scheduled_msg.text,
            )
            if success:
                await self._db.execute(
                    sa_text(
                        """
                        UPDATE scheduled_messages
                           SET status  = :sent,
                               sent_at = :sent_at
                         WHERE id = :id
                        """
                    ),
                    {
                        "id": str(scheduled_msg.id),
                        "sent": ScheduledMessageStatus.SENT.value,
                        "sent_at": datetime.now(timezone.utc),
                    },
                )
                await self._db.flush()
                logger.info(
                    "send_scheduled: sent id=%s chat=%s",
                    scheduled_msg.id, scheduled_msg.chat_id,
                )
                return True
            else:
                raise TelegramException("send_message returned False")

        except Exception as exc:  # noqa: BLE001
            new_retry = scheduled_msg.retry_count + 1
            new_status = (
                ScheduledMessageStatus.FAILED.value
                if new_retry >= self._MAX_RETRIES
                else ScheduledMessageStatus.PENDING.value
            )
            await self._db.execute(
                sa_text(
                    """
                    UPDATE scheduled_messages
                       SET retry_count = :retry_count,
                           last_error  = :last_error,
                           status      = :status
                     WHERE id = :id
                    """
                ),
                {
                    "id": str(scheduled_msg.id),
                    "retry_count": new_retry,
                    "last_error": str(exc)[:500],
                    "status": new_status,
                },
            )
            await self._db.flush()
            logger.warning(
                "send_scheduled failed: id=%s attempt=%d/%d error=%s",
                scheduled_msg.id, new_retry, self._MAX_RETRIES, exc,
            )
            return False

    async def retry_failed(self, max_retries: int = 3) -> int:
        """
        Re-queue failed messages that have not exceeded the retry ceiling.

        Args:
            max_retries: Messages with ``retry_count < max_retries`` will be
                         reset to PENDING so they are picked up next cycle.

        Returns:
            Number of messages re-queued.
        """
        result = await self._db.execute(
            sa_text(
                """
                UPDATE scheduled_messages
                   SET status = :pending
                 WHERE status      = :failed
                   AND retry_count < :max_retries
                RETURNING id
                """
            ),
            {
                "pending": ScheduledMessageStatus.PENDING.value,
                "failed": ScheduledMessageStatus.FAILED.value,
                "max_retries": max_retries,
            },
        )
        rows = result.fetchall()
        count = len(rows)
        if count:
            await self._db.flush()
            logger.info("retry_failed: re-queued=%d messages", count)
        return count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_msg(row: Any) -> ScheduledMessage:
        """Convert a DB row (mapping or tuple) to a ScheduledMessage dataclass."""
        def _val(key: str, idx: int) -> Any:
            try:
                return row[key]
            except (TypeError, KeyError):
                return row[idx]

        return ScheduledMessage(
            id=UUID(str(_val("id", 0))),
            user_id=UUID(str(_val("user_id", 1))),
            chat_id=str(_val("chat_id", 2)),
            text=str(_val("text", 3)),
            scheduled_at=_val("scheduled_at", 4),
            status=ScheduledMessageStatus(_val("status", 5)),
            retry_count=int(_val("retry_count", 6) or 0),
            last_error=_val("last_error", 7),
            created_at=_val("created_at", 8),
            sent_at=_val("sent_at", 9),
        )
