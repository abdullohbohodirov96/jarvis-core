"""
Celery tasks for Telegram integration.

Tasks:
- process_scheduled_messages: send due scheduled Telegram messages
- sync_telegram_dialogs:      sync user's chats to DB
- detect_tasks_from_chat:     AI-powered task extraction from recent messages
- send_telegram_message:      async message dispatch via Celery
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from celery.utils.log import get_task_logger

from app.workers.celery_app import get_celery_app

celery_app = get_celery_app()
logger = get_task_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_async(coro: Any) -> Any:
    """Run a coroutine from a synchronous Celery task context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# process_scheduled_messages
# ---------------------------------------------------------------------------


@celery_app.task(
    name="backend.app.workers.tasks.telegram_tasks.process_scheduled_messages",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
)
def process_scheduled_messages(self: Any) -> dict[str, Any]:
    """
    Periodic task: find scheduled Telegram messages whose send_at <= now
    and dispatch them.
    """
    logger.info("process_scheduled_messages: starting")
    try:
        return _run_async(_async_process_scheduled_messages())
    except Exception as exc:
        logger.error("process_scheduled_messages failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc) from exc


async def _async_process_scheduled_messages() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    sent = 0
    errors = 0

    try:
        from sqlalchemy import select, and_
        from app.database.connection import AsyncSessionLocal
        from app.models.scheduled_message import ScheduledMessage  # type: ignore[import]

        async with AsyncSessionLocal() as session:
            stmt = select(ScheduledMessage).where(
                and_(
                    ScheduledMessage.send_at <= now,
                    ScheduledMessage.sent.is_(False),
                    ScheduledMessage.is_deleted.is_(False),
                )
            )
            result = await session.execute(stmt)
            messages = result.scalars().all()

            logger.info("Found %d scheduled messages due.", len(messages))

            for msg in messages:
                try:
                    success = await _dispatch_telegram_message(
                        user_id=str(msg.user_id),
                        chat_id=int(msg.chat_id),
                        text=str(msg.text),
                    )
                    if success:
                        msg.sent = True
                        msg.sent_at = now
                        sent += 1
                    else:
                        errors += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to send scheduled message %s: %s",
                        msg.id,
                        exc,
                    )
                    errors += 1

            await session.commit()

    except ImportError:
        logger.warning("ScheduledMessage model not available; skipping.")

    logger.info(
        "process_scheduled_messages done: sent=%d errors=%d",
        sent,
        errors,
    )
    return {"sent": sent, "errors": errors}


# ---------------------------------------------------------------------------
# sync_telegram_dialogs
# ---------------------------------------------------------------------------


@celery_app.task(
    name="backend.app.workers.tasks.telegram_tasks.sync_telegram_dialogs",
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    acks_late=True,
)
def sync_telegram_dialogs(self: Any, user_id: str) -> dict[str, Any]:
    """
    Sync a user's Telegram dialog list to the local DB.

    Fetches all dialogs (chats) via the Telethon client and upserts
    TelegramChat records.
    """
    logger.info("sync_telegram_dialogs: user_id=%s", user_id)
    try:
        return _run_async(_async_sync_telegram_dialogs(user_id))
    except Exception as exc:
        logger.error(
            "sync_telegram_dialogs failed for user %s: %s",
            user_id,
            exc,
            exc_info=True,
        )
        raise self.retry(exc=exc) from exc


async def _async_sync_telegram_dialogs(user_id: str) -> dict[str, Any]:
    synced = 0
    errors = 0

    try:
        # Import Telegram client if available
        from app.telegram import get_telegram_client  # type: ignore[import]
        from sqlalchemy import select
        from app.database.connection import AsyncSessionLocal
        from app.models.telegram_chat import TelegramChat  # type: ignore[import]
        import uuid

        client = await get_telegram_client(user_id)
        dialogs = await client.get_dialogs()

        async with AsyncSessionLocal() as session:
            for dialog in dialogs:
                try:
                    chat_id = int(dialog.id)
                    stmt = select(TelegramChat).where(
                        TelegramChat.chat_id == chat_id,
                        TelegramChat.user_id == uuid.UUID(user_id),
                    )
                    result = await session.execute(stmt)
                    existing = result.scalar_one_or_none()

                    if existing is None:
                        chat = TelegramChat(
                            user_id=uuid.UUID(user_id),
                            chat_id=chat_id,
                            title=dialog.name or str(chat_id),
                            is_group=dialog.is_group,
                            is_channel=dialog.is_channel,
                        )
                        session.add(chat)
                    else:
                        existing.title = dialog.name or str(chat_id)

                    synced += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to sync dialog %s: %s", dialog.id, exc)
                    errors += 1

            await session.commit()

    except ImportError as exc:
        logger.warning("Telegram client not available: %s", exc)
        return {"synced": 0, "errors": 0, "skipped": True}

    logger.info(
        "sync_telegram_dialogs done for user %s: synced=%d errors=%d",
        user_id,
        synced,
        errors,
    )
    return {"synced": synced, "errors": errors}


# ---------------------------------------------------------------------------
# detect_tasks_from_chat
# ---------------------------------------------------------------------------


@celery_app.task(
    name="backend.app.workers.tasks.telegram_tasks.detect_tasks_from_chat",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def detect_tasks_from_chat(
    self: Any,
    user_id: str,
    chat_id: int,
    hours_back: int = 24,
) -> dict[str, Any]:
    """
    Use the AI to detect actionable tasks from recent Telegram messages.

    Scans messages from the last *hours_back* hours in *chat_id*, extracts
    tasks via GPT, and stores them in the tasks table.
    """
    logger.info(
        "detect_tasks_from_chat: user_id=%s chat_id=%d hours_back=%d",
        user_id,
        chat_id,
        hours_back,
    )
    try:
        return _run_async(_async_detect_tasks_from_chat(user_id, chat_id, hours_back))
    except Exception as exc:
        logger.error("detect_tasks_from_chat failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc) from exc


async def _async_detect_tasks_from_chat(
    user_id: str,
    chat_id: int,
    hours_back: int,
) -> dict[str, Any]:
    detected = 0

    try:
        from app.telegram import get_telegram_client  # type: ignore[import]
        from app.ai.client import get_openai_client

        since = datetime.now(timezone.utc) - timedelta(hours=hours_back)

        client = await get_telegram_client(user_id)
        messages = []
        async for msg in client.iter_messages(chat_id, offset_date=since, reverse=True):
            if msg.text:
                messages.append({"sender": str(msg.sender_id), "text": msg.text})

        if not messages:
            return {"detected": 0, "user_id": user_id, "chat_id": chat_id}

        # Build prompt for task extraction
        conversation_text = "\n".join(
            f"[{m['sender']}]: {m['text']}" for m in messages
        )
        prompt = (
            "Extract any actionable tasks or commitments from the following chat "
            "messages. Return a JSON array of objects with keys: title, description, "
            "due_date (ISO format or null), priority (high/medium/low). "
            "If there are no tasks, return an empty array.\n\n"
            f"Messages:\n{conversation_text}"
        )

        ai = get_openai_client()
        response = await ai.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        import json
        data = json.loads(str(response))
        tasks_data = data if isinstance(data, list) else data.get("tasks", [])

        if tasks_data:
            from sqlalchemy import select
            from app.database.connection import AsyncSessionLocal
            from app.models.task import Task  # type: ignore[import]
            import uuid

            async with AsyncSessionLocal() as session:
                for task_info in tasks_data:
                    task = Task(
                        user_id=uuid.UUID(user_id),
                        title=task_info.get("title", "Untitled task"),
                        description=task_info.get("description", ""),
                        source="telegram",
                        source_chat_id=chat_id,
                    )
                    session.add(task)
                    detected += 1
                await session.commit()

    except ImportError as exc:
        logger.warning(
            "detect_tasks_from_chat: missing dependency (%s); skipping.", exc
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("detect_tasks_from_chat error: %s", exc, exc_info=True)
        raise

    logger.info(
        "detect_tasks_from_chat done: user=%s chat=%d detected=%d",
        user_id,
        chat_id,
        detected,
    )
    return {"detected": detected, "user_id": user_id, "chat_id": chat_id}


# ---------------------------------------------------------------------------
# send_telegram_message
# ---------------------------------------------------------------------------


@celery_app.task(
    name="backend.app.workers.tasks.telegram_tasks.send_telegram_message",
    bind=True,
    max_retries=5,
    default_retry_delay=30,
    acks_late=True,
)
def send_telegram_message(
    self: Any,
    user_id: str,
    chat_id: int | None,
    text: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Send a Telegram message for *user_id* to *chat_id* with body *text*.

    Retries up to 5 times on transient failures.
    """
    logger.info(
        "send_telegram_message: user_id=%s chat_id=%s",
        user_id,
        chat_id,
    )
    try:
        return _run_async(_dispatch_telegram_message(user_id, chat_id, text))
    except Exception as exc:
        logger.error(
            "send_telegram_message failed (user=%s chat=%s): %s",
            user_id,
            chat_id,
            exc,
            exc_info=True,
        )
        raise self.retry(exc=exc) from exc


async def _dispatch_telegram_message(
    user_id: str,
    chat_id: int | None,
    text: str,
) -> dict[str, Any]:
    """
    Inner async implementation for sending a Telegram message.

    If the Telegram client is not available, logs the message and returns
    gracefully.
    """
    if not chat_id:
        # Look up the user's own Telegram ID from DB
        try:
            from sqlalchemy import select
            from app.database.connection import AsyncSessionLocal
            from app.models.user import User
            import uuid

            async with AsyncSessionLocal() as session:
                stmt = select(User).where(User.id == uuid.UUID(user_id))
                result = await session.execute(stmt)
                user = result.scalar_one_or_none()
                if user and user.telegram_user_id:
                    chat_id = user.telegram_user_id
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not resolve chat_id for user %s: %s", user_id, exc)

    if not chat_id:
        logger.warning(
            "send_telegram_message: no chat_id for user %s; message logged only: %r",
            user_id,
            text,
        )
        return {"sent": False, "reason": "no_chat_id"}

    try:
        from app.telegram import get_telegram_client  # type: ignore[import]

        client = await get_telegram_client(user_id)
        await client.send_message(int(chat_id), text)
        logger.info(
            "Telegram message sent: user=%s chat=%s chars=%d",
            user_id,
            chat_id,
            len(text),
        )
        return {"sent": True, "chat_id": chat_id}

    except ImportError:
        logger.warning(
            "Telegram client not available; message logged: user=%s chat=%s: %r",
            user_id,
            chat_id,
            text,
        )
        return {"sent": False, "reason": "client_unavailable"}
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Telegram send error: user=%s chat=%s: %s",
            user_id,
            chat_id,
            exc,
        )
        raise
