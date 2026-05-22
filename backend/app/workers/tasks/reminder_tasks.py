"""
Celery tasks for reminder and task management.

Tasks:
- check_and_send_reminders: periodic scan for due reminders
- send_task_reminder:       send a specific task reminder
- check_overdue_tasks:      detect and notify about overdue tasks
- send_daily_task_summary:  daily digest for all active users
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from celery import shared_task
from celery.utils.log import get_task_logger

from app.workers.celery_app import get_celery_app

celery_app = get_celery_app()
logger = get_task_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from a sync Celery task context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


async def _get_db_session():
    """Create and return an async DB session."""
    from app.database.connection import AsyncSessionLocal
    return AsyncSessionLocal()


# ---------------------------------------------------------------------------
# check_and_send_reminders
# ---------------------------------------------------------------------------


@celery_app.task(
    name="backend.app.workers.tasks.reminder_tasks.check_and_send_reminders",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
)
def check_and_send_reminders(self: Any) -> dict[str, Any]:
    """
    Periodic task: query tasks where reminder_at <= now and not yet sent.

    Sends reminders via Telegram for each due task, then marks them sent.
    Retries up to 3 times on failure.
    """
    logger.info("check_and_send_reminders: starting scan")
    try:
        return _run_async(_async_check_and_send_reminders())
    except Exception as exc:
        logger.error("check_and_send_reminders failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc) from exc


async def _async_check_and_send_reminders() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    sent_count = 0
    errors = 0

    try:
        from sqlalchemy import select, and_
        from app.database.connection import AsyncSessionLocal

        # Lazy import to avoid circular deps
        try:
            from app.models.task import Task  # type: ignore[import]
        except ImportError:
            logger.warning("Task model not available; skipping reminder scan.")
            return {"sent": 0, "errors": 0, "skipped": True}

        async with AsyncSessionLocal() as session:
            stmt = select(Task).where(
                and_(
                    Task.reminder_at <= now,
                    Task.reminder_sent.is_(False),
                    Task.is_deleted.is_(False),
                )
            )
            result = await session.execute(stmt)
            due_tasks = result.scalars().all()

            logger.info("Found %d due reminders.", len(due_tasks))

            for task in due_tasks:
                try:
                    await _send_task_reminder_async(str(task.id), session)
                    task.reminder_sent = True
                    sent_count += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to send reminder for task %s: %s",
                        task.id,
                        exc,
                    )
                    errors += 1

            await session.commit()

    except ImportError:
        logger.warning("DB models not importable; skipping reminder check.")

    logger.info(
        "check_and_send_reminders done: sent=%d errors=%d",
        sent_count,
        errors,
    )
    return {"sent": sent_count, "errors": errors}


# ---------------------------------------------------------------------------
# send_task_reminder
# ---------------------------------------------------------------------------


@celery_app.task(
    name="backend.app.workers.tasks.reminder_tasks.send_task_reminder",
    bind=True,
    max_retries=5,
    default_retry_delay=60,
    acks_late=True,
)
def send_task_reminder(self: Any, task_id: str, **kwargs: Any) -> dict[str, Any]:
    """
    Send a reminder for a specific task by ID.

    Can also accept ``reminder_text`` and ``user_id`` kwargs for voice-command
    originated reminders that don't yet have a DB record.
    """
    logger.info("send_task_reminder: task_id=%s", task_id)
    try:
        return _run_async(_send_task_reminder_standalone(task_id, **kwargs))
    except Exception as exc:
        logger.error("send_task_reminder failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc) from exc


async def _send_task_reminder_standalone(
    task_id: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Send a specific reminder; fall back gracefully if task not found."""
    reminder_text: str = kwargs.get("reminder_text", "")
    user_id: str = kwargs.get("user_id", "")

    if reminder_text and user_id:
        # Voice-command originated reminder (no DB record needed)
        await _notify_user_via_telegram(
            user_id=user_id,
            message=f"Reminder: {reminder_text}",
        )
        return {"sent": True, "task_id": task_id, "text": reminder_text}

    try:
        from sqlalchemy import select
        from app.database.connection import AsyncSessionLocal
        from app.models.task import Task  # type: ignore[import]

        async with AsyncSessionLocal() as session:
            return await _send_task_reminder_async(task_id, session)
    except ImportError:
        logger.warning("Task model unavailable; cannot send reminder %s", task_id)
        return {"sent": False, "task_id": task_id, "error": "Model not available"}


async def _send_task_reminder_async(
    task_id: str,
    session: Any,
) -> dict[str, Any]:
    """Inner async helper used by both check_and_send_reminders and send_task_reminder."""
    from sqlalchemy import select
    import uuid

    try:
        from app.models.task import Task  # type: ignore[import]
        from app.models.user import User

        stmt = select(Task).where(Task.id == uuid.UUID(task_id))
        result = await session.execute(stmt)
        task = result.scalar_one_or_none()

        if task is None:
            logger.warning("Task %s not found.", task_id)
            return {"sent": False, "task_id": task_id, "error": "Not found"}

        # Fetch user Telegram ID
        user_stmt = select(User).where(User.id == task.user_id)
        user_result = await session.execute(user_stmt)
        user = user_result.scalar_one_or_none()

        message = (
            f"Reminder: {task.title}\n"
            f"Due: {task.due_at.strftime('%Y-%m-%d %H:%M UTC') if hasattr(task, 'due_at') and task.due_at else 'N/A'}"
        )

        if user and user.telegram_user_id:
            await _notify_user_via_telegram(
                user_id=str(task.user_id),
                telegram_id=user.telegram_user_id,
                message=message,
            )
        else:
            logger.info(
                "No Telegram linked for user %s; reminder logged only.",
                task.user_id,
            )

        return {"sent": True, "task_id": task_id}

    except Exception as exc:  # noqa: BLE001
        logger.error("_send_task_reminder_async error: %s", exc)
        return {"sent": False, "task_id": task_id, "error": str(exc)}


# ---------------------------------------------------------------------------
# check_overdue_tasks
# ---------------------------------------------------------------------------


@celery_app.task(
    name="backend.app.workers.tasks.reminder_tasks.check_overdue_tasks",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def check_overdue_tasks(self: Any) -> dict[str, Any]:
    """
    Find overdue tasks, update their status to 'overdue', and notify users.
    """
    logger.info("check_overdue_tasks: starting")
    try:
        return _run_async(_async_check_overdue_tasks())
    except Exception as exc:
        logger.error("check_overdue_tasks failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc) from exc


async def _async_check_overdue_tasks() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    updated = 0
    notified = 0

    try:
        from sqlalchemy import select, and_
        from app.database.connection import AsyncSessionLocal
        from app.models.task import Task  # type: ignore[import]
        from app.models.user import User

        async with AsyncSessionLocal() as session:
            # Look for tasks past due date that are still open
            stmt = select(Task).where(
                and_(
                    Task.due_at <= now,
                    Task.status.notin_(["completed", "cancelled", "overdue"]),  # type: ignore[attr-defined]
                    Task.is_deleted.is_(False),
                )
            )
            result = await session.execute(stmt)
            overdue_tasks = result.scalars().all()

            logger.info("Found %d overdue tasks.", len(overdue_tasks))

            for task in overdue_tasks:
                task.status = "overdue"  # type: ignore[attr-defined]
                updated += 1

                user_stmt = select(User).where(User.id == task.user_id)
                user_res = await session.execute(user_stmt)
                user = user_res.scalar_one_or_none()

                if user and user.telegram_user_id:
                    try:
                        await _notify_user_via_telegram(
                            user_id=str(task.user_id),
                            telegram_id=user.telegram_user_id,
                            message=f"Overdue task: {task.title}",
                        )
                        notified += 1
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Notification failed for %s: %s", task.id, exc)

            await session.commit()

    except ImportError:
        logger.warning("Task model unavailable; skipping overdue check.")

    logger.info(
        "check_overdue_tasks done: updated=%d notified=%d",
        updated,
        notified,
    )
    return {"updated": updated, "notified": notified}


# ---------------------------------------------------------------------------
# send_daily_task_summary
# ---------------------------------------------------------------------------


@celery_app.task(
    name="backend.app.workers.tasks.reminder_tasks.send_daily_task_summary",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    acks_late=True,
)
def send_daily_task_summary(self: Any) -> dict[str, Any]:
    """
    Generate and send a daily task summary to all active users.

    Runs at 9 AM UTC via Beat.
    """
    logger.info("send_daily_task_summary: starting")
    try:
        return _run_async(_async_send_daily_task_summary())
    except Exception as exc:
        logger.error("send_daily_task_summary failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc) from exc


async def _async_send_daily_task_summary() -> dict[str, Any]:
    sent = 0
    errors = 0

    try:
        from sqlalchemy import select
        from app.database.connection import AsyncSessionLocal
        from app.models.user import User

        async with AsyncSessionLocal() as session:
            stmt = select(User).where(
                User.is_active.is_(True),
                User.is_deleted.is_(False),
            )
            result = await session.execute(stmt)
            users = result.scalars().all()

            for user in users:
                try:
                    # Delegate per-user summary to AI task
                    celery_app.send_task(
                        "backend.app.workers.tasks.ai_tasks.generate_daily_summary",
                        kwargs={"user_id": str(user.id)},
                    )
                    sent += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to queue summary for user %s: %s",
                        user.id,
                        exc,
                    )
                    errors += 1

    except ImportError:
        logger.warning("User model unavailable; skipping daily summary.")

    logger.info(
        "send_daily_task_summary queued: sent=%d errors=%d",
        sent,
        errors,
    )
    return {"queued": sent, "errors": errors}


# ---------------------------------------------------------------------------
# Internal notification helper
# ---------------------------------------------------------------------------


async def _notify_user_via_telegram(
    user_id: str,
    message: str,
    telegram_id: int | None = None,
) -> None:
    """
    Best-effort Telegram notification.

    Queues a Celery task rather than sending synchronously to avoid blocking
    the reminder scan loop.
    """
    try:
        celery_app.send_task(
            "backend.app.workers.tasks.telegram_tasks.send_telegram_message",
            kwargs={
                "user_id": user_id,
                "chat_id": telegram_id,
                "text": message,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not queue Telegram notification: %s", exc)
