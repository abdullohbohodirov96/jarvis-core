"""
Celery tasks for report and summary generation.

Tasks:
- generate_weekly_report:         weekly productivity digest
- summarize_long_conversation:    summarise an oversized conversation
- send_morning_briefing:          morning summary with tasks + context
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
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


async def _get_all_user_ids() -> list[str]:
    try:
        from sqlalchemy import select
        from app.database.connection import AsyncSessionLocal
        from app.models.user import User

        async with AsyncSessionLocal() as session:
            stmt = select(User.id).where(
                User.is_active.is_(True),
                User.is_deleted.is_(False),
            )
            result = await session.execute(stmt)
            return [str(row[0]) for row in result.fetchall()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch user IDs: %s", exc)
        return []


# ---------------------------------------------------------------------------
# generate_weekly_report
# ---------------------------------------------------------------------------


@celery_app.task(
    name="backend.app.workers.tasks.summary_tasks.generate_weekly_report",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    acks_late=True,
)
def generate_weekly_report(
    self: Any,
    user_id: str = "all",
) -> dict[str, Any]:
    """
    Generate a weekly productivity report for one or all users.

    Includes:
    - Tasks completed vs. created this week
    - Overdue tasks carried forward
    - AI-written narrative summary

    Args:
        user_id: Specific user UUID or "all" to run for every active user.
    """
    logger.info("generate_weekly_report: user_id=%s", user_id)
    try:
        return _run_async(_async_generate_weekly_report(user_id))
    except Exception as exc:
        logger.error(
            "generate_weekly_report failed for user %s: %s",
            user_id,
            exc,
            exc_info=True,
        )
        raise self.retry(exc=exc) from exc


async def _async_generate_weekly_report(user_id: str) -> dict[str, Any]:
    user_ids = (
        await _get_all_user_ids() if user_id == "all" else [user_id]
    )
    results: dict[str, Any] = {}

    for uid in user_ids:
        try:
            report = await _build_weekly_report_for_user(uid)
            results[uid] = {"generated": True, "report_length": len(report)}
            # Deliver via Telegram if possible
            try:
                from app.workers.tasks.telegram_tasks import (
                    _dispatch_telegram_message,
                )
                await _dispatch_telegram_message(uid, None, report)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not deliver weekly report via Telegram: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Weekly report failed for user %s: %s", uid, exc)
            results[uid] = {"generated": False, "error": str(exc)}

    return results


async def _build_weekly_report_for_user(user_id: str) -> str:
    """Build the full weekly report string for a single user."""
    now = datetime.now(timezone.utc)
    week_start = now - timedelta(days=7)
    week_label = f"{week_start.strftime('%b %d')} – {now.strftime('%b %d, %Y')}"

    lines: list[str] = [f"Weekly Report: {week_label}"]

    # Tasks stats
    try:
        from sqlalchemy import select, and_, func
        from app.database.connection import AsyncSessionLocal
        from app.models.task import Task  # type: ignore[import]
        import uuid

        async with AsyncSessionLocal() as session:
            # Completed this week
            completed_stmt = select(func.count(Task.id)).where(
                and_(
                    Task.user_id == uuid.UUID(user_id),
                    Task.status == "completed",  # type: ignore[attr-defined]
                    Task.updated_at >= week_start,
                    Task.is_deleted.is_(False),
                )
            )
            completed_res = await session.execute(completed_stmt)
            completed_count = completed_res.scalar() or 0

            # Created this week
            created_stmt = select(func.count(Task.id)).where(
                and_(
                    Task.user_id == uuid.UUID(user_id),
                    Task.created_at >= week_start,
                    Task.is_deleted.is_(False),
                )
            )
            created_res = await session.execute(created_stmt)
            created_count = created_res.scalar() or 0

            # Still overdue
            overdue_stmt = select(func.count(Task.id)).where(
                and_(
                    Task.user_id == uuid.UUID(user_id),
                    Task.status == "overdue",  # type: ignore[attr-defined]
                    Task.is_deleted.is_(False),
                )
            )
            overdue_res = await session.execute(overdue_stmt)
            overdue_count = overdue_res.scalar() or 0

            lines.append(
                f"\nTask Summary:"
                f"\n  Completed: {completed_count}"
                f"\n  Created:   {created_count}"
                f"\n  Overdue:   {overdue_count}"
            )

    except ImportError:
        lines.append("\nTask data unavailable.")

    # AI narrative
    try:
        from app.ai.client import get_openai_client

        ai = get_openai_client()
        context = "\n".join(lines)
        narrative_prompt = (
            "Write a brief 2-3 sentence encouraging productivity narrative "
            "based on the following weekly stats for a personal assistant app:\n\n"
            + context
        )
        narrative = await ai.chat_completion(
            messages=[{"role": "user", "content": narrative_prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        lines.append(f"\n{narrative}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("AI narrative generation failed: %s", exc)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# summarize_long_conversation
# ---------------------------------------------------------------------------


@celery_app.task(
    name="backend.app.workers.tasks.summary_tasks.summarize_long_conversation",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def summarize_long_conversation(
    self: Any,
    conversation_id: str,
) -> dict[str, Any]:
    """
    Summarise a conversation that has grown too large for the context window.

    Writes the summary back to Conversation.summary and retains only the
    most recent 10 messages in active context.
    """
    logger.info(
        "summarize_long_conversation: conversation_id=%s", conversation_id
    )
    try:
        return _run_async(_async_summarize_long_conversation(conversation_id))
    except Exception as exc:
        logger.error(
            "summarize_long_conversation failed: %s", exc, exc_info=True
        )
        raise self.retry(exc=exc) from exc


async def _async_summarize_long_conversation(
    conversation_id: str,
) -> dict[str, Any]:
    try:
        from sqlalchemy import select
        from app.database.connection import AsyncSessionLocal
        from app.models.conversation import Conversation
        from app.models.message import Message
        from app.ai.client import get_openai_client
        import uuid

        async with AsyncSessionLocal() as session:
            conv_stmt = select(Conversation).where(
                Conversation.id == uuid.UUID(conversation_id)
            )
            conv_res = await session.execute(conv_stmt)
            conversation = conv_res.scalar_one_or_none()

            if conversation is None:
                return {"summarized": False, "reason": "not_found"}

            msg_stmt = (
                select(Message)
                .where(Message.conversation_id == uuid.UUID(conversation_id))
                .order_by(Message.created_at)
            )
            msg_res = await session.execute(msg_stmt)
            messages = msg_res.scalars().all()

            if len(messages) < 15:
                return {
                    "summarized": False,
                    "reason": "too_short",
                    "message_count": len(messages),
                }

            # Summarise all but the last 10 messages
            to_summarise = messages[:-10]
            transcript = "\n".join(
                f"{m.role.value.upper()}: {m.content}" for m in to_summarise
            )

            ai = get_openai_client()
            prompt = (
                "Summarise the following conversation history into a concise "
                "paragraph that captures all important facts, decisions, and "
                "context needed to continue the conversation.\n\n" + transcript
            )

            summary = await ai.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            )

            conversation.summary = str(summary)
            await session.commit()

        logger.info(
            "summarize_long_conversation done: %d messages summarised.",
            len(to_summarise),
        )
        return {
            "summarized": True,
            "conversation_id": conversation_id,
            "messages_summarized": len(to_summarise),
        }

    except ImportError as exc:
        logger.warning("summarize_long_conversation skipped: %s", exc)
        return {"summarized": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# send_morning_briefing
# ---------------------------------------------------------------------------


@celery_app.task(
    name="backend.app.workers.tasks.summary_tasks.send_morning_briefing",
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    acks_late=True,
)
def send_morning_briefing(
    self: Any,
    user_id: str = "all",
) -> dict[str, Any]:
    """
    Send a morning briefing to one or all active users at 8 AM UTC.

    Briefing includes:
    - Tasks due today
    - High-priority pending tasks
    - Recent memories / context highlights
    - Weather placeholder (requires external API in production)
    """
    logger.info("send_morning_briefing: user_id=%s", user_id)
    try:
        return _run_async(_async_send_morning_briefing(user_id))
    except Exception as exc:
        logger.error("send_morning_briefing failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc) from exc


async def _async_send_morning_briefing(user_id: str) -> dict[str, Any]:
    user_ids = (
        await _get_all_user_ids() if user_id == "all" else [user_id]
    )
    sent = 0
    errors = 0

    for uid in user_ids:
        try:
            briefing = await _build_morning_briefing_for_user(uid)
            try:
                from app.workers.tasks.telegram_tasks import (
                    _dispatch_telegram_message,
                )
                await _dispatch_telegram_message(uid, None, briefing)
                sent += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not deliver briefing via Telegram: %s", exc)
                sent += 1  # Still count as generated even if delivery failed
        except Exception as exc:  # noqa: BLE001
            logger.error("Morning briefing failed for user %s: %s", uid, exc)
            errors += 1

    logger.info(
        "send_morning_briefing done: sent=%d errors=%d", sent, errors
    )
    return {"sent": sent, "errors": errors}


async def _build_morning_briefing_for_user(user_id: str) -> str:
    """Build the morning briefing text for a single user."""
    now = datetime.now(timezone.utc)
    today_str = now.strftime("%A, %B %d")
    end_of_day = now.replace(hour=23, minute=59, second=59)

    lines: list[str] = [
        f"Good morning! Here's your briefing for {today_str}.",
    ]

    # Tasks due today
    try:
        from sqlalchemy import select, and_
        from app.database.connection import AsyncSessionLocal
        from app.models.task import Task  # type: ignore[import]
        import uuid

        async with AsyncSessionLocal() as session:
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

            due_today_stmt = (
                select(Task)
                .where(
                    and_(
                        Task.user_id == uuid.UUID(user_id),
                        Task.due_at.between(today_start, end_of_day),  # type: ignore[attr-defined]
                        Task.status.notin_(["completed", "cancelled"]),  # type: ignore[attr-defined]
                        Task.is_deleted.is_(False),
                    )
                )
                .limit(10)
            )
            due_res = await session.execute(due_today_stmt)
            due_tasks = due_res.scalars().all()

            if due_tasks:
                task_lines = "\n".join(f"  • {t.title}" for t in due_tasks)  # type: ignore[attr-defined]
                lines.append(f"\nDue Today ({len(due_tasks)}):\n{task_lines}")
            else:
                lines.append("\nNo tasks due today.")

            # High priority pending
            hp_stmt = (
                select(Task)
                .where(
                    and_(
                        Task.user_id == uuid.UUID(user_id),
                        Task.priority == "high",  # type: ignore[attr-defined]
                        Task.status.notin_(["completed", "cancelled"]),  # type: ignore[attr-defined]
                        Task.is_deleted.is_(False),
                    )
                )
                .limit(5)
            )
            hp_res = await session.execute(hp_stmt)
            hp_tasks = hp_res.scalars().all()

            if hp_tasks:
                hp_lines = "\n".join(f"  • {t.title}" for t in hp_tasks)  # type: ignore[attr-defined]
                lines.append(f"\nHigh Priority:\n{hp_lines}")

    except ImportError:
        lines.append("\nTask data unavailable.")

    # Recent memories
    try:
        from sqlalchemy import select, and_
        from app.database.connection import AsyncSessionLocal
        from app.models.memory import Memory  # type: ignore[import]
        import uuid

        since = now - timedelta(hours=12)
        async with AsyncSessionLocal() as session:
            mem_stmt = (
                select(Memory)
                .where(
                    and_(
                        Memory.user_id == uuid.UUID(user_id),
                        Memory.created_at >= since,
                        Memory.is_deleted.is_(False),
                    )
                )
                .limit(5)
            )
            mem_res = await session.execute(mem_stmt)
            memories = mem_res.scalars().all()

            if memories:
                mem_lines = "\n".join(
                    f"  • {m.content[:100]}" for m in memories  # type: ignore[attr-defined]
                )
                lines.append(f"\nRecent Context:\n{mem_lines}")

    except ImportError:
        pass

    # Weather placeholder
    lines.append(
        "\nWeather: Connect a weather API for live forecast data."
    )

    return "\n".join(lines)
