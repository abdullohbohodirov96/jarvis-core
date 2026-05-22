"""
Celery tasks for AI / memory operations.

Tasks:
- consolidate_user_memories:        run memory consolidation per user
- generate_daily_summary:           build a comprehensive daily digest
- extract_tasks_from_conversation:  post-process a conversation for tasks
- cleanup_expired_memories:         remove expired / low-importance memories
"""

from __future__ import annotations

import asyncio
import json
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
    """Return list of active user ID strings from DB."""
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
# consolidate_user_memories
# ---------------------------------------------------------------------------


@celery_app.task(
    name="backend.app.workers.tasks.ai_tasks.consolidate_user_memories",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    acks_late=True,
)
def consolidate_user_memories(
    self: Any,
    user_id: str = "all",
) -> dict[str, Any]:
    """
    Run memory consolidation for one user or all users.

    Consolidation merges semantically similar short-term memories into
    longer-term summaries and decays importance scores over time.

    Args:
        user_id: Specific user UUID string, or "all" to process all users.
    """
    logger.info("consolidate_user_memories: user_id=%s", user_id)
    try:
        return _run_async(_async_consolidate_user_memories(user_id))
    except Exception as exc:
        logger.error(
            "consolidate_user_memories failed (%s): %s",
            user_id,
            exc,
            exc_info=True,
        )
        raise self.retry(exc=exc) from exc


async def _async_consolidate_user_memories(user_id: str) -> dict[str, Any]:
    user_ids = (
        await _get_all_user_ids() if user_id == "all" else [user_id]
    )

    consolidated = 0
    errors = 0

    for uid in user_ids:
        try:
            await _consolidate_single_user(uid)
            consolidated += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("Memory consolidation failed for user %s: %s", uid, exc)
            errors += 1

    logger.info(
        "consolidate_user_memories done: processed=%d errors=%d",
        consolidated,
        errors,
    )
    return {"processed": consolidated, "errors": errors}


async def _consolidate_single_user(user_id: str) -> None:
    """
    Consolidate memories for a single user.

    Strategy:
    1. Fetch recent short-term memories (last 48 h).
    2. Group by semantic similarity via embeddings.
    3. Summarise each group with GPT.
    4. Write consolidated memories back; mark originals as consolidated.
    """
    try:
        from sqlalchemy import select, and_
        from app.database.connection import AsyncSessionLocal
        from app.models.memory import Memory  # type: ignore[import]
        from app.ai.client import get_openai_client
        import uuid

        since = datetime.now(timezone.utc) - timedelta(hours=48)

        async with AsyncSessionLocal() as session:
            stmt = select(Memory).where(
                and_(
                    Memory.user_id == uuid.UUID(user_id),
                    Memory.created_at >= since,
                    Memory.memory_type == "short_term",  # type: ignore[attr-defined]
                    Memory.is_deleted.is_(False),
                )
            )
            result = await session.execute(stmt)
            memories = result.scalars().all()

            if len(memories) < 5:
                logger.debug(
                    "Not enough memories to consolidate for user %s (%d).",
                    user_id,
                    len(memories),
                )
                return

            # Build text corpus
            corpus = [
                f"- {m.content}" for m in memories  # type: ignore[attr-defined]
            ]
            joined = "\n".join(corpus)

            ai = get_openai_client()
            summary_prompt = (
                "Summarise the following recent memories into concise, "
                "factual bullet points. Merge related items. "
                "Preserve all important details.\n\n" + joined
            )
            summary = await ai.chat_completion(
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=512,
            )

            # Write consolidated memory
            consolidated = Memory(
                user_id=uuid.UUID(user_id),
                memory_type="long_term",
                content=str(summary),
                importance=0.8,
                source="consolidation",
            )
            session.add(consolidated)

            # Mark originals as consolidated
            for m in memories:
                m.memory_type = "consolidated"  # type: ignore[attr-defined]

            await session.commit()
            logger.info(
                "Consolidated %d memories for user %s.", len(memories), user_id
            )

    except ImportError as exc:
        logger.warning("Memory consolidation skipped (missing model): %s", exc)


# ---------------------------------------------------------------------------
# generate_daily_summary
# ---------------------------------------------------------------------------


@celery_app.task(
    name="backend.app.workers.tasks.ai_tasks.generate_daily_summary",
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    acks_late=True,
)
def generate_daily_summary(self: Any, user_id: str) -> str:
    """
    Generate a comprehensive daily summary for *user_id*.

    Includes: completed/pending tasks, recent conversations summary,
    and memory highlights.

    Returns the summary string.
    """
    logger.info("generate_daily_summary: user_id=%s", user_id)
    try:
        return _run_async(_async_generate_daily_summary(user_id))
    except Exception as exc:
        logger.error(
            "generate_daily_summary failed for user %s: %s",
            user_id,
            exc,
            exc_info=True,
        )
        raise self.retry(exc=exc) from exc


async def _async_generate_daily_summary(user_id: str) -> str:
    sections: list[str] = []
    today = datetime.now(timezone.utc).strftime("%A, %B %d %Y")

    sections.append(f"Daily Summary — {today}")

    # Tasks section
    try:
        from sqlalchemy import select, and_
        from app.database.connection import AsyncSessionLocal
        from app.models.task import Task  # type: ignore[import]
        import uuid

        async with AsyncSessionLocal() as session:
            tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
            stmt = select(Task).where(
                and_(
                    Task.user_id == uuid.UUID(user_id),
                    Task.status.notin_(["completed", "cancelled"]),  # type: ignore[attr-defined]
                    Task.is_deleted.is_(False),
                )
            ).limit(20)
            result = await session.execute(stmt)
            tasks = result.scalars().all()

            if tasks:
                task_lines = "\n".join(f"  • {t.title}" for t in tasks)  # type: ignore[attr-defined]
                sections.append(f"\nPending Tasks ({len(tasks)}):\n{task_lines}")
            else:
                sections.append("\nNo pending tasks.")
    except ImportError:
        sections.append("\nTasks unavailable.")

    # Memories / context section
    try:
        from sqlalchemy import select, and_
        from app.database.connection import AsyncSessionLocal
        from app.models.memory import Memory  # type: ignore[import]
        import uuid

        since = datetime.now(timezone.utc) - timedelta(hours=24)
        async with AsyncSessionLocal() as session:
            stmt = (
                select(Memory)
                .where(
                    and_(
                        Memory.user_id == uuid.UUID(user_id),
                        Memory.created_at >= since,
                        Memory.is_deleted.is_(False),
                    )
                )
                .limit(10)
            )
            result = await session.execute(stmt)
            memories = result.scalars().all()

            if memories:
                mem_lines = "\n".join(
                    f"  • {m.content[:120]}" for m in memories  # type: ignore[attr-defined]
                )
                sections.append(f"\nRecent Highlights:\n{mem_lines}")
    except ImportError:
        pass

    summary = "\n".join(sections)

    # Optionally send via Telegram
    try:
        from app.workers.tasks.telegram_tasks import _dispatch_telegram_message
        await _dispatch_telegram_message(user_id, None, summary)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not send daily summary via Telegram: %s", exc)

    logger.info("generate_daily_summary done for user %s.", user_id)
    return summary


# ---------------------------------------------------------------------------
# extract_tasks_from_conversation
# ---------------------------------------------------------------------------


@celery_app.task(
    name="backend.app.workers.tasks.ai_tasks.extract_tasks_from_conversation",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def extract_tasks_from_conversation(
    self: Any,
    conversation_id: str,
) -> dict[str, Any]:
    """
    Post-process a conversation to extract any actionable tasks.

    Runs after a conversation ends to surface commitments made during the
    chat.
    """
    logger.info(
        "extract_tasks_from_conversation: conversation_id=%s",
        conversation_id,
    )
    try:
        return _run_async(_async_extract_tasks_from_conversation(conversation_id))
    except Exception as exc:
        logger.error(
            "extract_tasks_from_conversation failed: %s",
            exc,
            exc_info=True,
        )
        raise self.retry(exc=exc) from exc


async def _async_extract_tasks_from_conversation(
    conversation_id: str,
) -> dict[str, Any]:
    extracted = 0

    try:
        from sqlalchemy import select
        from app.database.connection import AsyncSessionLocal
        from app.models.conversation import Conversation
        from app.models.message import Message, MessageRole
        from app.models.task import Task  # type: ignore[import]
        from app.ai.client import get_openai_client
        import uuid

        async with AsyncSessionLocal() as session:
            # Load conversation + messages
            conv_stmt = select(Conversation).where(
                Conversation.id == uuid.UUID(conversation_id)
            )
            conv_result = await session.execute(conv_stmt)
            conversation = conv_result.scalar_one_or_none()

            if conversation is None:
                logger.warning("Conversation %s not found.", conversation_id)
                return {"extracted": 0, "conversation_id": conversation_id}

            msg_stmt = (
                select(Message)
                .where(Message.conversation_id == uuid.UUID(conversation_id))
                .order_by(Message.created_at)
            )
            msg_result = await session.execute(msg_stmt)
            messages = msg_result.scalars().all()

            if not messages:
                return {"extracted": 0, "conversation_id": conversation_id}

            # Build conversation transcript
            transcript = "\n".join(
                f"{m.role.value.upper()}: {m.content}" for m in messages
            )

            ai = get_openai_client()
            prompt = (
                "Extract all actionable tasks, to-dos, or commitments from "
                "the following conversation. Return JSON: "
                '{"tasks": [{"title": ..., "description": ..., "priority": "high|medium|low"}]}. '
                "Return empty tasks array if none found.\n\n" + transcript
            )

            response = await ai.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            data = json.loads(str(response))
            task_list = data.get("tasks", [])

            for task_data in task_list:
                task = Task(
                    user_id=conversation.user_id,
                    title=task_data.get("title", "Untitled"),
                    description=task_data.get("description", ""),
                    priority=task_data.get("priority", "medium"),
                    source="conversation",
                    source_conversation_id=uuid.UUID(conversation_id),
                )
                session.add(task)
                extracted += 1

            await session.commit()

    except ImportError as exc:
        logger.warning("extract_tasks_from_conversation skipped: %s", exc)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse AI task extraction response: %s", exc)

    logger.info(
        "extract_tasks_from_conversation done: %d tasks extracted.",
        extracted,
    )
    return {"extracted": extracted, "conversation_id": conversation_id}


# ---------------------------------------------------------------------------
# cleanup_expired_memories
# ---------------------------------------------------------------------------


@celery_app.task(
    name="backend.app.workers.tasks.ai_tasks.cleanup_expired_memories",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    acks_late=True,
)
def cleanup_expired_memories(self: Any) -> dict[str, Any]:
    """
    Remove memories that have exceeded the configured decay window or have
    very low importance scores.

    Runs nightly at 00:30 UTC via Beat.
    """
    logger.info("cleanup_expired_memories: starting")
    try:
        return _run_async(_async_cleanup_expired_memories())
    except Exception as exc:
        logger.error("cleanup_expired_memories failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc) from exc


async def _async_cleanup_expired_memories() -> dict[str, Any]:
    deleted = 0

    try:
        from sqlalchemy import select, and_, or_
        from app.database.connection import AsyncSessionLocal
        from app.models.memory import Memory  # type: ignore[import]
        from app.core.config import get_settings

        settings = get_settings()
        decay_cutoff = datetime.now(timezone.utc) - timedelta(
            hours=settings.MEMORY_DECAY_HOURS
        )
        importance_cutoff = 0.1

        async with AsyncSessionLocal() as session:
            stmt = select(Memory).where(
                and_(
                    Memory.is_deleted.is_(False),
                    or_(
                        Memory.created_at <= decay_cutoff,
                        Memory.importance <= importance_cutoff,  # type: ignore[attr-defined]
                    ),
                )
            )
            result = await session.execute(stmt)
            expired = result.scalars().all()

            for mem in expired:
                mem.is_deleted = True
                deleted += 1

            await session.commit()

    except ImportError as exc:
        logger.warning("Memory cleanup skipped (missing model): %s", exc)

    logger.info("cleanup_expired_memories done: deleted=%d", deleted)
    return {"deleted": deleted}
