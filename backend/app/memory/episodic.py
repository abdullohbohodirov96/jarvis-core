"""
Episodic memory: short-term event log for a JARVIS user.

Records individual events (conversations, task completions, voice commands,
Telegram messages, etc.) with timestamps and optional metadata.  Supports
full-text search via the AI embedding layer and day-level summarisation.

Classes:
- EpisodicMemory: per-user episodic event manager
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

EPISODE_TYPES = {
    "conversation",
    "task_completed",
    "task_created",
    "telegram_received",
    "telegram_sent",
    "voice_command",
    "reminder_fired",
    "memory_consolidated",
    "system",
}


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------


class EpisodicMemory:
    """
    Async episodic memory store for a single JARVIS user.

    Episodes are persisted to the ``memories`` table (Memory ORM model) with
    ``memory_type = 'episodic'``.  Semantic search delegates to the AI
    embedding layer.

    Args:
        user_id:       UUID string of the owning user.
        db_session:    An open SQLAlchemy AsyncSession.
        max_episodes:  Maximum number of episodes to retain before the oldest
                       are soft-deleted (sliding window).
    """

    def __init__(
        self,
        user_id: str,
        db_session: Any,
        max_episodes: int = 100,
    ) -> None:
        self._user_id = user_id
        self._session = db_session
        self._max_episodes = max_episodes

    # ------------------------------------------------------------------
    # record_episode
    # ------------------------------------------------------------------

    async def record_episode(
        self,
        event_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Record a new episodic memory event.

        Args:
            event_type: Category string from EPISODE_TYPES (or any custom
                        string).
            content:    Human-readable description of the event.
            metadata:   Optional key-value pairs for structured data
                        (e.g. task_id, chat_id, command_type).

        Returns:
            The UUID string of the newly created Memory record.
        """
        import uuid

        try:
            from backend.app.models.memory import Memory  # type: ignore[import]

            episode_meta: dict[str, Any] = {
                "event_type": event_type,
                **(metadata or {}),
            }

            mem = Memory(
                user_id=uuid.UUID(self._user_id),
                memory_type="episodic",
                content=content,
                importance=self._default_importance(event_type),
                source=event_type,
                metadata_=episode_meta,
            )
            self._session.add(mem)
            await self._session.flush()  # Get the id without committing

            logger.debug(
                "EpisodicMemory.record_episode: user=%s type=%s id=%s",
                self._user_id,
                event_type,
                mem.id,
            )

            # Trim oldest episodes if we exceed max_episodes
            await self._trim_if_needed()

            return str(mem.id)

        except ImportError:
            logger.warning(
                "Memory model not available; episode not persisted: %s",
                content,
            )
            return ""
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "EpisodicMemory.record_episode error: %s", exc, exc_info=True
            )
            raise

    # ------------------------------------------------------------------
    # get_recent
    # ------------------------------------------------------------------

    async def get_recent(
        self,
        hours: int = 24,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return recent episode records.

        Args:
            hours:      How far back to look (default 24 h).
            event_type: Optional filter by event_type.

        Returns:
            List of episode dicts, newest first.
        """
        try:
            from sqlalchemy import select, and_
            from backend.app.models.memory import Memory  # type: ignore[import]
            import uuid

            since = datetime.now(timezone.utc) - timedelta(hours=hours)
            conditions = [
                Memory.user_id == uuid.UUID(self._user_id),
                Memory.memory_type == "episodic",
                Memory.created_at >= since,
                Memory.is_deleted.is_(False),
            ]

            if event_type:
                conditions.append(
                    Memory.source == event_type  # type: ignore[attr-defined]
                )

            stmt = (
                select(Memory)
                .where(and_(*conditions))
                .order_by(Memory.created_at.desc())
                .limit(self._max_episodes)
            )
            result = await self._session.execute(stmt)
            memories = result.scalars().all()

            return [self._memory_to_dict(m) for m in memories]

        except ImportError:
            logger.warning("Memory model not available.")
            return []

    # ------------------------------------------------------------------
    # search_episodes
    # ------------------------------------------------------------------

    async def search_episodes(
        self,
        query: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Semantic search over episodic memories using embedding similarity.

        Falls back to naive substring search if the AI client is unavailable.

        Args:
            query: Natural language search string.
            limit: Maximum results to return.

        Returns:
            List of episode dicts sorted by relevance (most relevant first).
        """
        try:
            from backend.app.ai.client import get_openai_client
            from backend.app.memory.vector_store import VectorStore

            ai = get_openai_client()
            embeddings = await ai.embedding(query)
            query_vec = embeddings[0]

            # Fetch candidate episodes from DB
            recent = await self.get_recent(hours=24 * 30)  # last 30 days
            if not recent:
                return []

            # Build transient vector store from candidates
            store = VectorStore()
            episodes_with_content: list[dict[str, Any]] = []

            # Get embeddings for all candidates (batch)
            texts = [ep["content"] for ep in recent]
            # Batch in chunks of 100 to avoid token limits
            chunk_size = 100
            all_vecs: list[list[float]] = []
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i : i + chunk_size]
                vecs = await ai.embedding(chunk)
                all_vecs.extend(vecs)

            for ep, vec in zip(recent, all_vecs):
                store.add(
                    id=ep["id"],
                    embedding=vec,
                    metadata={"event_type": ep.get("event_type", "")},
                )
                episodes_with_content.append(ep)

            results = store.search(query_vec, k=limit)
            id_to_ep = {ep["id"]: ep for ep in episodes_with_content}
            return [id_to_ep[r.id] for r in results if r.id in id_to_ep]

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Semantic search failed (%s); falling back to substring search.",
                exc,
            )
            # Fallback: naive substring match
            recent = await self.get_recent(hours=24 * 30)
            query_lower = query.lower()
            matched = [
                ep
                for ep in recent
                if query_lower in ep.get("content", "").lower()
            ]
            return matched[:limit]

    # ------------------------------------------------------------------
    # summarize_day
    # ------------------------------------------------------------------

    async def summarize_day(self, target_date: date) -> str:
        """
        Produce an AI-written narrative summary of all episodes on *target_date*.

        Args:
            target_date: The calendar date to summarise.

        Returns:
            Narrative summary string.
        """
        day_start = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            0, 0, 0,
            tzinfo=timezone.utc,
        )
        day_end = day_start + timedelta(days=1)

        try:
            from sqlalchemy import select, and_
            from backend.app.models.memory import Memory  # type: ignore[import]
            import uuid

            stmt = (
                select(Memory)
                .where(
                    and_(
                        Memory.user_id == uuid.UUID(self._user_id),
                        Memory.memory_type == "episodic",
                        Memory.created_at >= day_start,
                        Memory.created_at < day_end,
                        Memory.is_deleted.is_(False),
                    )
                )
                .order_by(Memory.created_at)
            )
            result = await self._session.execute(stmt)
            episodes = result.scalars().all()

        except ImportError:
            logger.warning("Memory model unavailable; cannot summarize day.")
            return f"No data available for {target_date.isoformat()}."

        if not episodes:
            return f"No activity recorded for {target_date.strftime('%B %d, %Y')}."

        bullet_lines = [
            f"  • [{m.source or 'event'}] {m.content}"  # type: ignore[attr-defined]
            for m in episodes
        ]
        corpus = "\n".join(bullet_lines)

        try:
            from backend.app.ai.client import get_openai_client

            ai = get_openai_client()
            prompt = (
                f"Write a concise narrative summary of the following events "
                f"that occurred on {target_date.strftime('%B %d, %Y')}. "
                "Keep it under 150 words, written in first-person past tense.\n\n"
                f"Events:\n{corpus}"
            )
            summary = await ai.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=200,
            )
            return str(summary)

        except Exception as exc:  # noqa: BLE001
            logger.warning("AI summary generation failed: %s", exc)
            return (
                f"Activity on {target_date.strftime('%B %d, %Y')}:\n{corpus}"
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _default_importance(self, event_type: str) -> float:
        """Return a default importance score based on event type."""
        importance_map: dict[str, float] = {
            "task_completed": 0.9,
            "task_created": 0.7,
            "voice_command": 0.6,
            "telegram_received": 0.5,
            "telegram_sent": 0.4,
            "conversation": 0.5,
            "reminder_fired": 0.8,
            "memory_consolidated": 0.3,
            "system": 0.2,
        }
        return importance_map.get(event_type, 0.5)

    def _memory_to_dict(self, memory: Any) -> dict[str, Any]:
        """Convert a Memory ORM instance to a plain dict."""
        meta: dict[str, Any] = memory.metadata_ or {}  # type: ignore[attr-defined]
        return {
            "id": str(memory.id),
            "event_type": meta.get("event_type", memory.source or ""),
            "content": memory.content,
            "importance": float(memory.importance or 0.5),  # type: ignore[attr-defined]
            "source": memory.source,  # type: ignore[attr-defined]
            "created_at": memory.created_at.isoformat(),
            "metadata": meta,
        }

    async def _trim_if_needed(self) -> None:
        """
        Soft-delete the oldest episodic memories when the count exceeds
        *max_episodes*.
        """
        try:
            from sqlalchemy import select, func
            from backend.app.models.memory import Memory  # type: ignore[import]
            import uuid

            count_stmt = select(func.count(Memory.id)).where(
                Memory.user_id == uuid.UUID(self._user_id),
                Memory.memory_type == "episodic",
                Memory.is_deleted.is_(False),
            )
            count_result = await self._session.execute(count_stmt)
            count = count_result.scalar() or 0

            if count <= self._max_episodes:
                return

            excess = count - self._max_episodes
            # Find and soft-delete oldest excess episodes
            oldest_stmt = (
                select(Memory)
                .where(
                    Memory.user_id == uuid.UUID(self._user_id),
                    Memory.memory_type == "episodic",
                    Memory.is_deleted.is_(False),
                )
                .order_by(Memory.created_at)
                .limit(excess)
            )
            oldest_result = await self._session.execute(oldest_stmt)
            to_delete = oldest_result.scalars().all()

            for mem in to_delete:
                mem.is_deleted = True

            logger.debug(
                "EpisodicMemory trimmed %d old episodes for user %s.",
                len(to_delete),
                self._user_id,
            )
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("EpisodicMemory._trim_if_needed error: %s", exc)
