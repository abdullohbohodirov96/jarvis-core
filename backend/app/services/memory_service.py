"""
Memory business logic service for JARVIS.

MemoryService stores, retrieves, and manages the user's personal memory:
facts, preferences, conversation context, and learned information.
Semantic search is performed using OpenAI embeddings and cosine similarity
over a ``memories`` Postgres table (requires pgvector or a fallback).
"""

from __future__ import annotations

import json
import logging
import math
import uuid as _uuid_module
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import text as sa_text

from backend.app.core.config import get_settings
from backend.app.core.exceptions import AIException, NotFoundException

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.app.ai.client import OpenAIClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIM = 1536


class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    CONTEXT = "context"
    CONVERSATION = "conversation"
    TASK = "task"
    EVENT = "event"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Lightweight Memory dataclass
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class Memory:
    id: UUID
    user_id: UUID
    content: str
    memory_type: MemoryType
    importance: float
    tags: list[str]
    embedding: list[float] | None
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None = None
    access_count: int = 0


# ---------------------------------------------------------------------------
# MemoryService
# ---------------------------------------------------------------------------


class MemoryService:
    """
    Manages user memory storage and retrieval.

    Uses OpenAI embeddings for semantic search when available.
    Falls back to keyword (ILIKE) search if embedding is unavailable.

    Args:
        db_session:    Async SQLAlchemy session.
        user_id:       UUID of the owning user.
        openai_client: OpenAIClient for embedding generation.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        user_id: UUID,
        openai_client: OpenAIClient,
    ) -> None:
        self._db = db_session
        self._user_id = user_id
        self._ai = openai_client
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Add / store
    # ------------------------------------------------------------------

    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> Memory:
        """
        Store a new memory item with an embedding vector.

        Args:
            content:     The text to remember.
            memory_type: Category of the memory.
            importance:  Relevance score 0.0–1.0 (higher = more important).
            tags:        Optional list of string tags for filtering.

        Returns:
            The persisted Memory object.
        """
        if not content.strip():
            raise ValueError("Memory content must not be empty.")

        tags = tags or []
        importance = max(0.0, min(1.0, importance))
        now = datetime.now(timezone.utc)
        mem_id = _uuid_module.uuid4()

        # Generate embedding (best-effort; proceed without on failure)
        embedding: list[float] | None = None
        try:
            vectors = await self._ai.embedding(content, model=_EMBEDDING_MODEL)
            embedding = vectors[0] if vectors else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("add_memory: embedding generation failed: %s", exc)

        await self._db.execute(
            sa_text(
                """
                INSERT INTO memories
                    (id, user_id, content, memory_type, importance, tags,
                     embedding, created_at, updated_at)
                VALUES
                    (:id, :user_id, :content, :memory_type, :importance, :tags,
                     :embedding, :now, :now)
                """
            ),
            {
                "id": str(mem_id),
                "user_id": str(self._user_id),
                "content": content.strip(),
                "memory_type": memory_type.value,
                "importance": importance,
                "tags": json.dumps(tags),
                "embedding": json.dumps(embedding) if embedding else None,
                "now": now,
            },
        )
        await self._db.flush()

        logger.info(
            "add_memory: id=%s type=%s importance=%.2f", mem_id, memory_type.value, importance
        )
        return Memory(
            id=mem_id,
            user_id=self._user_id,
            content=content.strip(),
            memory_type=memory_type,
            importance=importance,
            tags=tags,
            embedding=embedding,
            created_at=now,
            updated_at=now,
        )

    # ------------------------------------------------------------------
    # Search / retrieve
    # ------------------------------------------------------------------

    async def search(self, query: str, limit: int = 10) -> list[Memory]:
        """
        Semantic similarity search over stored memories.

        Attempts vector cosine similarity search (requires pgvector column).
        Falls back to ILIKE keyword search if vectors are unavailable.

        Args:
            query: Natural-language search query.
            limit: Maximum results to return.

        Returns:
            List of Memory objects ordered by relevance (desc).
        """
        if not query.strip():
            return []

        # Try embedding-based search
        query_embedding: list[float] | None = None
        try:
            vectors = await self._ai.embedding(query, model=_EMBEDDING_MODEL)
            query_embedding = vectors[0] if vectors else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("search: embedding failed, falling back to keyword: %s", exc)

        if query_embedding:
            memories = await self._vector_search(query_embedding, limit)
            if memories:
                return memories

        # Fallback: keyword search
        return await self._keyword_search(query, limit)

    async def get_context_for_prompt(
        self,
        query: str,
        max_tokens: int = 500,
    ) -> str:
        """
        Retrieve the most relevant memories as a formatted context block for
        inclusion in an LLM prompt.

        Args:
            query:      The current query / topic to find relevant memories for.
            max_tokens: Approximate token budget (4 chars ≈ 1 token).

        Returns:
            Formatted context string, or empty string if nothing relevant.
        """
        memories = await self.search(query, limit=15)
        if not memories:
            return ""

        char_budget = max_tokens * 4
        lines: list[str] = []
        used = 0

        for mem in sorted(memories, key=lambda m: m.importance, reverse=True):
            line = f"- [{mem.memory_type.value}] {mem.content}"
            if used + len(line) > char_budget:
                break
            lines.append(line)
            used += len(line)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def consolidate(self) -> dict[str, int]:
        """
        Consolidate the memory store:
        1. Mark low-importance, old memories as deleted (decay).
        2. Deduplicate near-identical entries (by text equality).

        Returns:
            Stats dict: {"decayed": int, "deduplicated": int}.
        """
        settings = get_settings()
        decay_cutoff = datetime.now(timezone.utc) - timedelta(
            hours=settings.MEMORY_DECAY_HOURS
        )

        # Decay: soft-delete old, low-importance non-preference memories
        decay_result = await self._db.execute(
            sa_text(
                """
                UPDATE memories
                   SET is_deleted = true
                 WHERE user_id     = :user_id
                   AND importance  < 0.3
                   AND memory_type != :pref
                   AND created_at  < :cutoff
                   AND is_deleted  = false
                RETURNING id
                """
            ),
            {
                "user_id": str(self._user_id),
                "pref": MemoryType.PREFERENCE.value,
                "cutoff": decay_cutoff,
            },
        )
        decayed_ids = decay_result.fetchall()
        decayed = len(decayed_ids)

        # Deduplicate: find exact content duplicates and keep newest
        dedup_result = await self._db.execute(
            sa_text(
                """
                UPDATE memories AS m
                   SET is_deleted = true
                  FROM (
                      SELECT MIN(id) AS keep_id, content
                        FROM memories
                       WHERE user_id    = :user_id
                         AND is_deleted = false
                       GROUP BY content
                      HAVING COUNT(*) > 1
                  ) AS dups
                 WHERE m.content   = dups.content
                   AND m.id       != dups.keep_id
                   AND m.user_id   = :user_id
                RETURNING m.id
                """
            ),
            {"user_id": str(self._user_id)},
        )
        deduped_ids = dedup_result.fetchall()
        deduplicated = len(deduped_ids)

        await self._db.flush()
        logger.info(
            "consolidate: decayed=%d deduplicated=%d user=%s",
            decayed, deduplicated, self._user_id,
        )
        return {"decayed": decayed, "deduplicated": deduplicated}

    async def get_user_facts(self) -> list[Memory]:
        """
        Return all stored facts for the user (memory_type == FACT).

        Returns:
            List of Memory objects.
        """
        result = await self._db.execute(
            sa_text(
                """
                SELECT id, user_id, content, memory_type, importance, tags,
                       embedding, created_at, updated_at, last_accessed_at, access_count
                  FROM memories
                 WHERE user_id     = :user_id
                   AND memory_type = :fact
                   AND is_deleted  = false
                 ORDER BY importance DESC, created_at DESC
                 LIMIT 200
                """
            ),
            {"user_id": str(self._user_id), "fact": MemoryType.FACT.value},
        )
        return [self._row_to_memory(row) for row in result.fetchall()]

    async def get_preferences(self) -> dict[str, Any]:
        """
        Return all stored preferences as a flat dict {key: value}.

        Preference memories are expected to be stored with content in the
        form ``"key: value"`` for easy parsing.

        Returns:
            Dict of preference key→value pairs.
        """
        result = await self._db.execute(
            sa_text(
                """
                SELECT content
                  FROM memories
                 WHERE user_id     = :user_id
                   AND memory_type = :pref
                   AND is_deleted  = false
                 ORDER BY updated_at DESC
                """
            ),
            {"user_id": str(self._user_id), "pref": MemoryType.PREFERENCE.value},
        )
        prefs: dict[str, Any] = {}
        for (content,) in result.fetchall():
            if ": " in content:
                key, _, value = content.partition(": ")
                prefs[key.strip()] = value.strip()
            else:
                prefs[content.strip()] = True
        return prefs

    async def update_preference(self, key: str, value: Any) -> Memory:
        """
        Upsert a preference memory item.

        Args:
            key:   Preference name.
            value: Preference value (stringified).

        Returns:
            The Memory object (new or updated).
        """
        content = f"{key}: {value}"

        # Check if a preference with this key already exists
        result = await self._db.execute(
            sa_text(
                """
                SELECT id FROM memories
                 WHERE user_id     = :user_id
                   AND memory_type = :pref
                   AND content     LIKE :key_like
                   AND is_deleted  = false
                 LIMIT 1
                """
            ),
            {
                "user_id": str(self._user_id),
                "pref": MemoryType.PREFERENCE.value,
                "key_like": f"{key}:%",
            },
        )
        row = result.fetchone()
        if row:
            existing_id = UUID(str(row[0]))
            await self._db.execute(
                sa_text(
                    """
                    UPDATE memories
                       SET content    = :content,
                           updated_at = :now
                     WHERE id = :id
                    """
                ),
                {
                    "id": str(existing_id),
                    "content": content,
                    "now": datetime.now(timezone.utc),
                },
            )
            await self._db.flush()
            logger.debug("update_preference: updated key=%s", key)
            # Re-fetch and return
            fetch_result = await self._db.execute(
                sa_text(
                    """
                    SELECT id, user_id, content, memory_type, importance, tags,
                           embedding, created_at, updated_at,
                           last_accessed_at, access_count
                      FROM memories WHERE id = :id
                    """
                ),
                {"id": str(existing_id)},
            )
            return self._row_to_memory(fetch_result.fetchone())
        else:
            return await self.add_memory(
                content=content,
                memory_type=MemoryType.PREFERENCE,
                importance=1.0,  # Preferences are always high-importance
                tags=["preference", key],
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _vector_search(
        self, query_embedding: list[float], limit: int
    ) -> list[Memory]:
        """
        Cosine similarity search using the stored embedding JSON column.

        This performs the dot-product calculation in Python (fetches all
        non-deleted memories with embeddings).  For production with large
        memory stores, replace with a pgvector ``<=>`` operator query.
        """
        try:
            result = await self._db.execute(
                sa_text(
                    """
                    SELECT id, user_id, content, memory_type, importance, tags,
                           embedding, created_at, updated_at,
                           last_accessed_at, access_count
                      FROM memories
                     WHERE user_id    = :user_id
                       AND is_deleted = false
                       AND embedding  IS NOT NULL
                     ORDER BY importance DESC
                     LIMIT 500
                    """
                ),
                {"user_id": str(self._user_id)},
            )
            rows = result.fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("_vector_search DB query failed: %s", exc)
            return []

        # Compute cosine similarities in Python
        scored: list[tuple[float, Any]] = []
        for row in rows:
            try:
                raw_emb = self._get_val(row, "embedding", 6)
                if raw_emb is None:
                    continue
                if isinstance(raw_emb, str):
                    emb: list[float] = json.loads(raw_emb)
                elif isinstance(raw_emb, list):
                    emb = raw_emb
                else:
                    continue
                score = self._cosine_similarity(query_embedding, emb)
                scored.append((score, row))
            except Exception:  # noqa: BLE001
                continue

        # Sort by score desc, take top-limit
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:limit]

        memories = [self._row_to_memory(row) for _, row in top]

        # Update last_accessed_at in background (best-effort)
        ids = [str(m.id) for m in memories]
        if ids:
            try:
                now = datetime.now(timezone.utc)
                await self._db.execute(
                    sa_text(
                        f"""
                        UPDATE memories
                           SET last_accessed_at = :now,
                               access_count     = access_count + 1
                         WHERE id = ANY(ARRAY[{','.join(f"'{i}'" for i in ids)}]::uuid[])
                        """
                    ),
                    {"now": now},
                )
                await self._db.flush()
            except Exception:  # noqa: BLE001
                pass

        return memories

    async def _keyword_search(self, query: str, limit: int) -> list[Memory]:
        """ILIKE keyword fallback search."""
        pattern = f"%{query.strip()}%"
        result = await self._db.execute(
            sa_text(
                """
                SELECT id, user_id, content, memory_type, importance, tags,
                       embedding, created_at, updated_at,
                       last_accessed_at, access_count
                  FROM memories
                 WHERE user_id    = :user_id
                   AND is_deleted = false
                   AND content    ILIKE :pattern
                 ORDER BY importance DESC, updated_at DESC
                 LIMIT :limit
                """
            ),
            {"user_id": str(self._user_id), "pattern": pattern, "limit": limit},
        )
        return [self._row_to_memory(row) for row in result.fetchall()]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two equal-length float vectors."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _get_val(row: Any, key: str, idx: int) -> Any:
        try:
            return row[key]
        except (TypeError, KeyError):
            return row[idx]

    def _row_to_memory(self, row: Any) -> Memory:
        g = self._get_val

        raw_tags = g(row, "tags", 5)
        tags: list[str] = []
        if isinstance(raw_tags, str):
            try:
                tags = json.loads(raw_tags)
            except Exception:  # noqa: BLE001
                pass
        elif isinstance(raw_tags, list):
            tags = raw_tags

        raw_emb = g(row, "embedding", 6)
        embedding: list[float] | None = None
        if raw_emb is not None:
            if isinstance(raw_emb, str):
                try:
                    embedding = json.loads(raw_emb)
                except Exception:  # noqa: BLE001
                    pass
            elif isinstance(raw_emb, list):
                embedding = raw_emb

        mem_type_val = g(row, "memory_type", 3)
        try:
            mem_type = MemoryType(mem_type_val)
        except ValueError:
            mem_type = MemoryType.OTHER

        return Memory(
            id=UUID(str(g(row, "id", 0))),
            user_id=UUID(str(g(row, "user_id", 1))),
            content=str(g(row, "content", 2) or ""),
            memory_type=mem_type,
            importance=float(g(row, "importance", 4) or 0.5),
            tags=tags,
            embedding=embedding,
            created_at=g(row, "created_at", 7),
            updated_at=g(row, "updated_at", 8),
            last_accessed_at=g(row, "last_accessed_at", 9),
            access_count=int(g(row, "access_count", 10) or 0),
        )
