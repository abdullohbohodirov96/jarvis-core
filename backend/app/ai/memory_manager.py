"""
Long-term memory management for JARVIS.

The MemoryManager handles:
- Storing new memories with optional embedding generation
- Semantic search via cosine similarity on stored embeddings
- Text-based fallback search when embeddings are not available
- Memory consolidation, decay, and context formatting
- AI-driven extraction of facts/preferences from conversation turns
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone, timedelta
from typing import Any

from sqlalchemy import select, text, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.client import OpenAIClient
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Maximum characters for the memory context block injected into prompts
_MAX_CONTEXT_CHARS = 4000


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------


class MemoryManager:
    """
    Manages long-term memories for a single user.

    Memories are stored in the ``memories`` database table with optional
    embedding vectors for semantic retrieval.
    """

    def __init__(
        self,
        user_id: int,
        db_session: AsyncSession,
        openai_client: OpenAIClient,
    ) -> None:
        self._user_id = user_id
        self._db = db_session
        self._client = openai_client

    # ------------------------------------------------------------------
    # store_memory
    # ------------------------------------------------------------------

    async def store_memory(
        self,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> Any:
        """
        Persist a memory to the database, including its embedding vector.

        Args:
            content: Text content to memorise.
            memory_type: Category string (fact, preference, event, contact, etc.).
            importance: Float 0–1 indicating how important the memory is.
            tags: Optional list of tag strings.

        Returns:
            The persisted Memory ORM object.
        """
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty.")

        importance = max(0.0, min(1.0, importance))
        tags = tags or []

        # Generate embedding for semantic search
        embedding: list[float] | None = None
        try:
            vectors = await self._client.embedding(content)
            embedding = vectors[0] if vectors else None
        except Exception as exc:
            logger.warning("Could not generate embedding for memory: %s", exc)

        try:
            from app.models.memory import Memory  # type: ignore[import]

            memory = Memory(
                user_id=self._user_id,
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=tags,
                embedding=embedding,
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
                access_count=0,
            )
            self._db.add(memory)
            await self._db.flush()
            await self._db.refresh(memory)

            logger.info(
                "MemoryManager.store_memory: id=%s type=%s importance=%.2f user=%s",
                memory.id,
                memory_type,
                importance,
                self._user_id,
            )
            return memory
        except Exception as exc:
            logger.exception("store_memory DB error: %s", exc)
            raise

    # ------------------------------------------------------------------
    # search_memories
    # ------------------------------------------------------------------

    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        memory_type: str | None = None,
    ) -> list[Any]:
        """
        Search memories using semantic similarity (preferred) or text search.

        1. Generates an embedding for the query.
        2. Fetches candidate memories from DB.
        3. Scores them by cosine similarity.
        4. Falls back to ILIKE text search if embedding generation fails.

        Args:
            query: Natural language search string.
            limit: Maximum number of results to return.
            memory_type: Optional filter by memory type.

        Returns:
            List of Memory ORM objects sorted by relevance.
        """
        query = query.strip()
        if not query:
            return []

        # Try embedding-based search
        query_embedding: list[float] | None = None
        try:
            vectors = await self._client.embedding(query)
            query_embedding = vectors[0] if vectors else None
        except Exception as exc:
            logger.warning("search_memories: could not embed query: %s", exc)

        try:
            from app.models.memory import Memory  # type: ignore[import]

            # Fetch a superset of candidates from DB (filter by type if given)
            stmt = select(Memory).where(Memory.user_id == self._user_id)
            if memory_type:
                stmt = stmt.where(Memory.memory_type == memory_type)
            # Limit the candidate pool for performance
            stmt = stmt.order_by(Memory.importance.desc()).limit(min(limit * 10, 200))

            result = await self._db.execute(stmt)
            candidates: list[Any] = list(result.scalars().all())

            if not candidates:
                return []

            # ── Embedding-based ranking ──────────────────────────────────────
            if query_embedding is not None:
                scored: list[tuple[float, Any]] = []
                for mem in candidates:
                    mem_emb = getattr(mem, "embedding", None)
                    if mem_emb:
                        try:
                            sim = self._compute_similarity(query_embedding, mem_emb)
                        except Exception:
                            sim = 0.0
                    else:
                        # For memories without embeddings, use a text heuristic
                        content_lower = (mem.content or "").lower()
                        query_lower = query.lower()
                        words = re.findall(r"\w+", query_lower)
                        hits = sum(1 for w in words if w in content_lower)
                        sim = hits / max(len(words), 1) * 0.5  # Cap at 0.5

                    scored.append((sim, mem))

                scored.sort(key=lambda x: x[0], reverse=True)
                top = [m for _, m in scored[:limit]]

            else:
                # ── Text search fallback ─────────────────────────────────────
                pattern = f"%{query}%"
                text_stmt = (
                    select(Memory)
                    .where(Memory.user_id == self._user_id)
                    .where(Memory.content.ilike(pattern))
                )
                if memory_type:
                    text_stmt = text_stmt.where(Memory.memory_type == memory_type)
                text_stmt = text_stmt.order_by(Memory.importance.desc()).limit(limit)

                text_result = await self._db.execute(text_stmt)
                top = list(text_result.scalars().all())

            # Update access metadata for retrieved memories
            now = datetime.now(timezone.utc)
            for mem in top:
                mem.last_accessed = now
                mem.access_count = (getattr(mem, "access_count", 0) or 0) + 1

            if top:
                await self._db.flush()

            return top

        except Exception as exc:
            logger.exception("search_memories error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # consolidate_memories
    # ------------------------------------------------------------------

    async def consolidate_memories(self) -> int:
        """
        Consolidate similar memories and remove expired/unimportant ones.

        - Deletes memories that have decayed below the importance threshold
          and have not been accessed recently.
        - Merges near-duplicate memories (cosine similarity > 0.95) by
          retaining the higher-importance one.

        Returns:
            Number of memories removed.
        """
        removed = 0
        try:
            from app.models.memory import Memory  # type: ignore[import]

            cutoff_date = datetime.now(timezone.utc) - timedelta(
                hours=settings.MEMORY_DECAY_HOURS
            )
            # Delete trivially unimportant, stale memories
            stmt = select(Memory).where(
                Memory.user_id == self._user_id,
                Memory.importance < 0.1,
                Memory.last_accessed < cutoff_date,
            )
            result = await self._db.execute(stmt)
            stale = result.scalars().all()

            for mem in stale:
                await self._db.delete(mem)
                removed += 1

            # Find near-duplicate pairs via embeddings
            all_stmt = (
                select(Memory)
                .where(Memory.user_id == self._user_id)
                .where(Memory.embedding.isnot(None))
                .order_by(Memory.importance.desc())
                .limit(500)
            )
            all_result = await self._db.execute(all_stmt)
            all_memories: list[Any] = list(all_result.scalars().all())

            seen_ids: set[int] = set()
            for i, mem_a in enumerate(all_memories):
                if mem_a.id in seen_ids:
                    continue
                for mem_b in all_memories[i + 1:]:
                    if mem_b.id in seen_ids:
                        continue
                    try:
                        sim = self._compute_similarity(mem_a.embedding, mem_b.embedding)
                    except Exception:
                        continue

                    if sim > 0.95:
                        # Keep the higher-importance memory; delete the other
                        keep = mem_a if (mem_a.importance or 0) >= (mem_b.importance or 0) else mem_b
                        drop = mem_b if keep is mem_a else mem_a
                        seen_ids.add(drop.id)
                        await self._db.delete(drop)
                        removed += 1

            await self._db.flush()
            logger.info(
                "consolidate_memories: removed %d memories for user=%s",
                removed,
                self._user_id,
            )
        except Exception as exc:
            logger.exception("consolidate_memories error: %s", exc)

        return removed

    # ------------------------------------------------------------------
    # get_relevant_context
    # ------------------------------------------------------------------

    async def get_relevant_context(
        self,
        query: str,
        max_tokens: int = 1000,
    ) -> str:
        """
        Retrieve the most relevant memories for injection into a prompt.

        Args:
            query: The current user message or conversation context.
            max_tokens: Approximate token budget for the memory block.

        Returns:
            A formatted string ready to be injected into the system prompt.
        """
        # Estimate character budget: ~4 chars per token
        char_budget = min(max_tokens * 4, _MAX_CONTEXT_CHARS)

        memories = await self.search_memories(query, limit=15)
        if not memories:
            return ""

        # Also fetch standing instructions (high importance, instruction type)
        try:
            from app.models.memory import Memory  # type: ignore[import]

            instr_stmt = (
                select(Memory)
                .where(
                    Memory.user_id == self._user_id,
                    Memory.memory_type == "instruction",
                    Memory.importance >= 0.7,
                )
                .order_by(Memory.importance.desc())
                .limit(5)
            )
            instr_result = await self._db.execute(instr_stmt)
            instructions = list(instr_result.scalars().all())

            # Merge, deduplicate by ID
            existing_ids = {m.id for m in memories}
            for m in instructions:
                if m.id not in existing_ids:
                    memories.insert(0, m)
                    existing_ids.add(m.id)
        except Exception:
            pass

        return self._format_memories_for_context(memories, char_budget)

    # ------------------------------------------------------------------
    # decay_memories
    # ------------------------------------------------------------------

    async def decay_memories(self) -> int:
        """
        Reduce the importance of old memories that have not been accessed recently.

        Applies a decay factor proportional to time since last access.

        Returns:
            Number of memories updated.
        """
        updated = 0
        try:
            from app.models.memory import Memory  # type: ignore[import]

            decay_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            stmt = select(Memory).where(
                Memory.user_id == self._user_id,
                Memory.last_accessed < decay_cutoff,
                Memory.importance > 0.1,
            )
            result = await self._db.execute(stmt)
            memories = result.scalars().all()

            now = datetime.now(timezone.utc)
            for mem in memories:
                days_stale = (now - (mem.last_accessed or mem.created_at)).days
                # Decay factor: 0.5% reduction per day stale, capped at 30% total
                decay = min(days_stale * 0.005, 0.3)
                mem.importance = max(0.0, (mem.importance or 0.5) - decay)
                updated += 1

            if updated:
                await self._db.flush()
                logger.info(
                    "decay_memories: updated %d memories for user=%s",
                    updated,
                    self._user_id,
                )
        except Exception as exc:
            logger.exception("decay_memories error: %s", exc)

        return updated

    # ------------------------------------------------------------------
    # extract_and_store_memories
    # ------------------------------------------------------------------

    async def extract_and_store_memories(
        self,
        conversation: list[dict[str, Any]],
    ) -> list[Any]:
        """
        Use AI to extract memorable facts/preferences from a conversation
        and persist them.

        Args:
            conversation: List of OpenAI message dicts (role + content).

        Returns:
            List of newly created Memory ORM objects.
        """
        if not conversation:
            return []

        from app.ai.prompts.system_prompts import MEMORY_EXTRACTION_PROMPT

        # Flatten conversation to text
        conv_text = "\n".join(
            f"{m.get('role', 'unknown').upper()}: {m.get('content', '')}"
            for m in conversation
            if m.get("content")
        )

        prompt = MEMORY_EXTRACTION_PROMPT.format(conversation=conv_text)

        try:
            raw = await self._client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You extract structured data from conversations. Return only JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.warning("extract_and_store_memories: AI call failed: %s", exc)
            return []

        # Parse the JSON response
        extracted: list[dict[str, Any]] = []
        try:
            # The model may return {"memories": [...]} or a bare array
            parsed = json.loads(raw)  # type: ignore[arg-type]
            if isinstance(parsed, list):
                extracted = parsed
            elif isinstance(parsed, dict):
                extracted = parsed.get("memories", [])
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("extract_and_store_memories: JSON parse error: %s", exc)
            return []

        stored: list[Any] = []
        for item in extracted:
            content = (item.get("content") or "").strip()
            if not content:
                continue
            memory_type = item.get("memory_type", "general")
            importance = float(item.get("importance", 0.5))
            tags = item.get("tags", [])

            try:
                mem = await self.store_memory(
                    content=content,
                    memory_type=memory_type,
                    importance=importance,
                    tags=tags,
                )
                stored.append(mem)
            except Exception as exc:
                logger.warning("extract_and_store_memories: failed to store: %s", exc)

        logger.info(
            "extract_and_store_memories: extracted %d memories for user=%s",
            len(stored),
            self._user_id,
        )
        return stored

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """
        Compute cosine similarity between two embedding vectors.

        Returns a float in [-1, 1] where 1.0 means identical direction.
        """
        if not embedding1 or not embedding2:
            return 0.0
        if len(embedding1) != len(embedding2):
            raise ValueError(
                f"Embedding dimension mismatch: {len(embedding1)} vs {len(embedding2)}"
            )

        dot = sum(a * b for a, b in zip(embedding1, embedding2))
        norm_a = math.sqrt(sum(a * a for a in embedding1))
        norm_b = math.sqrt(sum(b * b for b in embedding2))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _format_memories_for_context(
        self,
        memories: list[Any],
        char_budget: int = _MAX_CONTEXT_CHARS,
    ) -> str:
        """
        Format a list of Memory objects into a prompt-ready string.

        Args:
            memories: List of Memory ORM objects.
            char_budget: Maximum character count for the output.

        Returns:
            Multi-line string with one memory per line, grouped by type.
        """
        if not memories:
            return ""

        # Group by type
        by_type: dict[str, list[Any]] = {}
        for mem in memories:
            mtype = getattr(mem, "memory_type", "general")
            by_type.setdefault(mtype, []).append(mem)

        # Type display order / labels
        type_order = ["instruction", "preference", "fact", "contact", "event", "general"]
        type_labels = {
            "instruction": "Standing Instructions",
            "preference": "Preferences",
            "fact": "Known Facts",
            "contact": "Contacts",
            "event": "Events",
            "general": "Other Notes",
        }

        lines: list[str] = []
        total_chars = 0

        for mtype in type_order + [k for k in by_type if k not in type_order]:
            group = by_type.get(mtype, [])
            if not group:
                continue

            header = f"[{type_labels.get(mtype, mtype.title())}]"
            lines.append(header)
            total_chars += len(header) + 1

            for mem in sorted(group, key=lambda m: -(getattr(m, "importance", 0) or 0)):
                content = getattr(mem, "content", "")
                line = f"• {content}"
                if total_chars + len(line) > char_budget:
                    lines.append("• (additional memories truncated)")
                    break
                lines.append(line)
                total_chars += len(line) + 1

            lines.append("")

        return "\n".join(lines).strip()
