"""
Memory tools for the JARVIS agent.

Allow the AI to explicitly store, search, and retrieve long-term memories
on behalf of the user.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.tools.base import BaseTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StoreMemoryTool
# ---------------------------------------------------------------------------


class StoreMemoryTool(BaseTool):
    """Store an important piece of information in long-term memory."""

    name = "store_memory"
    description = (
        "Store an important piece of information, fact, preference, or note "
        "into the user's long-term memory. Use this when the user shares "
        "something they want JARVIS to remember permanently, or when you "
        "identify something critical to remember for future interactions."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The information to store. Be specific and concise.",
            },
            "memory_type": {
                "type": "string",
                "enum": [
                    "fact",
                    "preference",
                    "event",
                    "contact",
                    "instruction",
                    "general",
                ],
                "description": (
                    "Category of memory: 'fact' for objective facts, "
                    "'preference' for user preferences/likes/dislikes, "
                    "'event' for past or upcoming events, "
                    "'contact' for information about people, "
                    "'instruction' for standing orders/rules, "
                    "'general' for anything else."
                ),
            },
            "importance": {
                "type": "number",
                "description": (
                    "Importance score between 0.0 (trivial) and 1.0 (critical). "
                    "Defaults to 0.5."
                ),
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for easier retrieval later.",
            },
        },
        "required": ["content"],
    }

    def __init__(
        self,
        user_id: int,
        db_session: AsyncSession,
        memory_manager: Any | None = None,
    ) -> None:
        self._user_id = user_id
        self._db = db_session
        self._memory_manager = memory_manager

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        content: str = params["content"].strip()
        memory_type: str = params.get("memory_type", "general")
        importance: float = float(params.get("importance", 0.5))
        tags: list[str] = params.get("tags", [])

        if not content:
            return {"success": False, "error": "Memory content cannot be empty."}

        # Clamp importance
        importance = max(0.0, min(1.0, importance))

        try:
            if self._memory_manager is not None:
                memory = await self._memory_manager.store_memory(
                    content=content,
                    memory_type=memory_type,
                    importance=importance,
                    tags=tags,
                )
                memory_id = getattr(memory, "id", None)
            else:
                # Direct DB persistence fallback
                from app.models.memory import Memory  # type: ignore[import]

                memory_obj = Memory(
                    user_id=self._user_id,
                    content=content,
                    memory_type=memory_type,
                    importance=importance,
                    tags=tags,
                    created_at=datetime.now(timezone.utc),
                    last_accessed=datetime.now(timezone.utc),
                )
                self._db.add(memory_obj)
                await self._db.flush()
                await self._db.refresh(memory_obj)
                memory_id = memory_obj.id

            logger.info(
                "StoreMemoryTool: stored memory id=%s type=%s importance=%.2f",
                memory_id,
                memory_type,
                importance,
            )
            return {
                "success": True,
                "memory_id": memory_id,
                "memory_type": memory_type,
                "importance": importance,
                "message": "Memory stored successfully.",
            }
        except Exception as exc:
            logger.exception("StoreMemoryTool failed: %s", exc)
            return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# SearchMemoryTool
# ---------------------------------------------------------------------------


class SearchMemoryTool(BaseTool):
    """Search the user's long-term memories by query."""

    name = "search_memory"
    description = (
        "Search the user's long-term memories for relevant information. "
        "Use this to recall facts, preferences, or past information the user "
        "has shared previously. Performs semantic similarity search."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query.",
            },
            "memory_type": {
                "type": "string",
                "enum": [
                    "fact",
                    "preference",
                    "event",
                    "contact",
                    "instruction",
                    "general",
                    "all",
                ],
                "description": "Filter results to a specific memory type.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of memories to return. Defaults to 5.",
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        user_id: int,
        db_session: AsyncSession,
        memory_manager: Any | None = None,
    ) -> None:
        self._user_id = user_id
        self._db = db_session
        self._memory_manager = memory_manager

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        query: str = params["query"].strip()
        memory_type: str | None = params.get("memory_type")
        if memory_type == "all":
            memory_type = None
        limit: int = min(int(params.get("limit", 5)), 20)

        if not query:
            return {"success": False, "error": "Query cannot be empty.", "memories": []}

        try:
            if self._memory_manager is not None:
                memories = await self._memory_manager.search_memories(
                    query=query,
                    limit=limit,
                    memory_type=memory_type,
                )
                memory_list = [
                    {
                        "id": getattr(m, "id", None),
                        "content": getattr(m, "content", ""),
                        "type": getattr(m, "memory_type", "general"),
                        "importance": getattr(m, "importance", 0.5),
                        "tags": getattr(m, "tags", []),
                        "created_at": (
                            getattr(m, "created_at").isoformat()
                            if getattr(m, "created_at", None)
                            else None
                        ),
                    }
                    for m in memories
                ]
            else:
                from sqlalchemy import select
                from app.models.memory import Memory  # type: ignore[import]

                pattern = f"%{query}%"
                stmt = (
                    select(Memory)
                    .where(Memory.user_id == self._user_id)
                    .where(Memory.content.ilike(pattern))
                )
                if memory_type:
                    stmt = stmt.where(Memory.memory_type == memory_type)
                stmt = stmt.order_by(Memory.importance.desc()).limit(limit)

                result = await self._db.execute(stmt)
                rows = result.scalars().all()
                memory_list = [
                    {
                        "id": m.id,
                        "content": m.content,
                        "type": m.memory_type,
                        "importance": m.importance,
                        "tags": m.tags or [],
                        "created_at": m.created_at.isoformat() if m.created_at else None,
                    }
                    for m in rows
                ]

            return {
                "success": True,
                "query": query,
                "count": len(memory_list),
                "memories": memory_list,
            }
        except Exception as exc:
            logger.exception("SearchMemoryTool failed: %s", exc)
            return {"success": False, "error": str(exc), "memories": []}


# ---------------------------------------------------------------------------
# ListRecentMemoriesTool
# ---------------------------------------------------------------------------


class ListRecentMemoriesTool(BaseTool):
    """List the user's most recently stored memories."""

    name = "list_recent_memories"
    description = (
        "List the user's most recently stored long-term memories. "
        "Useful for reviewing what JARVIS currently knows about the user."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Number of recent memories to return. Defaults to 10.",
                "minimum": 1,
                "maximum": 30,
            },
            "memory_type": {
                "type": "string",
                "enum": [
                    "fact",
                    "preference",
                    "event",
                    "contact",
                    "instruction",
                    "general",
                    "all",
                ],
                "description": "Optional filter by memory type.",
            },
            "min_importance": {
                "type": "number",
                "description": "Only return memories with importance >= this value.",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": [],
    }

    def __init__(
        self,
        user_id: int,
        db_session: AsyncSession,
        memory_manager: Any | None = None,
    ) -> None:
        self._user_id = user_id
        self._db = db_session
        self._memory_manager = memory_manager

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        limit: int = min(int(params.get("limit", 10)), 30)
        memory_type: str | None = params.get("memory_type")
        if memory_type == "all":
            memory_type = None
        min_importance: float = float(params.get("min_importance", 0.0))

        try:
            from sqlalchemy import select
            from app.models.memory import Memory  # type: ignore[import]

            stmt = select(Memory).where(
                Memory.user_id == self._user_id,
                Memory.importance >= min_importance,
            )
            if memory_type:
                stmt = stmt.where(Memory.memory_type == memory_type)
            stmt = stmt.order_by(Memory.created_at.desc()).limit(limit)

            result = await self._db.execute(stmt)
            rows = result.scalars().all()

            memory_list = [
                {
                    "id": m.id,
                    "content": m.content,
                    "type": m.memory_type,
                    "importance": m.importance,
                    "tags": m.tags or [],
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in rows
            ]

            return {
                "success": True,
                "count": len(memory_list),
                "memories": memory_list,
            }
        except Exception as exc:
            logger.exception("ListRecentMemoriesTool failed: %s", exc)
            return {"success": False, "error": str(exc), "memories": []}
