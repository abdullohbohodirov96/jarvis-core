"""
Memory / knowledge-base endpoints for JARVIS.

POST   /memory                   – store a memory item
GET    /memory/search            – semantic search over memories
GET    /memory                   – list memories (paginated)
DELETE /memory/{memory_id}       – delete a memory item
POST   /memory/consolidate       – run memory consolidation (deduplicate / summarise)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field

from backend.app.core.config import get_settings
from backend.app.core.exceptions import AIException, NotFoundException
from backend.app.core.logging_config import get_logger
from backend.app.utils.helpers import generate_id, now_utc

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class MemoryCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=10_000)
    category: Optional[str] = Field(default=None, max_length=100)
    tags: List[str] = Field(default_factory=list)
    source: Optional[str] = Field(
        default=None,
        description="Source of this memory (e.g. 'chat', 'voice', 'telegram')",
    )
    importance: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Importance score (0=low, 1=high)",
    )


class MemoryOut(BaseModel):
    memory_id: str
    content: str
    category: Optional[str]
    tags: List[str]
    source: Optional[str]
    importance: float
    access_count: int
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime]


class MemoryListResponse(BaseModel):
    memories: List[MemoryOut]
    total: int
    page: int
    page_size: int


class MemorySearchResponse(BaseModel):
    results: List[MemoryOut]
    query: str
    total_found: int


class ConsolidateResponse(BaseModel):
    original_count: int
    consolidated_count: int
    removed_count: int
    summary: Optional[str]
    run_at: datetime


# ---------------------------------------------------------------------------
# In-process memory store
# ---------------------------------------------------------------------------

_memories: Dict[str, Dict[str, Any]] = {}


def _compute_expires_at(importance: float) -> Optional[datetime]:
    """Higher importance → longer retention."""
    base_hours = settings.MEMORY_DECAY_HOURS
    # Scale: importance 1.0 → 2× base, importance 0.0 → 0.5× base
    factor = 0.5 + importance * 1.5
    hours = base_hours * factor
    return now_utc() + timedelta(hours=hours)


def _dict_to_out(m: Dict[str, Any]) -> MemoryOut:
    return MemoryOut(
        memory_id=m["memory_id"],
        content=m["content"],
        category=m.get("category"),
        tags=m.get("tags", []),
        source=m.get("source"),
        importance=m.get("importance", 1.0),
        access_count=m.get("access_count", 0),
        created_at=datetime.fromisoformat(m["created_at"]),
        updated_at=datetime.fromisoformat(m["updated_at"]),
        expires_at=(
            datetime.fromisoformat(m["expires_at"]) if m.get("expires_at") else None
        ),
    )


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


async def _embed(text: str) -> Optional[List[float]]:
    """Return an OpenAI embedding vector, or None if unavailable."""
    try:
        from openai import AsyncOpenAI  # type: ignore

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        resp = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return resp.data[0].embedding
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=MemoryOut,
    status_code=status.HTTP_201_CREATED,
    summary="Store a new memory",
)
async def store_memory(payload: MemoryCreate) -> MemoryOut:
    """Persist a memory item, optionally computing an embedding for later search."""
    if len(_memories) >= settings.MAX_MEMORY_ITEMS:
        # Evict the least important, oldest memory
        oldest = min(
            _memories.values(),
            key=lambda m: (m.get("importance", 0.0), m["created_at"]),
        )
        del _memories[oldest["memory_id"]]
        logger.info("memory_evicted", memory_id=oldest["memory_id"])

    memory_id = generate_id()
    now = now_utc()
    embedding = await _embed(payload.content)

    record: Dict[str, Any] = {
        "memory_id": memory_id,
        "content": payload.content,
        "category": payload.category,
        "tags": payload.tags,
        "source": payload.source,
        "importance": payload.importance,
        "access_count": 0,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "expires_at": _compute_expires_at(payload.importance).isoformat(),
        "embedding": embedding,
    }
    _memories[memory_id] = record
    logger.info("memory_stored", memory_id=memory_id, content_len=len(payload.content))
    return _dict_to_out(record)


@router.get(
    "/search",
    response_model=MemorySearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic search over memories",
)
async def search_memories(
    q: str = Query(..., min_length=1, max_length=2000, description="Search query"),
    top_k: int = Query(default=10, ge=1, le=50),
    category: Optional[str] = Query(default=None),
) -> MemorySearchResponse:
    """Find the most semantically similar memories to the query.

    Falls back to keyword matching when OpenAI embeddings are unavailable.
    """
    now = now_utc()
    candidates = [
        m for m in _memories.values()
        if not (m.get("expires_at") and datetime.fromisoformat(m["expires_at"]) < now)
    ]
    if category:
        candidates = [m for m in candidates if m.get("category") == category]

    query_embedding = await _embed(q)

    if query_embedding is not None:
        # Semantic ranking
        scored = [
            (m, _cosine_similarity(query_embedding, m.get("embedding") or []))
            for m in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
    else:
        # Keyword fallback: count term occurrences
        query_terms = q.lower().split()
        scored = []
        for m in candidates:
            content_lower = m["content"].lower()
            score = sum(content_lower.count(term) for term in query_terms)
            scored.append((m, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)

    top = [m for m, _ in scored[:top_k]]

    # Increment access counts
    for m in top:
        m["access_count"] = m.get("access_count", 0) + 1

    return MemorySearchResponse(
        results=[_dict_to_out(m) for m in top],
        query=q,
        total_found=len(top),
    )


@router.get(
    "",
    response_model=MemoryListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all memories (paginated)",
)
async def list_memories(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    category: Optional[str] = Query(default=None),
    include_expired: bool = Query(default=False),
) -> MemoryListResponse:
    """Return all stored memories, sorted by importance then recency."""
    now = now_utc()
    items = list(_memories.values())

    if not include_expired:
        items = [
            m for m in items
            if not (m.get("expires_at") and datetime.fromisoformat(m["expires_at"]) < now)
        ]

    if category:
        items = [m for m in items if m.get("category") == category]

    items.sort(
        key=lambda m: (-m.get("importance", 0.0), m["created_at"]),
        reverse=False,
    )

    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size

    return MemoryListResponse(
        memories=[_dict_to_out(m) for m in items[start:end]],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.delete(
    "/{memory_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a memory item",
)
async def delete_memory(memory_id: str) -> Dict[str, Any]:
    if memory_id not in _memories:
        raise NotFoundException(
            message=f"Memory '{memory_id}' not found.",
            details={"memory_id": memory_id},
        )
    del _memories[memory_id]
    logger.info("memory_deleted", memory_id=memory_id)
    return {"status": "deleted", "memory_id": memory_id}


@router.post(
    "/consolidate",
    response_model=ConsolidateResponse,
    status_code=status.HTTP_200_OK,
    summary="Run memory consolidation",
)
async def consolidate_memories() -> ConsolidateResponse:
    """Remove expired and duplicate memories, optionally summarise similar ones.

    Steps:
      1. Remove expired items.
      2. Remove near-duplicate items (cosine similarity > 0.95).
      3. Use the LLM to produce a brief summary of what was consolidated.
    """
    now = now_utc()
    original_count = len(_memories)

    # Step 1: Remove expired
    expired_keys = [
        k for k, m in _memories.items()
        if m.get("expires_at") and datetime.fromisoformat(m["expires_at"]) < now
    ]
    for k in expired_keys:
        del _memories[k]

    # Step 2: Remove near-duplicates (O(n²) — acceptable for < 1000 items)
    memory_list = list(_memories.values())
    duplicate_ids: set[str] = set()

    for i in range(len(memory_list)):
        if memory_list[i]["memory_id"] in duplicate_ids:
            continue
        emb_i = memory_list[i].get("embedding")
        if not emb_i:
            continue
        for j in range(i + 1, len(memory_list)):
            if memory_list[j]["memory_id"] in duplicate_ids:
                continue
            emb_j = memory_list[j].get("embedding")
            if not emb_j:
                continue
            sim = _cosine_similarity(emb_i, emb_j)
            if sim > 0.95:
                # Keep the one with higher importance
                if memory_list[i]["importance"] >= memory_list[j]["importance"]:
                    duplicate_ids.add(memory_list[j]["memory_id"])
                else:
                    duplicate_ids.add(memory_list[i]["memory_id"])
                    break

    for dup_id in duplicate_ids:
        _memories.pop(dup_id, None)

    consolidated_count = len(_memories)
    removed_count = original_count - consolidated_count

    # Step 3: LLM summary
    summary_text: Optional[str] = None
    if removed_count > 0:
        remaining_sample = list(_memories.values())[:20]
        sample_texts = "\n".join(
            f"- {m['content'][:200]}" for m in remaining_sample
        )
        prompt = (
            f"I just consolidated a memory store, removing {removed_count} items. "
            f"Here are some of the remaining {consolidated_count} memories:\n"
            f"{sample_texts}\n\n"
            "Briefly describe what types of knowledge remain (1-2 sentences)."
        )
        try:
            from openai import AsyncOpenAI  # type: ignore

            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            resp = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                stream=False,
            )
            summary_text = (resp.choices[0].message.content or "").strip()
        except Exception:
            summary_text = (
                f"Consolidated memory store: removed {removed_count} items, "
                f"{consolidated_count} items remain."
            )

    logger.info(
        "memory_consolidated",
        original=original_count,
        removed=removed_count,
        remaining=consolidated_count,
    )

    return ConsolidateResponse(
        original_count=original_count,
        consolidated_count=consolidated_count,
        removed_count=removed_count,
        summary=summary_text,
        run_at=now_utc(),
    )
