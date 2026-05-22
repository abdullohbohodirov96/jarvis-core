"""
Task business logic service for JARVIS.

TaskService provides a clean async API for creating, querying, updating,
completing, and deleting tasks, as well as AI-powered task extraction from
natural language and summary/statistics generation.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import and_, func, or_, select, text as sa_text, update

from app.core.exceptions import AIException, NotFoundException, ValidationException

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.ai.client import OpenAIClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TaskCreate / Task dataclasses (lightweight; no ORM dependency required)
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class TaskCreate:
    """Data required to create a new task."""

    title: str
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.TODO
    due_date: datetime | None = None
    tags: list[str] = field(default_factory=list)
    source: str = "manual"  # manual | telegram | voice | ai


@dataclass
class Task:
    """Full task representation returned from the DB."""

    id: UUID
    user_id: UUID
    title: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    due_date: datetime | None
    tags: list[str]
    source: str
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# AI prompts
# ---------------------------------------------------------------------------

_EXTRACT_TASKS_SYSTEM = """You are a task extractor.  Parse the following natural language text and
return a JSON array of task objects.

Each object:
{
  "title": "<concise task title, max 100 chars>",
  "description": "<optional detail>",
  "priority": "low" | "medium" | "high" | "urgent",
  "due_date": "<ISO-8601 date or null>",
  "tags": ["<tag1>", "<tag2>"]
}

Return [] if no tasks are found.  Output valid JSON only — no markdown fences, no prose."""

_SEARCH_TASKS_SYSTEM = """You are a search assistant for a task management system.
Given a natural-language search query and a list of tasks (as JSON),
return the IDs of tasks that match the query.
Output a JSON array of ID strings only.  Example: ["id1","id2"]
Return [] if nothing matches."""


# ---------------------------------------------------------------------------
# TaskService
# ---------------------------------------------------------------------------


class TaskService:
    """
    High-level async service for task management.

    Args:
        db_session: Async SQLAlchemy session.
        user_id:    UUID of the authenticated user.
        ai_agent:   Optional OpenAIClient for AI-powered features.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        user_id: UUID,
        ai_agent: OpenAIClient | None = None,
    ) -> None:
        self._db = db_session
        self._user_id = user_id
        self._ai = ai_agent

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create_task(self, task_data: TaskCreate) -> Task:
        """
        Persist a new task to the database.

        Args:
            task_data: TaskCreate dataclass with required fields.

        Returns:
            The newly created Task.

        Raises:
            ValidationException: If the title is empty.
        """
        if not task_data.title.strip():
            raise ValidationException(
                message="Task title must not be empty.",
                code="EMPTY_TASK_TITLE",
            )

        import uuid as _uuid

        task_id = _uuid.uuid4()
        now = datetime.now(timezone.utc)

        await self._db.execute(
            sa_text(
                """
                INSERT INTO tasks
                    (id, user_id, title, description, priority, status,
                     due_date, tags, source, created_at, updated_at)
                VALUES
                    (:id, :user_id, :title, :description, :priority, :status,
                     :due_date, :tags, :source, :now, :now)
                """
            ),
            {
                "id": str(task_id),
                "user_id": str(self._user_id),
                "title": task_data.title.strip()[:200],
                "description": task_data.description.strip()[:2000],
                "priority": task_data.priority.value,
                "status": task_data.status.value,
                "due_date": task_data.due_date,
                "tags": json.dumps(task_data.tags),
                "source": task_data.source,
                "now": now,
            },
        )
        await self._db.flush()

        logger.info(
            "create_task: id=%s user=%s title=%r", task_id, self._user_id, task_data.title
        )

        return Task(
            id=task_id,
            user_id=self._user_id,
            title=task_data.title.strip(),
            description=task_data.description.strip(),
            priority=task_data.priority,
            status=task_data.status,
            due_date=task_data.due_date,
            tags=task_data.tags,
            source=task_data.source,
            created_at=now,
            updated_at=now,
        )

    async def get_tasks(
        self,
        status: TaskStatus | None = None,
        priority: TaskPriority | None = None,
        due_before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Task]:
        """
        Retrieve tasks for the current user with optional filters.

        Args:
            status:     Filter by task status.
            priority:   Filter by priority level.
            due_before: Only return tasks due before this datetime.
            limit:      Maximum rows to return.
            offset:     Pagination offset.

        Returns:
            List of Task objects ordered by due_date asc, created_at asc.
        """
        params: dict[str, Any] = {
            "user_id": str(self._user_id),
            "limit": limit,
            "offset": offset,
        }
        filters = ["user_id = :user_id", "is_deleted = false"]

        if status:
            filters.append("status = :status")
            params["status"] = status.value
        if priority:
            filters.append("priority = :priority")
            params["priority"] = priority.value
        if due_before:
            filters.append("due_date <= :due_before")
            params["due_before"] = due_before

        where_clause = " AND ".join(filters)
        result = await self._db.execute(
            sa_text(
                f"""
                SELECT id, user_id, title, description, priority, status,
                       due_date, tags, source, created_at, updated_at, completed_at
                  FROM tasks
                 WHERE {where_clause}
                 ORDER BY due_date ASC NULLS LAST, created_at ASC
                 LIMIT :limit OFFSET :offset
                """
            ),
            params,
        )
        return [self._row_to_task(row) for row in result.fetchall()]

    async def update_task(
        self,
        task_id: UUID,
        update_data: dict[str, Any],
    ) -> Task:
        """
        Apply partial updates to an existing task.

        Args:
            task_id:     UUID of the task to update.
            update_data: Dict of field names → new values.

        Returns:
            Updated Task.

        Raises:
            NotFoundException: If the task doesn't exist or belongs to another user.
        """
        allowed_fields = {
            "title", "description", "priority", "status",
            "due_date", "tags", "source",
        }
        safe = {k: v for k, v in update_data.items() if k in allowed_fields}
        if not safe:
            return await self._get_task_or_raise(task_id)

        set_clauses: list[str] = []
        params: dict[str, Any] = {
            "id": str(task_id),
            "user_id": str(self._user_id),
            "now": datetime.now(timezone.utc),
        }

        for key, val in safe.items():
            set_clauses.append(f"{key} = :{key}")
            if key == "tags" and isinstance(val, list):
                params[key] = json.dumps(val)
            elif key == "priority" and isinstance(val, TaskPriority):
                params[key] = val.value
            elif key == "status" and isinstance(val, TaskStatus):
                params[key] = val.value
            else:
                params[key] = val

        set_clauses.append("updated_at = :now")
        set_str = ", ".join(set_clauses)

        result = await self._db.execute(
            sa_text(
                f"""
                UPDATE tasks
                   SET {set_str}
                 WHERE id = :id AND user_id = :user_id AND is_deleted = false
                RETURNING id, user_id, title, description, priority, status,
                          due_date, tags, source, created_at, updated_at, completed_at
                """
            ),
            params,
        )
        row = result.fetchone()
        if not row:
            raise NotFoundException(
                message=f"Task {task_id} not found.",
                code="TASK_NOT_FOUND",
                details={"task_id": str(task_id)},
            )
        await self._db.flush()
        logger.info("update_task: id=%s fields=%s", task_id, list(safe.keys()))
        return self._row_to_task(row)

    async def complete_task(self, task_id: UUID) -> Task:
        """
        Mark a task as completed.

        Args:
            task_id: UUID of the task to complete.

        Returns:
            Updated Task with status=DONE.
        """
        now = datetime.now(timezone.utc)
        result = await self._db.execute(
            sa_text(
                """
                UPDATE tasks
                   SET status       = :done,
                       completed_at = :now,
                       updated_at   = :now
                 WHERE id       = :id
                   AND user_id  = :user_id
                   AND is_deleted = false
                RETURNING id, user_id, title, description, priority, status,
                          due_date, tags, source, created_at, updated_at, completed_at
                """
            ),
            {
                "id": str(task_id),
                "user_id": str(self._user_id),
                "done": TaskStatus.DONE.value,
                "now": now,
            },
        )
        row = result.fetchone()
        if not row:
            raise NotFoundException(
                message=f"Task {task_id} not found.",
                code="TASK_NOT_FOUND",
            )
        await self._db.flush()
        logger.info("complete_task: id=%s", task_id)
        return self._row_to_task(row)

    async def delete_task(self, task_id: UUID) -> bool:
        """
        Soft-delete a task (sets is_deleted=true).

        Args:
            task_id: UUID of the task to delete.

        Returns:
            True on success, False if not found.
        """
        result = await self._db.execute(
            sa_text(
                """
                UPDATE tasks
                   SET is_deleted = true,
                       updated_at = :now
                 WHERE id      = :id
                   AND user_id = :user_id
                RETURNING id
                """
            ),
            {
                "id": str(task_id),
                "user_id": str(self._user_id),
                "now": datetime.now(timezone.utc),
            },
        )
        row = result.fetchone()
        if not row:
            return False
        await self._db.flush()
        logger.info("delete_task: id=%s", task_id)
        return True

    # ------------------------------------------------------------------
    # AI-powered helpers
    # ------------------------------------------------------------------

    async def extract_tasks_from_text(self, text: str) -> list[TaskCreate]:
        """
        Use AI to parse tasks from free-form natural language text.

        Args:
            text: Raw text (e.g. from a voice transcript or Telegram message).

        Returns:
            List of TaskCreate objects ready to be persisted.

        Raises:
            AIException: If the AI client is not configured.
        """
        if self._ai is None:
            raise AIException(
                message="AI agent not configured for TaskService.",
                code="AI_NOT_CONFIGURED",
            )
        if not text.strip():
            return []

        try:
            response = await self._ai.chat_completion(
                messages=[
                    {"role": "system", "content": _EXTRACT_TASKS_SYSTEM},
                    {"role": "user", "content": text[:4000]},
                ],
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            raw: str = response if isinstance(response, str) else ""
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
            parsed = json.loads(cleaned)

            items: list[Any] = []
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                for key in ("tasks", "items", "results"):
                    if isinstance(parsed.get(key), list):
                        items = parsed[key]
                        break

            creates: list[TaskCreate] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                if not title:
                    continue

                due_date: datetime | None = None
                if item.get("due_date"):
                    try:
                        due_date = datetime.fromisoformat(
                            str(item["due_date"]).replace("Z", "+00:00")
                        )
                        if due_date.tzinfo is None:
                            due_date = due_date.replace(tzinfo=timezone.utc)
                    except ValueError:
                        pass

                priority_str = str(item.get("priority", "medium")).lower()
                priority = TaskPriority(priority_str) if priority_str in TaskPriority._value2member_map_ else TaskPriority.MEDIUM

                creates.append(
                    TaskCreate(
                        title=title[:200],
                        description=str(item.get("description", ""))[:1000],
                        priority=priority,
                        due_date=due_date,
                        tags=list(item.get("tags", [])),
                        source="ai",
                    )
                )

            logger.info(
                "extract_tasks_from_text: found=%d tasks in text len=%d",
                len(creates), len(text),
            )
            return creates

        except json.JSONDecodeError as exc:
            logger.warning("extract_tasks_from_text JSON error: %s", exc)
            return []
        except Exception as exc:  # noqa: BLE001
            logger.error("extract_tasks_from_text AI call failed: %s", exc)
            return []

    async def get_overdue_tasks(self) -> list[Task]:
        """
        Return all non-completed tasks whose due_date is in the past.

        Returns:
            List of overdue Task objects.
        """
        now = datetime.now(timezone.utc)
        result = await self._db.execute(
            sa_text(
                """
                SELECT id, user_id, title, description, priority, status,
                       due_date, tags, source, created_at, updated_at, completed_at
                  FROM tasks
                 WHERE user_id    = :user_id
                   AND is_deleted = false
                   AND status NOT IN (:done, :cancelled)
                   AND due_date IS NOT NULL
                   AND due_date < :now
                 ORDER BY due_date ASC
                """
            ),
            {
                "user_id": str(self._user_id),
                "done": TaskStatus.DONE.value,
                "cancelled": TaskStatus.CANCELLED.value,
                "now": now,
            },
        )
        tasks = [self._row_to_task(row) for row in result.fetchall()]
        logger.debug("get_overdue_tasks: found=%d", len(tasks))
        return tasks

    async def get_upcoming_reminders(self, hours_ahead: int = 24) -> list[Task]:
        """
        Return pending tasks due within the next ``hours_ahead`` hours.

        Args:
            hours_ahead: Look-ahead window in hours (default 24).

        Returns:
            List of upcoming Task objects ordered by due_date.
        """
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)
        result = await self._db.execute(
            sa_text(
                """
                SELECT id, user_id, title, description, priority, status,
                       due_date, tags, source, created_at, updated_at, completed_at
                  FROM tasks
                 WHERE user_id    = :user_id
                   AND is_deleted = false
                   AND status NOT IN (:done, :cancelled)
                   AND due_date IS NOT NULL
                   AND due_date >= :now
                   AND due_date <= :cutoff
                 ORDER BY due_date ASC
                """
            ),
            {
                "user_id": str(self._user_id),
                "done": TaskStatus.DONE.value,
                "cancelled": TaskStatus.CANCELLED.value,
                "now": now,
                "cutoff": cutoff,
            },
        )
        tasks = [self._row_to_task(row) for row in result.fetchall()]
        logger.debug(
            "get_upcoming_reminders: hours=%d found=%d", hours_ahead, len(tasks)
        )
        return tasks

    async def get_daily_summary(self) -> dict[str, Any]:
        """
        Return task statistics + today's task list for a daily briefing.

        Returns:
            Dict with counts, overdue list, and upcoming list.
        """
        now = datetime.now(timezone.utc)
        today_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)

        # Counts by status
        count_result = await self._db.execute(
            sa_text(
                """
                SELECT status, COUNT(*) AS cnt
                  FROM tasks
                 WHERE user_id = :user_id AND is_deleted = false
                 GROUP BY status
                """
            ),
            {"user_id": str(self._user_id)},
        )
        counts: dict[str, int] = {row[0]: row[1] for row in count_result.fetchall()}

        overdue = await self.get_overdue_tasks()
        upcoming = await self.get_upcoming_reminders(hours_ahead=24)
        all_tasks = await self.get_tasks(limit=50)

        return {
            "date": now.date().isoformat(),
            "counts": {
                "total": sum(counts.values()),
                "todo": counts.get(TaskStatus.TODO.value, 0),
                "in_progress": counts.get(TaskStatus.IN_PROGRESS.value, 0),
                "done": counts.get(TaskStatus.DONE.value, 0),
                "cancelled": counts.get(TaskStatus.CANCELLED.value, 0),
            },
            "overdue": [self._task_to_dict(t) for t in overdue],
            "upcoming_24h": [self._task_to_dict(t) for t in upcoming],
            "all_tasks": [self._task_to_dict(t) for t in all_tasks],
        }

    async def search_tasks(self, query: str) -> list[Task]:
        """
        Full-text search over task titles and descriptions.

        Performs a PostgreSQL ``ILIKE`` search first; optionally ranks results
        using AI if a client is available.

        Args:
            query: Search string.

        Returns:
            List of matching Task objects.
        """
        if not query.strip():
            return []

        like_pattern = f"%{query.strip()}%"
        result = await self._db.execute(
            sa_text(
                """
                SELECT id, user_id, title, description, priority, status,
                       due_date, tags, source, created_at, updated_at, completed_at
                  FROM tasks
                 WHERE user_id    = :user_id
                   AND is_deleted = false
                   AND (
                         title       ILIKE :pattern
                      OR description ILIKE :pattern
                   )
                 ORDER BY updated_at DESC
                 LIMIT 50
                """
            ),
            {"user_id": str(self._user_id), "pattern": like_pattern},
        )
        tasks = [self._row_to_task(row) for row in result.fetchall()]
        logger.debug("search_tasks: query=%r found=%d", query, len(tasks))
        return tasks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get_task_or_raise(self, task_id: UUID) -> Task:
        result = await self._db.execute(
            sa_text(
                """
                SELECT id, user_id, title, description, priority, status,
                       due_date, tags, source, created_at, updated_at, completed_at
                  FROM tasks
                 WHERE id = :id AND user_id = :user_id AND is_deleted = false
                """
            ),
            {"id": str(task_id), "user_id": str(self._user_id)},
        )
        row = result.fetchone()
        if not row:
            raise NotFoundException(
                message=f"Task {task_id} not found.",
                code="TASK_NOT_FOUND",
            )
        return self._row_to_task(row)

    @staticmethod
    def _row_to_task(row: Any) -> Task:
        """Convert a DB row (mapping or positional) to a Task dataclass."""

        def _val(key: str, idx: int) -> Any:
            try:
                return row[key]
            except (TypeError, KeyError):
                return row[idx]

        raw_tags = _val("tags", 7)
        if isinstance(raw_tags, str):
            try:
                tags = json.loads(raw_tags)
            except (json.JSONDecodeError, TypeError):
                tags = []
        elif isinstance(raw_tags, list):
            tags = raw_tags
        else:
            tags = []

        priority_val = _val("priority", 4)
        try:
            priority = TaskPriority(priority_val)
        except ValueError:
            priority = TaskPriority.MEDIUM

        status_val = _val("status", 5)
        try:
            status = TaskStatus(status_val)
        except ValueError:
            status = TaskStatus.TODO

        return Task(
            id=UUID(str(_val("id", 0))),
            user_id=UUID(str(_val("user_id", 1))),
            title=str(_val("title", 2)),
            description=str(_val("description", 3) or ""),
            priority=priority,
            status=status,
            due_date=_val("due_date", 6),
            tags=tags,
            source=str(_val("source", 8) or "manual"),
            created_at=_val("created_at", 9),
            updated_at=_val("updated_at", 10),
            completed_at=_val("completed_at", 11),
        )

    @staticmethod
    def _task_to_dict(task: Task) -> dict[str, Any]:
        return {
            "id": str(task.id),
            "user_id": str(task.user_id),
            "title": task.title,
            "description": task.description,
            "priority": task.priority.value,
            "status": task.status.value,
            "due_date": task.due_date.isoformat() if task.due_date else None,
            "tags": task.tags,
            "source": task.source,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        }
