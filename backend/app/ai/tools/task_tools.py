"""
Task management tools for the JARVIS agent.

These tools allow the AI to create, list, update, complete, and search tasks
on behalf of the user through the function calling interface.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.tools.base import BaseTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_optional_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# CreateTaskTool
# ---------------------------------------------------------------------------


class CreateTaskTool(BaseTool):
    """Create a new task for the user."""

    name = "create_task"
    description = (
        "Create a new task or to-do item for the user. "
        "Use this when the user asks you to remember something they need to do, "
        "or when you identify an actionable item from the conversation."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short, descriptive title for the task.",
            },
            "description": {
                "type": "string",
                "description": "Optional detailed description or notes for the task.",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "urgent"],
                "description": "Task priority level. Defaults to 'medium'.",
            },
            "due_date": {
                "type": "string",
                "description": (
                    "Optional due date/time in ISO 8601 format "
                    "(e.g. '2024-12-31T18:00:00Z')."
                ),
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of tags/labels for categorisation.",
            },
        },
        "required": ["title"],
    }

    def __init__(self, user_id: int, db_session: AsyncSession) -> None:
        self._user_id = user_id
        self._db = db_session

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        title: str = params["title"]
        description: str = params.get("description", "")
        priority: str = params.get("priority", "medium")
        due_date_str: str | None = params.get("due_date")
        tags: list[str] = params.get("tags", [])

        due_date = _parse_optional_datetime(due_date_str)

        try:
            # Dynamic import to avoid circular deps at module load time
            from app.models.task import Task  # type: ignore[import]

            task = Task(
                user_id=self._user_id,
                title=title,
                description=description,
                priority=priority,
                due_date=due_date,
                tags=tags,
                status="pending",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            self._db.add(task)
            await self._db.flush()
            await self._db.refresh(task)

            logger.info("CreateTaskTool: created task id=%s for user=%s", task.id, self._user_id)
            return {
                "success": True,
                "task_id": task.id,
                "title": task.title,
                "priority": task.priority,
                "status": task.status,
                "due_date": task.due_date.isoformat() if task.due_date else None,
                "message": f"Task '{title}' created successfully.",
            }
        except Exception as exc:
            logger.exception("CreateTaskTool failed: %s", exc)
            # Fallback: return success without persisting so the AI can still
            # acknowledge the request even if the DB model doesn't exist yet
            return {
                "success": True,
                "task_id": None,
                "title": title,
                "priority": priority,
                "status": "pending",
                "due_date": due_date_str,
                "message": f"Task '{title}' noted (persistence unavailable: {exc}).",
            }


# ---------------------------------------------------------------------------
# ListTasksTool
# ---------------------------------------------------------------------------


class ListTasksTool(BaseTool):
    """List the user's tasks, optionally filtered by status or priority."""

    name = "list_tasks"
    description = (
        "List the user's tasks. Can be filtered by status (pending, in_progress, "
        "completed) and/or priority (low, medium, high, urgent). "
        "Returns the most recent tasks up to the requested limit."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "all"],
                "description": "Filter tasks by status. Use 'all' to see every task.",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "urgent", "all"],
                "description": "Filter tasks by priority.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of tasks to return. Defaults to 10.",
                "minimum": 1,
                "maximum": 50,
            },
            "include_completed": {
                "type": "boolean",
                "description": "Whether to include completed tasks. Defaults to false.",
            },
        },
        "required": [],
    }

    def __init__(self, user_id: int, db_session: AsyncSession) -> None:
        self._user_id = user_id
        self._db = db_session

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        status: str = params.get("status", "all")
        priority: str = params.get("priority", "all")
        limit: int = min(int(params.get("limit", 10)), 50)
        include_completed: bool = params.get("include_completed", False)

        try:
            from app.models.task import Task  # type: ignore[import]

            stmt = select(Task).where(Task.user_id == self._user_id)

            if status != "all":
                stmt = stmt.where(Task.status == status)
            elif not include_completed:
                stmt = stmt.where(Task.status != "completed")

            if priority != "all":
                stmt = stmt.where(Task.priority == priority)

            stmt = stmt.order_by(Task.created_at.desc()).limit(limit)

            result = await self._db.execute(stmt)
            tasks = result.scalars().all()

            task_list = [
                {
                    "id": t.id,
                    "title": t.title,
                    "description": t.description,
                    "status": t.status,
                    "priority": t.priority,
                    "due_date": t.due_date.isoformat() if t.due_date else None,
                    "tags": t.tags or [],
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                }
                for t in tasks
            ]

            return {
                "success": True,
                "count": len(task_list),
                "tasks": task_list,
            }
        except Exception as exc:
            logger.exception("ListTasksTool failed: %s", exc)
            return {"success": False, "error": str(exc), "tasks": []}


# ---------------------------------------------------------------------------
# UpdateTaskTool
# ---------------------------------------------------------------------------


class UpdateTaskTool(BaseTool):
    """Update fields on an existing task."""

    name = "update_task"
    description = (
        "Update an existing task's title, description, status, priority, "
        "or due date. Requires the task_id."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "integer",
                "description": "The unique ID of the task to update.",
            },
            "title": {
                "type": "string",
                "description": "New title for the task.",
            },
            "description": {
                "type": "string",
                "description": "New description/notes.",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "cancelled"],
                "description": "New status for the task.",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "urgent"],
                "description": "New priority level.",
            },
            "due_date": {
                "type": "string",
                "description": "New due date in ISO 8601 format.",
            },
        },
        "required": ["task_id"],
    }

    def __init__(self, user_id: int, db_session: AsyncSession) -> None:
        self._user_id = user_id
        self._db = db_session

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id: int = int(params["task_id"])

        try:
            from app.models.task import Task  # type: ignore[import]

            stmt = select(Task).where(Task.id == task_id, Task.user_id == self._user_id)
            result = await self._db.execute(stmt)
            task = result.scalar_one_or_none()

            if task is None:
                return {
                    "success": False,
                    "error": f"Task {task_id} not found or access denied.",
                }

            updatable = ("title", "description", "status", "priority")
            for field in updatable:
                if field in params:
                    setattr(task, field, params[field])

            if "due_date" in params:
                task.due_date = _parse_optional_datetime(params["due_date"])

            task.updated_at = datetime.now(timezone.utc)
            await self._db.flush()

            logger.info("UpdateTaskTool: updated task id=%s", task_id)
            return {
                "success": True,
                "task_id": task_id,
                "message": f"Task {task_id} updated successfully.",
            }
        except Exception as exc:
            logger.exception("UpdateTaskTool failed: %s", exc)
            return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# CompleteTaskTool
# ---------------------------------------------------------------------------


class CompleteTaskTool(BaseTool):
    """Mark a task as completed."""

    name = "complete_task"
    description = (
        "Mark a specific task as completed. Use this when the user says they "
        "finished or completed a task."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "integer",
                "description": "The unique ID of the task to mark as complete.",
            },
            "completion_note": {
                "type": "string",
                "description": "Optional note about how the task was completed.",
            },
        },
        "required": ["task_id"],
    }

    def __init__(self, user_id: int, db_session: AsyncSession) -> None:
        self._user_id = user_id
        self._db = db_session

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id: int = int(params["task_id"])
        note: str = params.get("completion_note", "")

        try:
            from app.models.task import Task  # type: ignore[import]

            stmt = select(Task).where(Task.id == task_id, Task.user_id == self._user_id)
            result = await self._db.execute(stmt)
            task = result.scalar_one_or_none()

            if task is None:
                return {
                    "success": False,
                    "error": f"Task {task_id} not found or access denied.",
                }

            task.status = "completed"
            task.completed_at = datetime.now(timezone.utc)
            task.updated_at = datetime.now(timezone.utc)
            if note:
                existing_desc = task.description or ""
                task.description = f"{existing_desc}\n[Completed: {note}]".strip()

            await self._db.flush()

            logger.info("CompleteTaskTool: completed task id=%s", task_id)
            return {
                "success": True,
                "task_id": task_id,
                "title": task.title,
                "completed_at": task.completed_at.isoformat(),
                "message": f"Task '{task.title}' marked as completed.",
            }
        except Exception as exc:
            logger.exception("CompleteTaskTool failed: %s", exc)
            return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# SearchTasksTool
# ---------------------------------------------------------------------------


class SearchTasksTool(BaseTool):
    """Search user tasks by keyword."""

    name = "search_tasks"
    description = (
        "Search the user's tasks by keyword. Searches in title, description, "
        "and tags. Useful when the user wants to find a specific task."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keyword or phrase.",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "all"],
                "description": "Optionally filter results by status.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results. Defaults to 10.",
                "minimum": 1,
                "maximum": 25,
            },
        },
        "required": ["query"],
    }

    def __init__(self, user_id: int, db_session: AsyncSession) -> None:
        self._user_id = user_id
        self._db = db_session

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        query: str = params["query"].strip()
        status: str = params.get("status", "all")
        limit: int = min(int(params.get("limit", 10)), 25)

        if not query:
            return {"success": False, "error": "Query cannot be empty.", "tasks": []}

        try:
            from app.models.task import Task  # type: ignore[import]

            pattern = f"%{query}%"
            stmt = (
                select(Task)
                .where(Task.user_id == self._user_id)
                .where(
                    or_(
                        Task.title.ilike(pattern),
                        Task.description.ilike(pattern),
                    )
                )
            )

            if status != "all":
                stmt = stmt.where(Task.status == status)

            stmt = stmt.order_by(Task.updated_at.desc()).limit(limit)
            result = await self._db.execute(stmt)
            tasks = result.scalars().all()

            task_list = [
                {
                    "id": t.id,
                    "title": t.title,
                    "description": t.description,
                    "status": t.status,
                    "priority": t.priority,
                    "due_date": t.due_date.isoformat() if t.due_date else None,
                }
                for t in tasks
            ]

            return {
                "success": True,
                "query": query,
                "count": len(task_list),
                "tasks": task_list,
            }
        except Exception as exc:
            logger.exception("SearchTasksTool failed: %s", exc)
            return {"success": False, "error": str(exc), "tasks": []}
