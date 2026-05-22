"""
Task management endpoints for JARVIS.

POST   /tasks                         – create a task
GET    /tasks                         – list tasks (filters: status, priority, due_date)
GET    /tasks/{task_id}               – get a single task
PUT    /tasks/{task_id}               – update a task
DELETE /tasks/{task_id}              – delete a task
POST   /tasks/{task_id}/complete     – mark a task complete
POST   /tasks/extract                – extract tasks from free text using AI
GET    /tasks/reminders/upcoming     – upcoming reminders
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field, model_validator

from app.core.exceptions import AIException, NotFoundException, ValidationException
from app.core.logging_config import get_logger
from app.utils.helpers import generate_id, now_utc

logger = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    cancelled = "cancelled"


class TaskPriority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = Field(default=None, max_length=5000)
    priority: TaskPriority = Field(default=TaskPriority.medium)
    due_date: Optional[datetime] = None
    reminder_at: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def reminder_before_due(self) -> "TaskCreate":
        if self.reminder_at and self.due_date:
            if self.reminder_at > self.due_date:
                raise ValueError("reminder_at must be before due_date")
        return self


class TaskUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=500)
    description: Optional[str] = Field(default=None, max_length=5000)
    priority: Optional[TaskPriority] = None
    status: Optional[TaskStatus] = None
    due_date: Optional[datetime] = None
    reminder_at: Optional[datetime] = None
    tags: Optional[List[str]] = None


class TaskOut(BaseModel):
    id: str
    title: str
    description: Optional[str]
    priority: TaskPriority
    status: TaskStatus
    due_date: Optional[datetime]
    reminder_at: Optional[datetime]
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]


class TaskListResponse(BaseModel):
    tasks: List[TaskOut]
    total: int
    page: int
    page_size: int


class ExtractTasksRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)


class ExtractTasksResponse(BaseModel):
    tasks: List[TaskCreate]
    source_text_len: int
    extracted_count: int


class UpcomingRemindersResponse(BaseModel):
    reminders: List[TaskOut]
    total: int


# ---------------------------------------------------------------------------
# In-process task store  (swap for SQLAlchemy repo in production)
# ---------------------------------------------------------------------------

_tasks: Dict[str, Dict[str, Any]] = {}


def _task_to_out(t: Dict[str, Any]) -> TaskOut:
    return TaskOut(
        id=t["task_id"],
        title=t["title"],
        description=t.get("description"),
        priority=TaskPriority(t["priority"]),
        status=TaskStatus(t["status"]),
        due_date=datetime.fromisoformat(t["due_date"]) if t.get("due_date") else None,
        reminder_at=(
            datetime.fromisoformat(t["reminder_at"]) if t.get("reminder_at") else None
        ),
        tags=t.get("tags", []),
        created_at=datetime.fromisoformat(t["created_at"]),
        updated_at=datetime.fromisoformat(t["updated_at"]),
        completed_at=(
            datetime.fromisoformat(t["completed_at"]) if t.get("completed_at") else None
        ),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=TaskOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new task",
)
async def create_task(payload: TaskCreate) -> TaskOut:
    task_id = generate_id()
    now = now_utc()
    record: Dict[str, Any] = {
        "task_id": task_id,
        "title": payload.title,
        "description": payload.description,
        "priority": payload.priority.value,
        "status": TaskStatus.pending.value,
        "due_date": payload.due_date.isoformat() if payload.due_date else None,
        "reminder_at": payload.reminder_at.isoformat() if payload.reminder_at else None,
        "tags": payload.tags,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "completed_at": None,
    }
    _tasks[task_id] = record
    logger.info("task_created", task_id=task_id, title=payload.title)
    return _task_to_out(record)


@router.get(
    "",
    response_model=TaskListResponse,
    status_code=status.HTTP_200_OK,
    summary="List tasks with optional filters",
)
async def list_tasks(
    task_status: Optional[TaskStatus] = Query(default=None, alias="status"),
    priority: Optional[TaskPriority] = Query(default=None),
    due_before: Optional[datetime] = Query(default=None),
    due_after: Optional[datetime] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> TaskListResponse:
    items = list(_tasks.values())

    if task_status:
        items = [t for t in items if t["status"] == task_status.value]
    if priority:
        items = [t for t in items if t["priority"] == priority.value]
    if due_before:
        items = [
            t
            for t in items
            if t.get("due_date")
            and datetime.fromisoformat(t["due_date"]) <= due_before
        ]
    if due_after:
        items = [
            t
            for t in items
            if t.get("due_date")
            and datetime.fromisoformat(t["due_date"]) >= due_after
        ]

    # Sort by due_date ascending (tasks without due date go last)
    items.sort(
        key=lambda t: t.get("due_date") or "9999-12-31T23:59:59"
    )

    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = items[start:end]

    return TaskListResponse(
        tasks=[_task_to_out(t) for t in page_items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/reminders/upcoming",
    response_model=UpcomingRemindersResponse,
    status_code=status.HTTP_200_OK,
    summary="Get upcoming reminders",
)
async def upcoming_reminders(
    within_hours: int = Query(default=24, ge=1, le=168),
) -> UpcomingRemindersResponse:
    """Return tasks with a reminder scheduled within the next *within_hours* hours."""
    now = now_utc()
    results = []
    for t in _tasks.values():
        if t["status"] in (TaskStatus.completed.value, TaskStatus.cancelled.value):
            continue
        reminder_str = t.get("reminder_at")
        if not reminder_str:
            continue
        reminder_dt = datetime.fromisoformat(reminder_str)
        if reminder_dt.tzinfo is None:
            reminder_dt = reminder_dt.replace(tzinfo=timezone.utc)
        delta = (reminder_dt - now).total_seconds()
        if 0 <= delta <= within_hours * 3600:
            results.append(_task_to_out(t))

    results.sort(key=lambda t: t.reminder_at or datetime.max.replace(tzinfo=timezone.utc))
    return UpcomingRemindersResponse(reminders=results, total=len(results))


@router.get(
    "/{task_id}",
    response_model=TaskOut,
    status_code=status.HTTP_200_OK,
    summary="Get a task by ID",
)
async def get_task(task_id: str) -> TaskOut:
    if task_id not in _tasks:
        raise NotFoundException(
            message=f"Task '{task_id}' not found.",
            details={"task_id": task_id},
        )
    return _task_to_out(_tasks[task_id])


@router.put(
    "/{task_id}",
    response_model=TaskOut,
    status_code=status.HTTP_200_OK,
    summary="Update a task",
)
async def update_task(task_id: str, payload: TaskUpdate) -> TaskOut:
    if task_id not in _tasks:
        raise NotFoundException(
            message=f"Task '{task_id}' not found.",
            details={"task_id": task_id},
        )
    record = _tasks[task_id]
    updates = payload.model_dump(exclude_unset=True)

    for field, value in updates.items():
        if isinstance(value, datetime):
            record[field] = value.isoformat()
        elif isinstance(value, Enum):
            record[field] = value.value
        elif value is not None or field in updates:
            record[field] = value

    record["updated_at"] = now_utc().isoformat()
    logger.info("task_updated", task_id=task_id, fields=list(updates.keys()))
    return _task_to_out(record)


@router.delete(
    "/{task_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a task",
)
async def delete_task(task_id: str) -> Dict[str, Any]:
    if task_id not in _tasks:
        raise NotFoundException(
            message=f"Task '{task_id}' not found.",
            details={"task_id": task_id},
        )
    del _tasks[task_id]
    logger.info("task_deleted", task_id=task_id)
    return {"status": "deleted", "task_id": task_id}


@router.post(
    "/{task_id}/complete",
    response_model=TaskOut,
    status_code=status.HTTP_200_OK,
    summary="Mark a task as completed",
)
async def complete_task(task_id: str) -> TaskOut:
    if task_id not in _tasks:
        raise NotFoundException(
            message=f"Task '{task_id}' not found.",
            details={"task_id": task_id},
        )
    record = _tasks[task_id]
    if record["status"] == TaskStatus.completed.value:
        raise ValidationException(
            message="Task is already completed.",
            details={"task_id": task_id},
        )
    now = now_utc()
    record["status"] = TaskStatus.completed.value
    record["completed_at"] = now.isoformat()
    record["updated_at"] = now.isoformat()
    logger.info("task_completed", task_id=task_id)
    return _task_to_out(record)


@router.post(
    "/extract",
    response_model=ExtractTasksResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract tasks from free-form text using AI",
)
async def extract_tasks(payload: ExtractTasksRequest) -> ExtractTasksResponse:
    """Use the LLM to identify and structure tasks within *text*."""
    system_prompt = (
        "You are a task extraction assistant. "
        "Given the following text, extract all actionable tasks. "
        "Return a JSON array where each element has:\n"
        '  "title": string (required, max 500 chars)\n'
        '  "description": string or null\n'
        '  "priority": "low" | "medium" | "high" | "urgent"\n'
        '  "due_date": ISO-8601 datetime string or null\n'
        "Return ONLY the JSON array, nothing else."
    )

    raw_json: str = "[]"
    try:
        from openai import AsyncOpenAI  # type: ignore
        from app.core.config import get_settings

        cfg = get_settings()
        client = AsyncOpenAI(api_key=cfg.OPENAI_API_KEY)
        resp = await client.chat.completions.create(
            model=cfg.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload.text},
            ],
            max_tokens=2048,
            stream=False,
        )
        raw_json = resp.choices[0].message.content or "[]"
    except ImportError:
        # Mock response when openai is not installed
        raw_json = json.dumps(
            [
                {
                    "title": "Sample extracted task",
                    "description": "Extracted from text",
                    "priority": "medium",
                    "due_date": None,
                }
            ]
        )
    except Exception as exc:
        raise AIException(
            message="Failed to extract tasks from text.",
            details={"error": str(exc)},
        ) from exc

    from app.utils.helpers import extract_json_from_text

    parsed = extract_json_from_text(raw_json)
    if not isinstance(parsed, list):
        parsed = []

    extracted: List[TaskCreate] = []
    for item in parsed:
        try:
            due = None
            if item.get("due_date"):
                try:
                    due = datetime.fromisoformat(item["due_date"])
                except (ValueError, TypeError):
                    pass
            extracted.append(
                TaskCreate(
                    title=str(item.get("title", "Untitled"))[:500],
                    description=item.get("description"),
                    priority=TaskPriority(item.get("priority", "medium")),
                    due_date=due,
                )
            )
        except Exception:
            continue

    return ExtractTasksResponse(
        tasks=extracted,
        source_text_len=len(payload.text),
        extracted_count=len(extracted),
    )
