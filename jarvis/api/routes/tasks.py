"""
Task management routes for JARVIS.

Prefix: /api/tasks

Endpoints
---------
POST   /                      — create task
GET    /                      — list tasks (filterable)
GET    /search                — keyword search
GET    /{task_id}             — get task
PUT    /{task_id}             — update task
DELETE /{task_id}             — soft-delete task
POST   /{task_id}/complete    — mark task complete
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.events import EventType, event_bus
from core.logger import get_logger
from memory.models import Task, TaskPriority, TaskStatus

log = get_logger(__name__)

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


# --------------------------------------------------------------------------- #
# Pydantic schemas                                                               #
# --------------------------------------------------------------------------- #


class TaskCreate(BaseModel):
    """Request body for creating a task."""

    title: str = Field(..., min_length=1, max_length=512, description="Task title")
    description: str | None = Field(None, description="Optional longer description")
    priority: TaskPriority = Field(TaskPriority.MEDIUM, description="Task priority")
    due_date: datetime | None = Field(None, description="Optional due date (ISO-8601)")
    source: str | None = Field(None, max_length=64, description="Origin of the task")


class TaskUpdate(BaseModel):
    """Request body for updating a task (all fields optional)."""

    title: str | None = Field(None, min_length=1, max_length=512)
    description: str | None = None
    priority: TaskPriority | None = None
    status: TaskStatus | None = None
    due_date: datetime | None = None


class TaskResponse(BaseModel):
    """API response shape for a single task."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: str | None
    status: TaskStatus
    priority: TaskPriority
    due_date: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime
    is_deleted: bool
    source: str | None


# --------------------------------------------------------------------------- #
# Helpers                                                                        #
# --------------------------------------------------------------------------- #


async def _get_task_or_404(task_id: int, db: AsyncSession) -> Task:
    """Fetch a non-deleted task by ID or raise HTTP 404."""
    result = await db.execute(
        select(Task).where(Task.id == task_id, Task.is_deleted.is_(False))
    )
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )
    return task


# --------------------------------------------------------------------------- #
# POST /api/tasks/                                                               #
# --------------------------------------------------------------------------- #


@router.post(
    "/",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a task",
)
async def create_task(
    payload: TaskCreate,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Create a new task and persist it to the database."""
    task = Task(
        title=payload.title,
        description=payload.description,
        priority=payload.priority,
        status=TaskStatus.PENDING,
        due_date=payload.due_date,
        source=payload.source,
        is_deleted=False,
    )
    db.add(task)
    await db.flush()  # Populate task.id before commit.
    await db.refresh(task)

    log.info("Task created: id={id} title={title!r}", id=task.id, title=task.title)

    await event_bus.publish(
        EventType.TASK_CREATED,
        {"task_id": task.id, "title": task.title, "priority": task.priority.value},
    )

    return task


# --------------------------------------------------------------------------- #
# GET /api/tasks/                                                                #
# --------------------------------------------------------------------------- #


@router.get(
    "/",
    response_model=list[TaskResponse],
    summary="List tasks",
)
async def list_tasks(
    task_status: TaskStatus | None = Query(None, alias="status"),
    priority: TaskPriority | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Return a paginated list of tasks, with optional status/priority filters."""
    query = select(Task).where(Task.is_deleted.is_(False))

    if task_status is not None:
        query = query.where(Task.status == task_status)
    if priority is not None:
        query = query.where(Task.priority == priority)

    query = query.order_by(Task.created_at.desc()).limit(limit).offset(offset)

    result = await db.execute(query)
    return result.scalars().all()


# --------------------------------------------------------------------------- #
# GET /api/tasks/search                                                          #
# --------------------------------------------------------------------------- #


@router.get(
    "/search",
    response_model=list[TaskResponse],
    summary="Search tasks by keyword",
)
async def search_tasks(
    q: str = Query(..., min_length=1, description="Keyword to search in title/description"),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Full-text keyword search across task title and description fields."""
    like_term = f"%{q}%"
    query = (
        select(Task)
        .where(
            Task.is_deleted.is_(False),
            or_(
                Task.title.ilike(like_term),
                Task.description.ilike(like_term),
            ),
        )
        .order_by(Task.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(query)
    return result.scalars().all()


# --------------------------------------------------------------------------- #
# GET /api/tasks/{task_id}                                                       #
# --------------------------------------------------------------------------- #


@router.get(
    "/{task_id}",
    response_model=TaskResponse,
    summary="Get a task by ID",
)
async def get_task(
    task_id: int,
    db: AsyncSession = Depends(get_db),
) -> Any:
    return await _get_task_or_404(task_id, db)


# --------------------------------------------------------------------------- #
# PUT /api/tasks/{task_id}                                                       #
# --------------------------------------------------------------------------- #


@router.put(
    "/{task_id}",
    response_model=TaskResponse,
    summary="Update a task",
)
async def update_task(
    task_id: int,
    payload: TaskUpdate,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Partially update a task.  Only fields included in the request are changed."""
    task = await _get_task_or_404(task_id, db)

    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(task, field, value)

    await db.flush()
    await db.refresh(task)
    log.info("Task updated: id={id}", id=task.id)
    return task


# --------------------------------------------------------------------------- #
# DELETE /api/tasks/{task_id}                                                    #
# --------------------------------------------------------------------------- #


@router.delete(
    "/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft-delete a task",
)
async def delete_task(
    task_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Mark a task as deleted (soft delete)."""
    task = await _get_task_or_404(task_id, db)
    task.is_deleted = True
    task.status = TaskStatus.DELETED
    await db.flush()
    log.info("Task soft-deleted: id={id}", id=task.id)


# --------------------------------------------------------------------------- #
# POST /api/tasks/{task_id}/complete                                             #
# --------------------------------------------------------------------------- #


@router.post(
    "/{task_id}/complete",
    response_model=TaskResponse,
    summary="Mark a task complete",
)
async def complete_task(
    task_id: int,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Set task status to COMPLETED and record the completion timestamp."""
    task = await _get_task_or_404(task_id, db)

    if task.status == TaskStatus.COMPLETED:
        return task  # idempotent

    task.status = TaskStatus.COMPLETED
    task.completed_at = datetime.utcnow()

    await db.flush()
    await db.refresh(task)

    log.info("Task completed: id={id} title={title!r}", id=task.id, title=task.title)

    await event_bus.publish(
        EventType.TASK_COMPLETED,
        {"task_id": task.id, "title": task.title},
    )

    return task
