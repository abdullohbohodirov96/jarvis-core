"""
Task-related Pydantic v2 schemas.

Covers task CRUD, AI extraction from free-form text, and reminder queries.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.task import TaskPriority, TaskSource, TaskStatus


# ---------------------------------------------------------------------------
# Task CRUD schemas
# ---------------------------------------------------------------------------

class TaskBase(BaseModel):
    """Common optional fields shared by create and update schemas."""

    description: str | None = None
    priority: TaskPriority | None = None
    due_date: datetime | None = Field(
        default=None,
        description="Timezone-aware deadline; null means no fixed deadline",
    )
    reminder_at: datetime | None = Field(
        default=None,
        description="Timezone-aware reminder timestamp",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Free-form labels for filtering and grouping",
    )
    recurrence: dict[str, Any] | None = Field(
        default=None,
        description="iCalendar-style RRULE stored as a dict; null for one-off tasks",
    )


class TaskCreate(TaskBase):
    """Schema for POST /tasks."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        min_length=1,
        max_length=512,
        description="Short imperative description of the task",
    )
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
    source: TaskSource = Field(
        default=TaskSource.MANUAL,
        description="How the task was created",
    )
    source_message_id: UUID | None = Field(
        default=None,
        description="Message that prompted AI extraction of this task",
    )

    @model_validator(mode="after")
    def reminder_before_due(self) -> "TaskCreate":
        """Reminder must not be after the due date if both are provided."""
        if (
            self.reminder_at is not None
            and self.due_date is not None
            and self.reminder_at > self.due_date
        ):
            raise ValueError("reminder_at must be before or equal to due_date")
        return self


class TaskUpdate(TaskBase):
    """Schema for PATCH /tasks/{id} — all fields optional."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=512)
    status: TaskStatus | None = None
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def auto_completed_at(self) -> "TaskUpdate":
        """Set completed_at automatically when status transitions to COMPLETED."""
        if self.status == TaskStatus.COMPLETED and self.completed_at is None:
            from datetime import timezone
            self.completed_at = datetime.now(timezone.utc)
        return self


class TaskResponse(BaseModel):
    """Serialised task returned to the client."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    title: str
    description: str | None
    status: TaskStatus
    priority: TaskPriority
    due_date: datetime | None
    reminder_at: datetime | None
    reminder_sent: bool
    source: TaskSource
    source_message_id: UUID | None
    tags: list[str] | None
    recurrence: dict[str, Any] | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# AI extraction schemas
# ---------------------------------------------------------------------------

class TaskExtractRequest(BaseModel):
    """
    Request payload for the AI task-extraction endpoint.

    The service will parse *text* and return zero or more structured tasks.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=1,
        max_length=16_000,
        description="Free-form text from which to extract actionable tasks",
    )
    timezone: str = Field(
        default="UTC",
        description="IANA timezone name used to resolve relative dates ('tomorrow', 'next Friday')",
    )


class TaskExtractResponse(BaseModel):
    """AI task-extraction result."""

    tasks: list[TaskCreate] = Field(
        description="Zero or more structured tasks extracted from the input text"
    )
    raw_response: str | None = Field(
        default=None,
        description="Full LLM response for debugging (omitted in production)",
    )


# ---------------------------------------------------------------------------
# Reminder schemas
# ---------------------------------------------------------------------------

class ReminderResponse(BaseModel):
    """Summary of an upcoming or triggered reminder."""

    task_id: UUID
    title: str
    reminder_at: datetime
    due_date: datetime | None
    priority: TaskPriority
    is_overdue: bool = Field(
        description="True when due_date is in the past at the time of this response"
    )
