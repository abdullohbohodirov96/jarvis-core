"""
Common reusable Pydantic v2 schemas shared across the application.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# Generic type variable for paginated responses
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

class PaginationParams(BaseModel):
    """Query-parameter schema for paginated list endpoints."""

    model_config = ConfigDict(extra="forbid")

    skip: int = Field(
        default=0,
        ge=0,
        description="Number of items to skip (offset-based pagination)",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Maximum number of items to return per page",
    )


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic wrapper for list endpoints that support pagination."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    items: list[T] = Field(description="Page of results")
    total: int = Field(ge=0, description="Total number of matching rows")
    skip: int = Field(ge=0, description="Number of items skipped")
    limit: int = Field(ge=1, description="Maximum items per page")
    has_more: bool = Field(
        description="True when further pages are available (skip + len(items) < total)"
    )

    @classmethod
    def create(
        cls,
        items: list[T],
        total: int,
        skip: int,
        limit: int,
    ) -> "PaginatedResponse[T]":
        """Convenience constructor that derives *has_more* automatically."""
        return cls(
            items=items,
            total=total,
            skip=skip,
            limit=limit,
            has_more=(skip + len(items)) < total,
        )


# ---------------------------------------------------------------------------
# Simple response envelopes
# ---------------------------------------------------------------------------

class MessageResponse(BaseModel):
    """Generic success/info message response."""

    message: str = Field(description="Human-readable status message")


class IDResponse(BaseModel):
    """Response containing only a created/affected resource ID."""

    id: UUID = Field(description="UUID of the affected resource")


class ErrorResponse(BaseModel):
    """Structured error envelope returned on 4xx / 5xx responses."""

    error: str = Field(description="Short error summary")
    code: str = Field(description="Machine-readable error code (e.g. VALIDATION_ERROR)")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (field errors, hints, etc.)",
    )


# ---------------------------------------------------------------------------
# Health / Status
# ---------------------------------------------------------------------------

class StatusEnum(str, Enum):
    """Service operational status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status of a single infrastructure component."""

    status: StatusEnum
    latency_ms: float | None = Field(
        default=None,
        description="Round-trip latency in milliseconds (null if not measured)",
    )
    detail: str | None = Field(
        default=None,
        description="Optional human-readable note (error message, version, etc.)",
    )


class HealthResponse(BaseModel):
    """Aggregated health report returned by the /health endpoint."""

    status: StatusEnum = Field(description="Overall service status")
    version: str = Field(description="Application version string")
    environment: str = Field(description="Deployment environment name")
    components: dict[str, ComponentHealth] = Field(
        default_factory=dict,
        description="Per-component health breakdown",
    )
