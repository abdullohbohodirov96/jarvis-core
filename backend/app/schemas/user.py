"""
User-related Pydantic v2 schemas.

These schemas are used for request validation, response serialisation, and
internal data transfer.  They deliberately omit ``hashed_password`` from all
public-facing responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# ---------------------------------------------------------------------------
# User preferences sub-model
# ---------------------------------------------------------------------------

class UserPreferences(BaseModel):
    """Structured representation of the ``users.preferences`` JSON column."""

    model_config = ConfigDict(extra="allow")  # Forward-compatible: extra keys are kept

    timezone: str = Field(
        default="UTC",
        description="IANA timezone name (e.g. 'Europe/London')",
    )
    language: str = Field(
        default="en",
        description="BCP-47 language tag for UI and AI responses",
    )
    theme: str = Field(
        default="system",
        description="UI colour scheme preference: light | dark | system",
    )
    notifications_enabled: bool = Field(
        default=True,
        description="Master switch for push / Telegram notifications",
    )
    ai_personality: str = Field(
        default="assistant",
        description="AI persona hint: assistant | friendly | concise | detailed",
    )
    default_reminder_minutes: int = Field(
        default=15,
        ge=0,
        description="How many minutes before a task due date to fire the default reminder",
    )

    @field_validator("theme")
    @classmethod
    def validate_theme(cls, v: str) -> str:
        allowed = {"light", "dark", "system"}
        if v not in allowed:
            raise ValueError(f"theme must be one of {allowed}")
        return v


# ---------------------------------------------------------------------------
# User CRUD schemas
# ---------------------------------------------------------------------------

class UserBase(BaseModel):
    """Fields shared by create and update operations."""

    email: EmailStr | None = Field(default=None, description="Primary e-mail address")
    username: str | None = Field(
        default=None,
        min_length=3,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_.-]+$",
        description="Unique display handle (alphanumeric, underscores, dots, hyphens)",
    )
    full_name: str | None = Field(default=None, max_length=255)
    voice_enabled: bool | None = None
    preferences: UserPreferences | None = None


class UserCreate(BaseModel):
    """Schema for POST /users (registration)."""

    model_config = ConfigDict(extra="forbid")

    email: EmailStr = Field(description="Primary e-mail address")
    username: str = Field(
        min_length=3,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_.-]+$",
    )
    password: str = Field(
        min_length=8,
        max_length=128,
        description="Plaintext password; will be hashed before storage",
    )
    full_name: str | None = Field(default=None, max_length=255)


class UserUpdate(UserBase):
    """Schema for PATCH /users/{id} — all fields are optional."""

    model_config = ConfigDict(extra="forbid")

    password: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
        description="New plaintext password; omit to keep existing",
    )
    telegram_username: str | None = Field(default=None, max_length=128)


class UserResponse(BaseModel):
    """Public-facing user representation (no hashed_password)."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: EmailStr
    username: str
    full_name: str | None
    is_active: bool
    is_superuser: bool
    telegram_user_id: int | None
    telegram_username: str | None
    voice_enabled: bool
    preferences: dict[str, Any] | None
    last_seen: datetime | None
    created_at: datetime
    updated_at: datetime


class UserInDB(UserResponse):
    """Internal user representation including the hashed password."""

    hashed_password: str


# ---------------------------------------------------------------------------
# Authentication schemas
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    """Schema for POST /auth/login."""

    model_config = ConfigDict(extra="forbid")

    email: EmailStr = Field(description="Registered e-mail address")
    password: str = Field(description="Plaintext password")


class RefreshTokenRequest(BaseModel):
    """Schema for POST /auth/refresh."""

    model_config = ConfigDict(extra="forbid")

    refresh_token: str = Field(description="Opaque refresh token issued at login")


class Token(BaseModel):
    """OAuth2-compatible token response."""

    access_token: str = Field(description="Short-lived JWT bearer token")
    refresh_token: str = Field(description="Long-lived opaque refresh token")
    token_type: str = Field(default="bearer", description="Always 'bearer'")
    expires_in: int = Field(description="Access token TTL in seconds")


class TokenData(BaseModel):
    """Payload decoded from a JWT access token."""

    user_id: UUID | None = None
    email: str | None = None
    is_superuser: bool = False
