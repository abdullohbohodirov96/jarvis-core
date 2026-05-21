"""
Application configuration using Pydantic v2 Settings.
All settings are loaded from environment variables or .env file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = Field(default="JARVIS", description="Application name")
    APP_VERSION: str = Field(default="1.0.0", description="Application version")
    DEBUG: bool = Field(default=False, description="Debug mode flag")
    ENVIRONMENT: str = Field(
        default="development",
        description="Deployment environment: development | staging | production",
    )

    # ── Security & JWT ────────────────────────────────────────────────────────
    SECRET_KEY: str = Field(
        default="change-me-in-production-at-least-32-chars-long!!",
        description="Secret key for JWT signing",
    )
    ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, description="Access token TTL in minutes"
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7, description="Refresh token TTL in days"
    )

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://jarvis:jarvis@localhost:5432/jarvis_db",
        description="Async PostgreSQL connection URL",
    )

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    OPENAI_MODEL: str = Field(default="gpt-4o", description="OpenAI model to use")
    OPENAI_MAX_TOKENS: int = Field(
        default=4096, description="Maximum tokens per OpenAI completion"
    )

    # ── Telegram ──────────────────────────────────────────────────────────────
    TELEGRAM_API_ID: int = Field(default=0, description="Telegram API ID")
    TELEGRAM_API_HASH: str = Field(default="", description="Telegram API hash")
    TELEGRAM_PHONE: str = Field(default="", description="Telegram phone number")
    TELEGRAM_SESSION_NAME: str = Field(
        default="jarvis_session", description="Telegram session file name"
    )

    # ── ElevenLabs TTS ────────────────────────────────────────────────────────
    ELEVENLABS_API_KEY: str = Field(default="", description="ElevenLabs API key")
    ELEVENLABS_VOICE_ID: str = Field(
        default="21m00Tcm4TlvDq8ikWAM", description="ElevenLabs voice ID"
    )

    # ── Whisper STT ───────────────────────────────────────────────────────────
    WHISPER_MODEL: str = Field(
        default="base",
        description="Whisper model size: base | small | medium | large",
    )

    # ── Celery ────────────────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/1",
        description="Celery broker URL",
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/2",
        description="Celery result backend URL",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # ── Memory ────────────────────────────────────────────────────────────────
    MAX_MEMORY_ITEMS: int = Field(
        default=1000, description="Maximum number of memory items to retain"
    )
    MEMORY_DECAY_HOURS: int = Field(
        default=720, description="Hours before memory items decay (30 days default)"
    )

    # ── Reminders ─────────────────────────────────────────────────────────────
    REMINDER_CHECK_INTERVAL: int = Field(
        default=60, description="Interval in seconds to check for due reminders"
    )

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = Field(
        default=100, description="Maximum requests per window"
    )
    RATE_LIMIT_WINDOW_SECONDS: int = Field(
        default=60, description="Rate limit window size in seconds"
    )

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of: {allowed}")
        return v.lower()

    @field_validator("WHISPER_MODEL")
    @classmethod
    def validate_whisper_model(cls, v: str) -> str:
        allowed = {"base", "small", "medium", "large", "large-v2", "large-v3"}
        if v.lower() not in allowed:
            raise ValueError(f"WHISPER_MODEL must be one of: {allowed}")
        return v.lower()

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of: {allowed}")
        return v.upper()

    @field_validator("ALGORITHM")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        allowed = {"HS256", "HS384", "HS512", "RS256", "RS384", "RS512"}
        if v.upper() not in allowed:
            raise ValueError(f"ALGORITHM must be one of: {allowed}")
        return v.upper()

    # ── Derived properties ────────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    @property
    def is_debug(self) -> bool:
        return self.DEBUG or self.is_development


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Using lru_cache ensures the .env file is read only once per process
    lifetime, avoiding repeated disk I/O and keeping settings immutable.
    """
    return Settings()
