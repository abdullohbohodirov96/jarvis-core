"""
JARVIS configuration settings.

Loaded from environment variables / .env file via Pydantic v2 BaseSettings.
A cached singleton is available as module-level `settings`.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ App
    APP_NAME: str = "JARVIS"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"  # development | production
    LOG_LEVEL: str = "INFO"

    # ----------------------------------------------------------- API server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8765
    API_SECRET_KEY: str = "change-me-in-production"

    # ------------------------------------------------------------ Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./jarvis.db"

    # -------------------------------------------------------------- OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_MAX_TOKENS: int = 2048
    OPENAI_TEMPERATURE: float = 0.7

    # ------------------------------------------------------------ Telegram
    TELEGRAM_API_ID: int = 0
    TELEGRAM_API_HASH: str = ""
    TELEGRAM_PHONE: str = ""
    TELEGRAM_SESSION: str = "/tmp/jarvis_telegram"

    # --------------------------------------------------------------- Voice
    WHISPER_MODEL: str = "base"  # tiny | base | small | medium
    WAKE_WORD: str = "jarvis"
    VOICE_SAMPLE_RATE: int = 16000
    VOICE_SILENCE_THRESHOLD: float = 0.01
    VOICE_SILENCE_DURATION: float = 1.5  # seconds of silence to stop recording
    TTS_VOICE: str = "en-US-GuyNeural"
    VOICE_SPEED: str = "+0%"

    # ------------------------------------------------------------- Desktop
    DESKTOP_AUTOMATION_ENABLED: bool = True
    SCREENSHOT_DIR: str = "/tmp/jarvis_screenshots"

    # -------------------------------------------------------------- Memory
    MAX_CONVERSATION_HISTORY: int = 20
    TASK_DB_PATH: str = "./jarvis.db"

    # --------------------------------------------------- Computed helpers
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def database_url_async(self) -> str:
        """Return the DATABASE_URL, ensuring the aiosqlite driver is present."""
        url = self.DATABASE_URL
        if url.startswith("sqlite:///") and "aiosqlite" not in url:
            url = url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
        return url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()


# Module-level singleton — import this everywhere.
settings: Settings = get_settings()
