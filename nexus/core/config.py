from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """
    Central configuration loaded from environment variables or a .env file.
    All fields map 1-to-1 with .env.example entries.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    PROJECT_NAME: str = "NEXUS-Core"
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str

    # ── OpenAI ────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str

    # ── Telegram ──────────────────────────────────────────────────────────────
    TELEGRAM_API_ID: int
    TELEGRAM_API_HASH: str
    TELEGRAM_BOT_TOKEN: str
    TARGET_CHAT_ID: int = 0

    # ── SaaS NEXUS Additions ──────────────────────────────────────────────────
    ENCRYPTION_KEY: str = "transient-fallback-key-replace-in-production-000="
    MINI_APP_URL: str = ""
    OWNER_ID: int = 0
    ALLOWED_USER_IDS: str = ""

    @property
    def allowed_ids(self) -> set[int]:
        ids = {self.OWNER_ID} if self.OWNER_ID != 0 else set()
        for x in self.ALLOWED_USER_IDS.split(","):
            x = x.strip()
            if x.isdigit():
                ids.add(int(x))
        return ids

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() in {"development", "dev"}

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()

settings: Settings = get_settings()
