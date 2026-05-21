"""
Application-wide constants for JARVIS.

All magic numbers, string literals, and enum values live here.
Import from this module instead of hard-coding values elsewhere so
that a single change propagates to all callers.
"""

from __future__ import annotations

from enum import Enum

# =============================================================================
# API
# =============================================================================

API_V1_PREFIX: str = "/api/v1"

#: Maximum page size for paginated list endpoints.
MAX_PAGE_SIZE: int = 100

#: Default page size when the caller does not specify one.
DEFAULT_PAGE_SIZE: int = 20

#: Hard limit on the length of a single user / assistant message (characters).
MAX_MESSAGE_LENGTH: int = 10_000

#: Maximum number of audio bytes accepted in a single voice transcription upload.
#: 50 MB — enough for ~50 minutes of compressed audio.
MAX_AUDIO_UPLOAD_BYTES: int = 50 * 1024 * 1024


# =============================================================================
# AI / LLM
# =============================================================================

#: Default OpenAI completion model (overridden by OPENAI_MODEL env var).
DEFAULT_AI_MODEL: str = "gpt-4o"

#: Model used for lightweight / cheap classification tasks.
FAST_AI_MODEL: str = "gpt-4o-mini"

#: Maximum number of tokens allowed in the full context window (prompt + reply).
#: Keeps under gpt-4o's 128k limit while leaving room for the system prompt.
MAX_CONTEXT_TOKENS: int = 8_000

#: Embedding dimensionality produced by text-embedding-3-small.
EMBEDDING_DIMENSION: int = 1536

#: Number of recent conversation turns to include in the AI context.
MAX_CONVERSATION_HISTORY_TURNS: int = 20

#: Minimum cosine similarity for a memory to be considered relevant.
MEMORY_SIMILARITY_THRESHOLD: float = 0.75


# =============================================================================
# Audio / Voice
# =============================================================================

#: Audio formats accepted for transcription uploads.
SUPPORTED_AUDIO_FORMATS: list[str] = [
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/ogg",
    "audio/webm",
    "audio/flac",
    "video/webm",   # Some browsers send WebM with audio track
]

#: File extensions mapped from MIME types (for validation).
SUPPORTED_AUDIO_EXTENSIONS: list[str] = [
    ".wav", ".mp3", ".mp4", ".ogg", ".webm", ".flac", ".m4a",
]

#: Words or phrases that trigger the voice assistant.
VOICE_WAKE_WORDS: list[str] = [
    "jarvis",
    "hey jarvis",
    "ok jarvis",
    "yo jarvis",
]

#: Audio sample rate used throughout the voice pipeline (Hz).
VOICE_SAMPLE_RATE: int = 16_000

#: RMS silence threshold for voice-activity detection (0.0 – 1.0).
VOICE_SILENCE_THRESHOLD: float = 0.02

#: Seconds of silence before the utterance is considered complete.
VOICE_SILENCE_DURATION: float = 1.5


# =============================================================================
# Task management
# =============================================================================

class TaskStatus(str, Enum):
    """Lifecycle states of a JARVIS task item."""
    PENDING    = "pending"     # Created but not yet started
    IN_PROGRESS = "in_progress" # Actively being worked on
    COMPLETED  = "completed"   # Done and verified
    CANCELLED  = "cancelled"   # Abandoned by the user
    OVERDUE    = "overdue"     # Past due date and still not done

class TaskPriority(str, Enum):
    """User-assigned priority level of a task."""
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    URGENT = "urgent"

#: Default task priority when the AI cannot determine one from context.
DEFAULT_TASK_PRIORITY: TaskPriority = TaskPriority.MEDIUM

#: Keyword fragments that signal high priority in free-text input.
HIGH_PRIORITY_KEYWORDS: list[str] = [
    "urgent", "asap", "immediately", "critical", "emergency",
    "deadline today", "right now", "must do",
]


# =============================================================================
# Memory system
# =============================================================================

class MemoryType(str, Enum):
    """Categories of stored memory items."""
    FACT          = "fact"          # A general fact about the user or the world
    PREFERENCE    = "preference"    # User likes, dislikes, or habitual choices
    TASK          = "task"          # A remembered task or to-do
    CONVERSATION  = "conversation"  # Summary of a past conversation
    REMINDER      = "reminder"      # A time-bound reminder
    EMOTION       = "emotion"       # Noted emotional state or reaction
    CONTACT       = "contact"       # Information about a person the user knows
    CUSTOM        = "custom"        # User-defined category

#: Memory items with importance below this threshold are not persisted.
MEMORY_IMPORTANCE_THRESHOLD: float = 0.3

#: Maximum number of memory items returned in a relevance search.
MEMORY_SEARCH_TOP_K: int = 10


# =============================================================================
# Conversation / messaging
# =============================================================================

class MessageRole(str, Enum):
    """Roles of participants in a conversation turn."""
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"
    TOOL      = "tool"       # Function / tool call result

class ConversationStatus(str, Enum):
    """Lifecycle state of a conversation."""
    ACTIVE   = "active"
    ARCHIVED = "archived"
    DELETED  = "deleted"

#: System prompt prefix injected at the start of every AI context.
SYSTEM_PROMPT_PREFIX: str = (
    "You are JARVIS, an intelligent personal AI assistant. "
    "You are helpful, concise, and proactive. "
    "You have access to the user's tasks, memories, and Telegram messages. "
    "Always respond in the same language as the user's last message."
)


# =============================================================================
# Celery task queues
# =============================================================================

class CeleryQueue(str, Enum):
    """Named Celery task queues."""
    DEFAULT        = "default"
    AI_TASKS       = "ai_tasks"
    VOICE_TASKS    = "voice_tasks"
    TELEGRAM_TASKS = "telegram_tasks"
    REMINDERS      = "reminders"

#: All queue names (used when starting the worker with -Q flag).
ALL_CELERY_QUEUES: list[str] = [q.value for q in CeleryQueue]


# =============================================================================
# Telegram
# =============================================================================

#: Maximum characters in a single Telegram message (API hard limit is 4096).
TELEGRAM_MAX_MESSAGE_LENGTH: int = 4_096

#: Telegram message parse modes.
class TelegramParseMode(str, Enum):
    MARKDOWN = "Markdown"
    MARKDOWN_V2 = "MarkdownV2"
    HTML = "HTML"


# =============================================================================
# Cache TTLs (seconds)
# =============================================================================

CACHE_TTL_USER_PROFILE: int     = 300    #  5 minutes
CACHE_TTL_CONVERSATION: int     = 600    # 10 minutes
CACHE_TTL_AI_RESPONSE: int      = 3_600  #  1 hour
CACHE_TTL_TASK_LIST: int        = 120    #  2 minutes
CACHE_TTL_MEMORY_SEARCH: int    = 300    #  5 minutes
CACHE_TTL_DAILY_SUMMARY: int    = 86_400 # 24 hours


# =============================================================================
# Rate limiting
# =============================================================================

RATE_LIMIT_VOICE_UPLOAD_PER_MINUTE: int = 10
RATE_LIMIT_AI_CHAT_PER_MINUTE: int      = 60
RATE_LIMIT_AUTH_PER_MINUTE: int         = 5


# =============================================================================
# HTTP headers
# =============================================================================

HEADER_REQUEST_ID: str  = "X-Request-ID"
HEADER_PROCESS_TIME: str = "X-Process-Time"
HEADER_RATE_LIMIT: str   = "X-RateLimit-Limit"
HEADER_RATE_REMAINING: str = "X-RateLimit-Remaining"
