"""
ORM model registry.

Importing this package ensures every model class is registered on
``Base.metadata`` before Alembic or ``init_db()`` call ``create_all``.

Usage (in application code)::

    import backend.app.models  # noqa: F401 – registers all models

Or import individual models directly::

    from backend.app.models import User, Conversation, Message
"""

from backend.app.models.base import TimestampedBase
from backend.app.models.conversation import Conversation, ConversationSource
from backend.app.models.memory import Memory, MemoryType
from backend.app.models.message import Message, MessageRole, MessageSource
from backend.app.models.scheduled_message import ScheduledMessage, ScheduledMessageStatus
from backend.app.models.task import Task, TaskPriority, TaskSource, TaskStatus
from backend.app.models.telegram_chat import TelegramChat, TelegramChatType
from backend.app.models.user import User

__all__ = [
    # Base
    "TimestampedBase",
    # User
    "User",
    # Conversation + Message
    "Conversation",
    "ConversationSource",
    "Message",
    "MessageRole",
    "MessageSource",
    # Task
    "Task",
    "TaskStatus",
    "TaskPriority",
    "TaskSource",
    # Memory
    "Memory",
    "MemoryType",
    # Telegram
    "TelegramChat",
    "TelegramChatType",
    "ScheduledMessage",
    "ScheduledMessageStatus",
]
