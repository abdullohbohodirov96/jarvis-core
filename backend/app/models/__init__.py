"""
ORM model registry.

Importing this package ensures every model class is registered on
``Base.metadata`` before Alembic or ``init_db()`` call ``create_all``.

Usage (in application code)::

    import app.models  # noqa: F401 – registers all models

Or import individual models directly::

    from app.models import User, Conversation, Message
"""

from app.models.base import TimestampedBase
from app.models.conversation import Conversation, ConversationSource
from app.models.memory import Memory, MemoryType
from app.models.message import Message, MessageRole, MessageSource
from app.models.scheduled_message import ScheduledMessage, ScheduledMessageStatus
from app.models.task import Task, TaskPriority, TaskSource, TaskStatus
from app.models.telegram_chat import TelegramChat, TelegramChatType
from app.models.user import User

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
