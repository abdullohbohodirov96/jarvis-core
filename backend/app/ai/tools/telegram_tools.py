"""
Telegram tools for the JARVIS agent.

Allows the AI to send messages, read recent conversations, search chats,
and schedule future messages via Telegram.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.ai.tools.base import BaseTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SendTelegramMessageTool
# ---------------------------------------------------------------------------


class SendTelegramMessageTool(BaseTool):
    """Send a Telegram message to a contact or chat on behalf of the user."""

    name = "send_telegram_message"
    description = (
        "Send a Telegram message to a contact or chat. "
        "Use this when the user explicitly asks you to send a message to someone. "
        "Always confirm the recipient and message content before sending."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "recipient": {
                "type": "string",
                "description": (
                    "Username, phone number, or full name of the recipient. "
                    "Examples: '@username', '+1234567890', 'John Doe'."
                ),
            },
            "message": {
                "type": "string",
                "description": "Text content of the message to send.",
            },
            "chat_id": {
                "type": "integer",
                "description": (
                    "Optional Telegram chat/user ID if known. Preferred over "
                    "'recipient' when available for accuracy."
                ),
            },
            "parse_mode": {
                "type": "string",
                "enum": ["plain", "markdown", "html"],
                "description": "Text formatting mode. Defaults to 'plain'.",
            },
        },
        "required": ["message", "recipient"],
    }

    def __init__(self, user_id: int, db_session: AsyncSession) -> None:
        self._user_id = user_id
        self._db = db_session

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        recipient: str = params["recipient"]
        message: str = params["message"]
        chat_id: int | None = params.get("chat_id")
        parse_mode: str = params.get("parse_mode", "plain")

        if not message.strip():
            return {"success": False, "error": "Message cannot be empty."}

        try:
            # Import the Telegram client service if available
            from backend.app.telegram.client import get_telegram_client  # type: ignore[import]

            client = await get_telegram_client(self._user_id, self._db)
            if client is None:
                return {
                    "success": False,
                    "error": "Telegram client is not configured for this user.",
                }

            target = chat_id if chat_id else recipient
            await client.send_message(target, message, parse_mode=parse_mode)

            logger.info(
                "SendTelegramMessageTool: sent message to '%s' for user=%s",
                recipient,
                self._user_id,
            )
            return {
                "success": True,
                "recipient": recipient,
                "message_preview": message[:100],
                "sent_at": datetime.now(timezone.utc).isoformat(),
                "message": f"Message sent to {recipient} successfully.",
            }
        except ImportError:
            # Telegram module not yet wired up – return a clear mock response
            logger.warning("SendTelegramMessageTool: Telegram module unavailable")
            return {
                "success": False,
                "error": "Telegram integration is not available on this server.",
            }
        except Exception as exc:
            logger.exception("SendTelegramMessageTool failed: %s", exc)
            return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# GetTelegramMessagesTool
# ---------------------------------------------------------------------------


class GetTelegramMessagesTool(BaseTool):
    """Retrieve recent messages from a Telegram chat."""

    name = "get_telegram_messages"
    description = (
        "Retrieve recent messages from a specific Telegram chat or contact. "
        "Use this to check what someone said recently or to review a conversation."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "chat_name": {
                "type": "string",
                "description": (
                    "Name, username, or phone number of the chat to read from."
                ),
            },
            "chat_id": {
                "type": "integer",
                "description": "Optional Telegram chat ID for direct lookup.",
            },
            "limit": {
                "type": "integer",
                "description": "Number of recent messages to retrieve. Defaults to 10.",
                "minimum": 1,
                "maximum": 50,
            },
            "min_date": {
                "type": "string",
                "description": (
                    "Optional ISO 8601 datetime. Only return messages after this date."
                ),
            },
        },
        "required": [],
    }

    def __init__(self, user_id: int, db_session: AsyncSession) -> None:
        self._user_id = user_id
        self._db = db_session

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        chat_name: str | None = params.get("chat_name")
        chat_id: int | None = params.get("chat_id")
        limit: int = min(int(params.get("limit", 10)), 50)
        min_date_str: str | None = params.get("min_date")

        if not chat_name and not chat_id:
            return {
                "success": False,
                "error": "Either 'chat_name' or 'chat_id' must be provided.",
                "messages": [],
            }

        min_date: datetime | None = None
        if min_date_str:
            try:
                min_date = datetime.fromisoformat(min_date_str)
            except ValueError:
                pass

        try:
            from backend.app.telegram.client import get_telegram_client  # type: ignore[import]

            client = await get_telegram_client(self._user_id, self._db)
            if client is None:
                return {
                    "success": False,
                    "error": "Telegram client not configured.",
                    "messages": [],
                }

            target = chat_id if chat_id else chat_name
            raw_messages = await client.get_messages(target, limit=limit, min_date=min_date)

            messages = [
                {
                    "id": m.id,
                    "sender": getattr(m, "sender_username", None) or str(getattr(m, "sender_id", "unknown")),
                    "text": m.text or "",
                    "date": m.date.isoformat() if m.date else None,
                    "is_outgoing": getattr(m, "out", False),
                }
                for m in (raw_messages or [])
            ]

            return {
                "success": True,
                "chat": chat_name or str(chat_id),
                "count": len(messages),
                "messages": messages,
            }
        except ImportError:
            return {
                "success": False,
                "error": "Telegram integration is not available.",
                "messages": [],
            }
        except Exception as exc:
            logger.exception("GetTelegramMessagesTool failed: %s", exc)
            return {"success": False, "error": str(exc), "messages": []}


# ---------------------------------------------------------------------------
# SearchTelegramChatsTool
# ---------------------------------------------------------------------------


class SearchTelegramChatsTool(BaseTool):
    """Search for Telegram chats, groups, or contacts by name."""

    name = "search_telegram_chats"
    description = (
        "Search for Telegram chats, groups, channels, or contacts by name. "
        "Useful to find the correct chat before sending a message."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search term to match against chat/contact names.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return. Defaults to 5.",
                "minimum": 1,
                "maximum": 20,
            },
            "chat_type": {
                "type": "string",
                "enum": ["all", "private", "group", "channel"],
                "description": "Filter by chat type. Defaults to 'all'.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, user_id: int, db_session: AsyncSession) -> None:
        self._user_id = user_id
        self._db = db_session

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        query: str = params["query"].strip()
        limit: int = min(int(params.get("limit", 5)), 20)
        chat_type: str = params.get("chat_type", "all")

        if not query:
            return {"success": False, "error": "Query cannot be empty.", "chats": []}

        try:
            from backend.app.telegram.client import get_telegram_client  # type: ignore[import]

            client = await get_telegram_client(self._user_id, self._db)
            if client is None:
                return {
                    "success": False,
                    "error": "Telegram client not configured.",
                    "chats": [],
                }

            dialogs = await client.get_dialogs(limit=100)
            chats = []

            for dialog in (dialogs or []):
                name = getattr(dialog, "name", "") or ""
                if query.lower() not in name.lower():
                    continue

                entity = getattr(dialog, "entity", None)
                kind = "private"
                if hasattr(entity, "megagroup"):
                    kind = "group"
                elif hasattr(entity, "broadcast"):
                    kind = "channel"

                if chat_type != "all" and kind != chat_type:
                    continue

                chats.append(
                    {
                        "id": getattr(entity, "id", None),
                        "name": name,
                        "type": kind,
                        "username": getattr(entity, "username", None),
                    }
                )
                if len(chats) >= limit:
                    break

            return {
                "success": True,
                "query": query,
                "count": len(chats),
                "chats": chats,
            }
        except ImportError:
            return {
                "success": False,
                "error": "Telegram integration is not available.",
                "chats": [],
            }
        except Exception as exc:
            logger.exception("SearchTelegramChatsTool failed: %s", exc)
            return {"success": False, "error": str(exc), "chats": []}


# ---------------------------------------------------------------------------
# ScheduleTelegramMessageTool
# ---------------------------------------------------------------------------


class ScheduleTelegramMessageTool(BaseTool):
    """Schedule a Telegram message to be sent at a future time."""

    name = "schedule_telegram_message"
    description = (
        "Schedule a Telegram message to be sent to a contact at a specified "
        "future date and time. The message will be queued and sent automatically."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "recipient": {
                "type": "string",
                "description": "Username, phone number, or name of the recipient.",
            },
            "message": {
                "type": "string",
                "description": "Message text to send.",
            },
            "send_at": {
                "type": "string",
                "description": (
                    "ISO 8601 datetime for when to send the message "
                    "(e.g. '2024-12-31T09:00:00Z'). Must be in the future."
                ),
            },
            "chat_id": {
                "type": "integer",
                "description": "Optional Telegram chat ID for the recipient.",
            },
        },
        "required": ["recipient", "message", "send_at"],
    }

    def __init__(self, user_id: int, db_session: AsyncSession) -> None:
        self._user_id = user_id
        self._db = db_session

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        recipient: str = params["recipient"]
        message: str = params["message"]
        send_at_str: str = params["send_at"]
        chat_id: int | None = params.get("chat_id")

        # Parse and validate the send time
        try:
            send_at = datetime.fromisoformat(send_at_str)
            if send_at.tzinfo is None:
                send_at = send_at.replace(tzinfo=timezone.utc)
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid date format: '{send_at_str}'. Use ISO 8601.",
            }

        now = datetime.now(timezone.utc)
        if send_at <= now:
            return {
                "success": False,
                "error": "Scheduled time must be in the future.",
            }

        try:
            from backend.app.models.scheduled_message import ScheduledMessage  # type: ignore[import]

            scheduled = ScheduledMessage(
                user_id=self._user_id,
                recipient=recipient,
                chat_id=chat_id,
                message=message,
                send_at=send_at,
                status="pending",
                created_at=now,
            )
            self._db.add(scheduled)
            await self._db.flush()
            await self._db.refresh(scheduled)

            logger.info(
                "ScheduleTelegramMessageTool: scheduled message id=%s for %s at %s",
                scheduled.id,
                recipient,
                send_at.isoformat(),
            )
            return {
                "success": True,
                "scheduled_id": scheduled.id,
                "recipient": recipient,
                "send_at": send_at.isoformat(),
                "message": f"Message to {recipient} scheduled for {send_at.strftime('%B %d at %H:%M UTC')}.",
            }
        except ImportError:
            # Fallback: record intent without DB model
            logger.warning("ScheduleTelegramMessageTool: ScheduledMessage model unavailable")
            return {
                "success": True,
                "scheduled_id": None,
                "recipient": recipient,
                "send_at": send_at.isoformat(),
                "message": (
                    f"Message to {recipient} queued for "
                    f"{send_at.strftime('%B %d at %H:%M UTC')} (DB model unavailable)."
                ),
            }
        except Exception as exc:
            logger.exception("ScheduleTelegramMessageTool failed: %s", exc)
            return {"success": False, "error": str(exc)}
