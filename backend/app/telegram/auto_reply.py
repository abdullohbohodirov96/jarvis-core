"""
AI-powered auto-reply system for Telegram.

TelegramAutoReply decides whether an incoming message warrants an automatic
AI-generated response, produces a contextual reply in the user's voice, and
sends it — with per-chat rate limiting to prevent spam.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.app.ai.client import OpenAIClient
    from backend.app.telegram.client import TelegramClientManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter defaults
# ---------------------------------------------------------------------------

_DEFAULT_MAX_REPLIES_PER_HOUR: int = 10
_DEFAULT_WINDOW_SECONDS: int = 3600  # 1 hour

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SHOULD_REPLY_SYSTEM = """You decide if a Telegram message warrants an auto-reply from an AI assistant.

Rules:
- Reply "yes" only if the message contains a direct question, a specific request, or requires an acknowledgement.
- Reply "no" for casual chat, announcements, media, stickers, or if the message was already answered.
- Reply "no" for outgoing messages (the user's own messages).

Output ONLY "yes" or "no"."""

_GENERATE_REPLY_SYSTEM = """You are replying to Telegram messages on behalf of the user.

User persona / style:
{persona}

User context:
{user_context}

Guidelines:
- Match the user's tone and language style.
- Be concise — Telegram messages should be short (1–4 sentences unless detail is required).
- Be natural and conversational, not robotic.
- Do not start with "I" or "Sure".
- Never reveal you are an AI unless explicitly asked.
- Only answer what was asked; do not pad with pleasantries.
"""

_HANDLE_QUESTION_SYSTEM = """Answer the following question concisely and accurately.
Keep the answer under 3 sentences.  Use plain text — no markdown.
If you don't know, say so briefly."""

_HANDLE_TASK_REQUEST_SYSTEM = """The user has received a task request in a Telegram message.
Draft a polite acknowledgement confirming you've noted the task.
Keep it to 1–2 sentences.  Be professional but warm."""

# ---------------------------------------------------------------------------
# Per-chat rate limiter
# ---------------------------------------------------------------------------


class _ChatRateLimiter:
    """Sliding-window rate limiter keyed by chat_id."""

    def __init__(
        self,
        max_replies: int = _DEFAULT_MAX_REPLIES_PER_HOUR,
        window_seconds: int = _DEFAULT_WINDOW_SECONDS,
    ) -> None:
        self._max = max_replies
        self._window = window_seconds
        # chat_id -> list of send timestamps
        self._history: dict[int | str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(self, chat_id: int | str) -> bool:
        """Return True if a reply to chat_id is within the rate limit."""
        now = time.monotonic()
        async with self._lock:
            timestamps = self._history[chat_id]
            # Purge old entries outside the sliding window
            cutoff = now - self._window
            self._history[chat_id] = [t for t in timestamps if t >= cutoff]
            if len(self._history[chat_id]) < self._max:
                self._history[chat_id].append(now)
                return True
            return False

    async def reset(self, chat_id: int | str) -> None:
        """Clear history for a specific chat (e.g. after user disables auto-reply)."""
        async with self._lock:
            self._history.pop(chat_id, None)


# ---------------------------------------------------------------------------
# TelegramAutoReply
# ---------------------------------------------------------------------------


class TelegramAutoReply:
    """
    Evaluates incoming Telegram messages and optionally sends an AI-generated
    reply on the user's behalf.

    Args:
        client:        Connected TelegramClientManager.
        openai_client: OpenAIClient singleton.
        db_session:    Async SQLAlchemy session (for storing reply history).
        max_replies_per_hour: Rate limit per chat (default 10).
    """

    def __init__(
        self,
        client: TelegramClientManager,
        openai_client: OpenAIClient,
        db_session: AsyncSession,
        max_replies_per_hour: int = _DEFAULT_MAX_REPLIES_PER_HOUR,
    ) -> None:
        self._client = client
        self._ai = openai_client
        self._db = db_session
        self._rate_limiter = _ChatRateLimiter(max_replies=max_replies_per_hour)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def should_reply(
        self,
        message: dict[str, Any],
        chat_config: dict[str, Any],
    ) -> bool:
        """
        Decide whether an incoming message warrants an auto-reply.

        Decision factors (in order):
        1. ``chat_config["auto_reply_enabled"]`` must be True.
        2. Message must be inbound (not the user's own outgoing message).
        3. Message must have non-empty text.
        4. Rate limit: not exceeded for this chat.
        5. AI judgement: model says "yes".

        Args:
            message:     Serialised message dict (from message_handler).
            chat_config: Per-chat settings dict; expected keys:
                         ``auto_reply_enabled`` (bool),
                         ``auto_reply_keywords`` (list[str], optional).

        Returns:
            True if an auto-reply should be sent.
        """
        if not chat_config.get("auto_reply_enabled", False):
            return False

        if message.get("is_out", False):
            return False

        text: str = message.get("text", "").strip()
        if not text:
            return False

        chat_id = message.get("chat_id")
        if not await self._rate_limiter.is_allowed(chat_id):
            logger.debug("should_reply: rate limit reached for chat=%s", chat_id)
            return False

        # Keyword shortcut: if any configured keywords are present, always reply
        keywords: list[str] = chat_config.get("auto_reply_keywords", [])
        if keywords:
            text_lower = text.lower()
            if any(kw.lower() in text_lower for kw in keywords):
                return True

        # AI judgement for everything else
        try:
            response = await self._ai.chat_completion(
                messages=[
                    {"role": "system", "content": _SHOULD_REPLY_SYSTEM},
                    {"role": "user", "content": text[:500]},
                ],
                temperature=0.0,
                max_tokens=5,
            )
            verdict = (response or "").strip().lower()
            logger.debug(
                "should_reply: chat=%s verdict=%s text_snippet=%r",
                chat_id, verdict, text[:60],
            )
            return verdict == "yes"
        except Exception as exc:  # noqa: BLE001
            logger.warning("should_reply AI check failed: %s", exc)
            # Default to not replying on AI failure to avoid spam
            return False

    async def generate_reply(
        self,
        message: dict[str, Any],
        chat_history: list[dict[str, Any]],
        user_context: str = "",
    ) -> str:
        """
        Generate a contextual AI reply that matches the user's voice and style.

        Args:
            message:      The incoming message to reply to.
            chat_history: Recent conversation history (oldest first).
            user_context: Optional user profile context (name, preferences, facts).

        Returns:
            Generated reply text, or empty string on failure.
        """
        persona = self._get_reply_persona()
        system = _GENERATE_REPLY_SYSTEM.format(
            persona=persona,
            user_context=user_context or "No additional context available.",
        )

        # Build conversation turns from history
        history_messages: list[dict[str, str]] = []
        for hist_msg in chat_history[-10:]:  # last 10 messages for context
            role = "assistant" if hist_msg.get("is_out") else "user"
            text = hist_msg.get("text", "").strip()
            if text:
                history_messages.append({"role": role, "content": text})

        # Append the current incoming message
        history_messages.append({"role": "user", "content": message.get("text", "")})

        try:
            reply = await self._ai.chat_completion(
                messages=[{"role": "system", "content": system}] + history_messages,
                temperature=0.75,
                max_tokens=300,
            )
            result: str = (reply or "").strip()
            logger.debug(
                "generate_reply: chat=%s reply_len=%d", message.get("chat_id"), len(result)
            )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("generate_reply failed: %s", exc)
            return ""

    async def send_reply(
        self,
        chat_id: int | str,
        reply_text: str,
        reply_to_id: int | None = None,
    ) -> bool:
        """
        Send a generated reply to a Telegram chat.

        Args:
            chat_id:      Target chat ID or username.
            reply_text:   The text to send.
            reply_to_id:  Optional message ID to quote/reply to.

        Returns:
            True if sent successfully.
        """
        if not reply_text.strip():
            logger.warning("send_reply called with empty text for chat=%s", chat_id)
            return False

        try:
            success = await self._client.send_message(
                chat_id_or_username=chat_id,
                text=reply_text,
                reply_to=reply_to_id,
            )
            if success:
                logger.info(
                    "send_reply: sent to chat=%s reply_to=%s len=%d",
                    chat_id, reply_to_id, len(reply_text),
                )
            return success
        except Exception as exc:  # noqa: BLE001
            logger.error("send_reply failed for chat=%s: %s", chat_id, exc)
            return False

    async def handle_question(self, message: dict[str, Any]) -> str | None:
        """
        Directly answer a question detected in an incoming message.

        Intended for messages classified as ``intent == "question"`` by
        TelegramTaskDetector.

        Args:
            message: Serialised message dict.

        Returns:
            Answer string, or None if unanswerable / unsure.
        """
        text = message.get("text", "").strip()
        if not text:
            return None

        try:
            response = await self._ai.chat_completion(
                messages=[
                    {"role": "system", "content": _HANDLE_QUESTION_SYSTEM},
                    {"role": "user", "content": text[:800]},
                ],
                temperature=0.4,
                max_tokens=200,
            )
            answer = (response or "").strip()
            logger.debug("handle_question: chat=%s answer_len=%d", message.get("chat_id"), len(answer))
            return answer or None
        except Exception as exc:  # noqa: BLE001
            logger.warning("handle_question AI call failed: %s", exc)
            return None

    async def handle_task_request(self, message: dict[str, Any]) -> str | None:
        """
        Generate a polite acknowledgement for a task request received in a message.

        Args:
            message: Serialised message dict.

        Returns:
            Acknowledgement text, or None on failure.
        """
        text = message.get("text", "").strip()
        if not text:
            return None

        try:
            response = await self._ai.chat_completion(
                messages=[
                    {"role": "system", "content": _HANDLE_TASK_REQUEST_SYSTEM},
                    {"role": "user", "content": text[:600]},
                ],
                temperature=0.6,
                max_tokens=100,
            )
            ack = (response or "").strip()
            logger.debug(
                "handle_task_request: chat=%s ack_len=%d", message.get("chat_id"), len(ack)
            )
            return ack or None
        except Exception as exc:  # noqa: BLE001
            logger.warning("handle_task_request AI call failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_reply_persona(self) -> str:
        """
        Return a description of the user's communication style / persona to
        include in the reply system prompt.

        This can be extended to pull dynamic persona data from MemoryService.
        """
        return (
            "Professional yet approachable. Writes in clear, direct sentences. "
            "Uses minimal punctuation. Favours short replies. Occasionally uses "
            "casual contractions (I'll, we'll, it's). Never uses emoji unless "
            "the other person used them first. Does not use exclamation marks."
        )
