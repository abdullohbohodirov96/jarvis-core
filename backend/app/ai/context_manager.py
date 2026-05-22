"""
Conversation context management for JARVIS.

ContextManager is responsible for:
- Building a complete message list for each OpenAI call (system prompt + history)
- Injecting relevant memories into the system prompt
- Managing the token budget (trimming old messages, inserting summaries)
- Optionally summarising long conversations to keep context fresh
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.client import OpenAIClient
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Tokens reserved for the model's completion (not burned on history)
_COMPLETION_RESERVE_TOKENS = 1024
# Minimum messages to keep even when trimming
_MIN_MESSAGES_TO_KEEP = 4
# Summarise conversation when it exceeds this many messages
_SUMMARIZE_THRESHOLD = 30


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------


class ContextManager:
    """
    Builds the full conversation context for each JARVIS inference call.

    Injects: system prompt, relevant memories, conversation history.
    Trims:   oldest messages when over the token budget.
    Summarises: long histories to keep the model's window fresh.
    """

    def __init__(
        self,
        user_id: int,
        db_session: AsyncSession,
        max_tokens: int = 8000,
    ) -> None:
        self._user_id = user_id
        self._db = db_session
        self._max_tokens = max_tokens
        self._client: OpenAIClient | None = None
        self._memory_manager: Any | None = None

    def set_client(self, client: OpenAIClient) -> None:
        self._client = client

    def set_memory_manager(self, memory_manager: Any) -> None:
        self._memory_manager = memory_manager

    # ------------------------------------------------------------------
    # build_context
    # ------------------------------------------------------------------

    async def build_context(
        self,
        conversation_id: int,
        new_message: str,
        system_prompt: str | None = None,
        user: Any | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the full messages list to send to OpenAI.

        Steps:
        1. Fetch conversation history from DB.
        2. Retrieve relevant memory context.
        3. Build (or use supplied) system prompt.
        4. Assemble messages in order: system → summary → history → new message.
        5. Trim to token budget if needed.

        Args:
            conversation_id: DB ID of the current conversation.
            new_message: The new user message being processed.
            system_prompt: Pre-built system prompt. If None, a basic one is used.
            user: User ORM object for personalised prompts.

        Returns:
            List of dicts in OpenAI chat format (role + content).
        """
        # 1. Fetch history
        history = await self.get_conversation_history(conversation_id, limit=50)

        # 2. Get memory context
        memory_context = ""
        if self._memory_manager is not None:
            try:
                memory_context = await self._memory_manager.get_relevant_context(
                    query=new_message,
                    max_tokens=600,
                )
            except Exception as exc:
                logger.warning("build_context: memory retrieval failed: %s", exc)

        # 3. Build system prompt
        if system_prompt is None:
            if user is not None:
                from app.ai.prompts.system_prompts import build_system_prompt

                current_time = datetime.now(timezone.utc).strftime(
                    "%A, %B %d %Y at %H:%M UTC"
                )
                system_prompt = build_system_prompt(
                    user=user,
                    memories=memory_context,
                    current_time=current_time,
                )
            else:
                from app.ai.prompts.system_prompts import JARVIS_SYSTEM_PROMPT

                system_prompt = JARVIS_SYSTEM_PROMPT.format(
                    user_name="User",
                    current_time=datetime.now(timezone.utc).strftime("%A, %B %d %Y at %H:%M UTC"),
                    user_context="",
                    memories=memory_context or "No memories stored yet.",
                )

        # 4. Assemble messages
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        # Check if there's a stored conversation summary to inject
        summary = await self._get_conversation_summary(conversation_id)
        if summary:
            messages = self._inject_summary(messages, summary)

        # Convert history (DB messages) to OpenAI format and append
        openai_history = self._messages_to_openai_format(history)
        messages.extend(openai_history)

        # Append the new user message
        messages.append({"role": "user", "content": new_message})

        # 5. Trim to token budget
        budget = self._max_tokens - _COMPLETION_RESERVE_TOKENS
        messages = self._trim_to_budget(messages, budget)

        logger.debug(
            "build_context: conversation=%d messages=%d estimated_tokens=%d",
            conversation_id,
            len(messages),
            self._estimate_tokens(messages),
        )
        return messages

    # ------------------------------------------------------------------
    # summarize_if_needed
    # ------------------------------------------------------------------

    async def summarize_if_needed(self, conversation_id: int) -> str | None:
        """
        Summarise the conversation if it exceeds the threshold.

        Stores the summary in the DB and marks older messages as summarised
        so they can be excluded from future context builds.

        Args:
            conversation_id: DB ID of the conversation.

        Returns:
            The summary string if one was generated, else None.
        """
        history = await self.get_conversation_history(
            conversation_id, limit=_SUMMARIZE_THRESHOLD + 10
        )
        if len(history) < _SUMMARIZE_THRESHOLD:
            return None

        # Messages to summarise: everything except the last 10
        to_summarise = history[:-10]
        if not to_summarise:
            return None

        if self._client is None:
            logger.warning("summarize_if_needed: no OpenAI client set")
            return None

        from app.ai.summarizer import ConversationSummarizer

        summarizer = ConversationSummarizer(self._client)
        openai_msgs = self._messages_to_openai_format(to_summarise)
        summary = await summarizer.summarize(openai_msgs, max_length=400)

        if not summary:
            return None

        # Persist summary
        try:
            from app.models.conversation import Conversation  # type: ignore[import]

            stmt = select(Conversation).where(Conversation.id == conversation_id)
            result = await self._db.execute(stmt)
            conv = result.scalar_one_or_none()
            if conv:
                conv.summary = summary
                conv.summary_up_to_message = len(to_summarise)
                conv.updated_at = datetime.now(timezone.utc)
                await self._db.flush()

            # Mark summarised messages
            for msg in to_summarise:
                msg_id = getattr(msg, "id", None)
                if msg_id:
                    setattr(msg, "is_summarised", True)
            await self._db.flush()

        except Exception as exc:
            logger.warning("summarize_if_needed: could not persist summary: %s", exc)

        logger.info(
            "summarize_if_needed: summarised %d messages for conversation=%d",
            len(to_summarise),
            conversation_id,
        )
        return summary

    # ------------------------------------------------------------------
    # get_conversation_history
    # ------------------------------------------------------------------

    async def get_conversation_history(
        self,
        conversation_id: int,
        limit: int = 50,
    ) -> list[Any]:
        """
        Fetch recent messages for a conversation from the DB.

        Args:
            conversation_id: The conversation's DB ID.
            limit: Maximum number of messages to retrieve.

        Returns:
            List of Message ORM objects ordered oldest-first.
        """
        try:
            from app.models.message import Message  # type: ignore[import]

            stmt = (
                select(Message)
                .where(
                    Message.conversation_id == conversation_id,
                    Message.is_summarised.is_(False),
                )
                .order_by(Message.created_at.asc())
                .limit(limit)
            )
            result = await self._db.execute(stmt)
            messages = list(result.scalars().all())
            return messages
        except Exception as exc:
            logger.warning("get_conversation_history: DB error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _messages_to_openai_format(
        self, messages: list[Any]
    ) -> list[dict[str, Any]]:
        """
        Convert DB Message ORM objects to OpenAI chat message dicts.

        Handles role mapping: "user" | "assistant" | "system" | "tool".
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "") or ""

            # Normalise roles to OpenAI-accepted values
            if role not in ("user", "assistant", "system", "tool"):
                role = "user"

            entry: dict[str, Any] = {"role": role, "content": content}

            # Preserve tool_call_id for tool messages
            tool_call_id = getattr(msg, "tool_call_id", None)
            if role == "tool" and tool_call_id:
                entry["tool_call_id"] = tool_call_id
                name = getattr(msg, "tool_name", None)
                if name:
                    entry["name"] = name

            # Preserve tool_calls on assistant messages
            tool_calls = getattr(msg, "tool_calls", None)
            if role == "assistant" and tool_calls:
                entry["tool_calls"] = tool_calls

            result.append(entry)

        return result

    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """
        Estimate total token count for a messages list.

        Uses the OpenAI client's tiktoken-based counter when available,
        otherwise falls back to a char/4 heuristic.
        """
        if self._client is not None:
            return self._client.count_tokens(messages)

        # Fallback: ~4 chars per token, ~3 tokens per message overhead
        total = 0
        for msg in messages:
            total += 3
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total += len(part.get("text", "")) // 4
        return total + 3

    def _trim_to_budget(
        self,
        messages: list[dict[str, Any]],
        budget: int,
    ) -> list[dict[str, Any]]:
        """
        Trim the messages list to fit within the token budget.

        Always preserves:
        - The system message (index 0)
        - The last ``_MIN_MESSAGES_TO_KEEP`` messages
        - The final user message

        Removes messages from the middle (oldest non-system messages first).

        Args:
            messages: Full messages list to trim.
            budget: Token limit to stay under.

        Returns:
            Trimmed messages list.
        """
        if self._estimate_tokens(messages) <= budget:
            return messages

        if not messages:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if not non_system:
            return messages

        # Split: tail we must keep, middle we can trim from
        keep_tail_count = _MIN_MESSAGES_TO_KEEP
        tail = non_system[-keep_tail_count:]
        middle = non_system[:-keep_tail_count]

        # Remove from the front of middle until we're under budget
        while middle and self._estimate_tokens(system_msgs + middle + tail) > budget:
            middle.pop(0)

        trimmed = system_msgs + middle + tail
        if len(trimmed) < len(messages):
            logger.debug(
                "_trim_to_budget: trimmed %d messages (budget=%d tokens)",
                len(messages) - len(trimmed),
                budget,
            )
        return trimmed

    def _inject_summary(
        self,
        messages: list[dict[str, Any]],
        summary: str,
    ) -> list[dict[str, Any]]:
        """
        Inject a conversation summary as a system message directly after the
        main system prompt.

        Args:
            messages: Current messages list (must start with a system message).
            summary: Summary text to inject.

        Returns:
            Updated messages list with summary injected.
        """
        summary_msg: dict[str, Any] = {
            "role": "system",
            "content": (
                f"[Conversation Summary — earlier exchanges]\n{summary}\n"
                "[End of summary — the conversation continues below]"
            ),
        }
        # Insert after the first system message
        if messages and messages[0].get("role") == "system":
            return [messages[0], summary_msg] + messages[1:]
        return [summary_msg] + messages

    async def _get_conversation_summary(
        self, conversation_id: int
    ) -> str | None:
        """Retrieve a stored summary for this conversation from the DB."""
        try:
            from app.models.conversation import Conversation  # type: ignore[import]

            stmt = select(Conversation).where(Conversation.id == conversation_id)
            result = await self._db.execute(stmt)
            conv = result.scalar_one_or_none()
            if conv:
                return getattr(conv, "summary", None)
        except Exception:
            pass
        return None
