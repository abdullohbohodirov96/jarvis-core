"""
JARVIS main agent — orchestrates the complete AI reasoning loop.

Architecture overview:
- ConversationMemory: sliding-window in-memory store for the current session.
- JarvisAgent: ties together AIClient, CommandParser, ConversationMemory and
  the optional external memory store.  Does NOT execute actions directly;
  instead, it publishes an event on the event_bus so the desktop layer can
  react without tight coupling.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ai.client import AIClient, get_ai_client
from ai.command_parser import CommandParser, ParsedResponse
from ai.prompts import build_system_prompt, format_conversation_history
from config.settings import settings
from core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------


class ConversationMemory:
    """
    Sliding-window in-memory conversation history.

    Keeps the last ``max_turns`` *pairs* of user+assistant messages.
    One "turn" equals one user message plus one assistant reply.
    """

    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns: int = max_turns
        self.messages: list[dict[str, Any]] = []

    def add(self, role: str, content: str) -> None:
        """
        Append a single message.

        Args:
            role:    "user" | "assistant" | "system"
            content: Message text.
        """
        self.messages.append({"role": role, "content": content})
        self._enforce_window()

    def get_history(self) -> list[dict[str, Any]]:
        """
        Return a copy of the current sliding-window message list.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts.
        """
        return list(self.messages)

    def clear(self) -> None:
        """Discard all stored messages."""
        self.messages.clear()

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise to a JSON-safe dict for persistence.

        Returns:
            Dict with ``max_turns`` and ``messages`` keys.
        """
        return {
            "max_turns": self.max_turns,
            "messages": list(self.messages),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationMemory":
        """
        Reconstruct a ConversationMemory from a previously serialised dict.

        Args:
            data: Dict as returned by ``to_dict()``.

        Returns:
            Populated ``ConversationMemory`` instance.
        """
        obj = cls(max_turns=data.get("max_turns", 20))
        for msg in data.get("messages", []):
            if isinstance(msg, dict):
                obj.messages.append(msg)
        obj._enforce_window()
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enforce_window(self) -> None:
        """Trim the list to the last max_turns * 2 messages."""
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]


# ---------------------------------------------------------------------------
# AgentResponse dataclass
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    """Complete response from a single JarvisAgent.process() call."""

    speech: str
    """What JARVIS says out loud."""

    action_type: str | None
    """Action type to execute, or None."""

    action_params: dict[str, Any]
    """Parameters for the action."""

    response_text: str
    """Raw full response string from the model."""

    conversation_id: str
    """UUID of the current conversation session."""

    tokens_used: int
    """Tokens consumed by this call (cumulative diff)."""

    processing_time: float
    """Wall-clock seconds from user input to parsed response."""

    error: str | None = None
    """Error message if something went wrong, else None."""

    follow_up: str | None = None
    """Optional follow-up statement from JARVIS."""


# ---------------------------------------------------------------------------
# JarvisAgent
# ---------------------------------------------------------------------------


class JarvisAgent:
    """
    Top-level JARVIS orchestrator.

    Responsibilities:
    - Builds per-call context (system prompt + sliding window history).
    - Calls ``AIClient.chat()`` for inference.
    - Parses the response with ``CommandParser``.
    - Publishes action events to the event bus (decoupled from execution).
    - Maintains ``ConversationMemory`` across turns.
    - Supports streaming via ``process_stream()``.
    """

    def __init__(
        self,
        ai_client: AIClient | None = None,
        command_parser: CommandParser | None = None,
        memory_store: Any | None = None,
    ) -> None:
        """
        Initialise the agent.

        Args:
            ai_client:      AsyncOpenAI client wrapper.  Defaults to the
                            process-level singleton.
            command_parser: Response parser.  Defaults to a fresh instance.
            memory_store:   Optional external memory store (any object with an
                            ``async store(key, value)`` and ``async retrieve(key)``
                            interface).  Not required for basic operation.
        """
        self._ai_client: AIClient = ai_client or get_ai_client()
        self._command_parser: CommandParser = command_parser or CommandParser()
        self._memory_store: Any | None = memory_store

        self.conversation_memory: ConversationMemory = ConversationMemory(
            max_turns=settings.MAX_CONVERSATION_HISTORY
        )

        # Conversation identity — reset when the session changes
        self._conversation_id: str = str(uuid.uuid4())

        # Event bus — lazily imported to avoid circular imports at module load
        self._event_bus: Any | None = None

        logger.info(
            "JarvisAgent ready (conversation_id={}).",
            self._conversation_id,
        )

    # ------------------------------------------------------------------
    # process
    # ------------------------------------------------------------------

    async def process(
        self,
        user_input: str,
        conversation_id: str | None = None,
        execute_actions: bool = True,
    ) -> AgentResponse:
        """
        Process a single user input through the full JARVIS pipeline.

        Flow:
        1. Normalise / sanitise the user input.
        2. Add the user message to ConversationMemory.
        3. Build the messages list: system prompt + memory history + user turn.
        4. Call ``AIClient.chat()`` with retry logic.
        5. Parse the raw model response.
        6. Optionally publish an action event to the event bus.
        7. Record the assistant's speech in ConversationMemory.
        8. Return a fully populated ``AgentResponse``.

        Args:
            user_input:      Raw text from the user.
            conversation_id: Override the session UUID.  If None, the agent's
                             current ``_conversation_id`` is used.
            execute_actions: If True and an action is present, publish it to
                             the event bus.  Set to False in tests or dry-runs.

        Returns:
            ``AgentResponse`` dataclass.
        """
        start_time = time.monotonic()
        tokens_before = self._ai_client.total_tokens_used
        conv_id = conversation_id or self._conversation_id

        user_input = user_input.strip()
        if not user_input:
            return AgentResponse(
                speech="I didn't catch that. Could you repeat?",
                action_type=None,
                action_params={},
                response_text="",
                conversation_id=conv_id,
                tokens_used=0,
                processing_time=0.0,
                error="Empty user input.",
            )

        logger.info(
            "JarvisAgent.process: conv={} input_len={}.",
            conv_id,
            len(user_input),
        )

        # 1. Record user turn
        self.conversation_memory.add("user", user_input)

        # 2. Build messages
        messages = self._build_messages(user_input)

        # 3. Call model
        response_text: str
        error: str | None = None
        try:
            result = await self._ai_client.chat(
                messages=messages,
                stream=False,
            )
            # chat() returns str when stream=False
            response_text = str(result)
        except Exception as exc:
            logger.exception("JarvisAgent.process: model call failed — {}.", exc)
            error = str(exc)
            response_text = (
                '{"speech": "I apologise — I encountered a technical difficulty. '
                'Please try again in a moment.", "action": null, "follow_up": null}'
            )

        # 4. Parse
        parsed: ParsedResponse = self._command_parser.parse(response_text)

        # 5. Publish action event
        if parsed.action_type and execute_actions:
            await self._publish_action(parsed, conv_id)

        # 6. Record assistant turn (speech is what JARVIS says)
        self.conversation_memory.add("assistant", parsed.speech)

        # 7. Persist to external memory store if provided
        if self._memory_store is not None:
            await self._persist_turn(user_input, parsed.speech, conv_id)

        tokens_used = self._ai_client.total_tokens_used - tokens_before
        elapsed = time.monotonic() - start_time

        logger.info(
            "JarvisAgent.process: done in {:.2f}s, tokens={}, action={}.",
            elapsed,
            tokens_used,
            parsed.action_type,
        )

        return AgentResponse(
            speech=parsed.speech,
            action_type=parsed.action_type,
            action_params=parsed.action_params,
            response_text=response_text,
            conversation_id=conv_id,
            tokens_used=tokens_used,
            processing_time=elapsed,
            error=error,
            follow_up=parsed.follow_up,
        )

    # ------------------------------------------------------------------
    # process_stream
    # ------------------------------------------------------------------

    async def process_stream(
        self,
        user_input: str,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming variant of ``process()``.

        Yields text token chunks as they arrive from the model.
        Note: when streaming, action parsing and event publishing happen
        after the full response has been accumulated.

        Args:
            user_input: Raw text from the user.

        Yields:
            Partial text strings (token deltas) from the model.
        """
        user_input = user_input.strip()
        if not user_input:
            yield "I didn't catch that. Could you repeat?"
            return

        self.conversation_memory.add("user", user_input)
        messages = self._build_messages(user_input)

        accumulated: list[str] = []

        try:
            stream_or_str = await self._ai_client.chat(
                messages=messages,
                stream=True,
            )
            # When stream=True, chat() returns an AsyncGenerator
            async for chunk in stream_or_str:  # type: ignore[union-attr]
                accumulated.append(chunk)
                yield chunk
        except Exception as exc:
            logger.exception("JarvisAgent.process_stream: error — {}.", exc)
            fallback = " I encountered a technical difficulty. Please try again."
            accumulated.append(fallback)
            yield fallback

        # Post-stream: parse and publish action
        full_response = "".join(accumulated)
        parsed: ParsedResponse = self._command_parser.parse(full_response)

        if parsed.action_type:
            await self._publish_action(parsed, self._conversation_id)

        self.conversation_memory.add("assistant", parsed.speech)

        if self._memory_store is not None:
            await self._persist_turn(user_input, parsed.speech, self._conversation_id)

    # ------------------------------------------------------------------
    # summarize_conversation
    # ------------------------------------------------------------------

    async def summarize_conversation(self) -> str:
        """
        Ask the model to summarise the current conversation.

        Useful for compressing long sessions before persisting them.

        Returns:
            Summary string from the model, or an empty string on failure.
        """
        history = self.conversation_memory.get_history()
        if not history:
            return ""

        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history
        )

        prompt = (
            "Summarise the following conversation in 3-5 sentences. "
            "Capture the key topics, decisions, and any tasks created.\n\n"
            f"{conversation_text}"
        )

        try:
            summary = await self._ai_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise summariser. Return only the "
                            "summary — no preamble or extra commentary."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
                stream=False,
            )
            result: str = str(summary).strip()
            logger.info(
                "summarize_conversation: {} chars summarised.",
                len(result),
            )
            return result
        except Exception as exc:
            logger.warning("summarize_conversation: failed — {}.", exc)
            return ""

    # ------------------------------------------------------------------
    # reset_conversation
    # ------------------------------------------------------------------

    async def reset_conversation(self) -> None:
        """
        Clear conversation memory and generate a new session UUID.

        Optionally persist the old session summary before clearing.
        """
        old_id = self._conversation_id

        # Optionally summarise before clearing
        if self.conversation_memory.messages and self._memory_store is not None:
            try:
                summary = await self.summarize_conversation()
                if summary:
                    await self._memory_store.store(
                        f"conversation_summary:{old_id}", summary
                    )
            except Exception as exc:
                logger.warning("reset_conversation: summary persist failed — {}.", exc)

        self.conversation_memory.clear()
        self._conversation_id = str(uuid.uuid4())
        logger.info(
            "Conversation reset. Old id={}. New id={}.",
            old_id,
            self._conversation_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, user_input: str) -> list[dict[str, Any]]:
        """
        Assemble the full messages list for a model call.

        Order: system prompt → formatted conversation history → user message.
        The user message is intentionally NOT added to ConversationMemory
        here — it was already added in ``process()`` before this call.
        """
        current_time = datetime.now(timezone.utc).strftime(
            "%A, %B %d %Y at %H:%M UTC"
        )
        system_prompt = build_system_prompt(
            user_name="Sir",
            current_time=current_time,
            pending_tasks=0,
        )

        history = self.conversation_memory.get_history()
        # The last entry is the user's current message — already appended.
        # Exclude it from the history block so it appears exactly once.
        if history and history[-1]["role"] == "user":
            context_history = history[:-1]
        else:
            context_history = history

        formatted_history = format_conversation_history(context_history)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *formatted_history,
            {"role": "user", "content": user_input},
        ]
        return messages

    async def _publish_action(
        self, parsed: ParsedResponse, conversation_id: str
    ) -> None:
        """
        Publish a parsed action to the event bus.

        The event bus is imported lazily to avoid circular imports.
        If no event bus is available, the action is logged and silently skipped.

        Event payload:
            {
                "type":            action type string,
                "params":          action params dict,
                "conversation_id": session UUID,
                "speech":          what JARVIS said,
            }
        """
        event_payload = {
            "type": parsed.action_type,
            "params": parsed.action_params,
            "conversation_id": conversation_id,
            "speech": parsed.speech,
        }

        if self._event_bus is None:
            self._event_bus = self._try_load_event_bus()

        if self._event_bus is not None:
            try:
                if asyncio.iscoroutinefunction(getattr(self._event_bus, "publish", None)):
                    await self._event_bus.publish("jarvis.action", event_payload)
                else:
                    publish_fn = getattr(self._event_bus, "publish", None)
                    if publish_fn is not None:
                        publish_fn("jarvis.action", event_payload)
                logger.info(
                    "_publish_action: published action={}.",
                    parsed.action_type,
                )
            except Exception as exc:
                logger.warning(
                    "_publish_action: event bus publish failed — {}.", exc
                )
        else:
            logger.debug(
                "_publish_action: no event bus — action={} params={}.",
                parsed.action_type,
                parsed.action_params,
            )

    def _try_load_event_bus(self) -> Any | None:
        """
        Attempt to import and return the JARVIS event bus singleton.

        Returns None if the event bus module is not yet available so that
        the agent can still function without the desktop layer.
        """
        try:
            from core import event_bus  # type: ignore[import]
            return event_bus
        except ImportError:
            pass
        try:
            from jarvis.core import event_bus  # type: ignore[import]
            return event_bus
        except ImportError:
            pass
        return None

    async def _persist_turn(
        self,
        user_message: str,
        assistant_speech: str,
        conversation_id: str,
    ) -> None:
        """
        Persist the current turn to the external memory store (fire-and-forget).

        Failures are logged but never surfaced to the user.
        """
        try:
            key = f"turn:{conversation_id}:{int(time.time())}"
            value = {
                "user": user_message,
                "assistant": assistant_speech,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            store_fn = getattr(self._memory_store, "store", None)
            if store_fn is not None:
                if asyncio.iscoroutinefunction(store_fn):
                    await store_fn(key, value)
                else:
                    store_fn(key, value)
        except Exception as exc:
            logger.warning("_persist_turn: failed — {}.", exc)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_agent_instance: JarvisAgent | None = None


def get_agent(
    ai_client: AIClient | None = None,
    command_parser: CommandParser | None = None,
    memory_store: Any | None = None,
) -> JarvisAgent:
    """
    Return the process-level JarvisAgent singleton.

    On first call the agent is created with the supplied (or default)
    dependencies.  Subsequent calls return the same instance regardless of
    arguments.

    Args:
        ai_client:      Optional AIClient override.
        command_parser: Optional CommandParser override.
        memory_store:   Optional external memory store.

    Returns:
        ``JarvisAgent`` singleton.
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = JarvisAgent(
            ai_client=ai_client,
            command_parser=command_parser,
            memory_store=memory_store,
        )
    return _agent_instance
