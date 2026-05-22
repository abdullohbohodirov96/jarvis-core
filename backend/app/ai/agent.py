"""
JARVIS main agent — orchestrates the full AI reasoning loop.

JarvisAgent is the top-level object that:
1. Builds the conversation context (system prompt + history + memories)
2. Calls OpenAI with all registered tools
3. Executes any requested tool calls (with parallel execution)
4. Loops back to OpenAI with tool results until the model returns a final answer
5. Extracts tasks and updates memory after each turn
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.client import OpenAIClient, get_openai_client
from app.ai.tools.base import ToolRegistry
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Maximum tool call iterations per request (prevents infinite loops)
_MAX_TOOL_ITERATIONS = 10


# ---------------------------------------------------------------------------
# JarvisAgent
# ---------------------------------------------------------------------------


class JarvisAgent:
    """
    The JARVIS AI agent.

    Handles the full OpenAI function-calling loop for a single user request:
    - Build context
    - Call GPT-4o with tools
    - Execute tool calls concurrently
    - Feed results back and repeat until a final text answer is produced
    - Post-process: extract tasks, update memories

    Usage::

        agent = JarvisAgent(
            user_id=42,
            db_session=db,
            memory_manager=memory_mgr,
            tool_registry=registry,
        )
        response = await agent.run("Remind me to call Alice tomorrow at 9am")
    """

    def __init__(
        self,
        user_id: int,
        db_session: AsyncSession,
        memory_manager: Any | None = None,
        tool_registry: ToolRegistry | None = None,
        openai_client: OpenAIClient | None = None,
    ) -> None:
        self._user_id = user_id
        self._db = db_session
        self._memory_manager = memory_manager
        self._tool_registry = tool_registry or ToolRegistry()
        self._client: OpenAIClient = openai_client or get_openai_client()
        self._context_manager: Any | None = None

        # Lazily set; caller may set after construction
        self._user: Any | None = None

    def set_user(self, user: Any) -> None:
        """Attach the ORM User object for personalised prompts."""
        self._user = user

    def set_context_manager(self, ctx_manager: Any) -> None:
        """Attach a ContextManager instance."""
        self._context_manager = ctx_manager

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    async def run(
        self,
        user_message: str,
        conversation_history: list[dict[str, Any]],
        conversation_id: int | None = None,
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """
        Process a user message through the full JARVIS agent loop.

        Args:
            user_message: The raw text input from the user.
            conversation_history: Prior messages in OpenAI format
                (can be empty for a fresh conversation).
            conversation_id: DB conversation ID (used for context building
                and post-processing persistence).
            stream: If True, returns an AsyncGenerator yielding text chunks.

        Returns:
            Final response string, or AsyncGenerator[str, None] if streaming.
        """
        logger.info(
            "JarvisAgent.run: user=%d conv=%s stream=%s message_len=%d",
            self._user_id,
            conversation_id,
            stream,
            len(user_message),
        )

        if stream:
            return self._run_streaming(
                user_message=user_message,
                conversation_history=conversation_history,
                conversation_id=conversation_id,
            )

        return await self._run_sync(
            user_message=user_message,
            conversation_history=conversation_history,
            conversation_id=conversation_id,
        )

    # ------------------------------------------------------------------
    # _run_sync
    # ------------------------------------------------------------------

    async def _run_sync(
        self,
        user_message: str,
        conversation_history: list[dict[str, Any]],
        conversation_id: int | None,
    ) -> str:
        """Non-streaming agent loop."""
        messages = await self._prepare_messages(
            user_message, conversation_history, conversation_id
        )
        tools = self._tool_registry.to_openai_tools()

        final_response = ""
        iteration = 0

        while iteration < _MAX_TOOL_ITERATIONS:
            iteration += 1

            try:
                if tools:
                    response = await self._client.chat_completion_with_tools(
                        messages=messages,
                        tools=tools,
                        model=settings.OPENAI_MODEL,
                        temperature=0.7,
                    )
                else:
                    raw = await self._client.chat_completion(
                        messages=messages,
                        model=settings.OPENAI_MODEL,
                        temperature=0.7,
                    )
                    final_response = raw or ""
                    break
            except Exception as exc:
                logger.exception("JarvisAgent: OpenAI call failed on iteration %d: %s", iteration, exc)
                final_response = (
                    "I apologise — I encountered an error while processing your request. "
                    "Please try again."
                )
                break

            choice = response.choices[0]
            finish_reason = choice.finish_reason

            # ── Final answer reached ──────────────────────────────────────────
            if finish_reason == "stop" or finish_reason == "length":
                final_response = choice.message.content or ""
                break

            # ── Tool calls requested ─────────────────────────────────────────
            if finish_reason == "tool_calls":
                tool_calls = choice.message.tool_calls
                if not tool_calls:
                    # Model said tool_calls but provided none — treat as stop
                    final_response = choice.message.content or ""
                    break

                # Append assistant message with tool_calls to history
                assistant_msg = {
                    "role": "assistant",
                    "content": choice.message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
                messages.append(assistant_msg)

                # Execute all tool calls in parallel
                tool_results = await self._process_tool_calls(tool_calls)

                # Append tool result messages
                for tr in tool_results:
                    messages.append(tr)

                logger.debug(
                    "JarvisAgent: iteration %d executed %d tool(s)",
                    iteration,
                    len(tool_results),
                )
                continue

            # ── Unexpected finish reason ─────────────────────────────────────
            logger.warning(
                "JarvisAgent: unexpected finish_reason='%s' on iteration %d",
                finish_reason,
                iteration,
            )
            final_response = choice.message.content or ""
            break

        if iteration >= _MAX_TOOL_ITERATIONS:
            logger.warning(
                "JarvisAgent: hit max tool iterations (%d) for user=%d",
                _MAX_TOOL_ITERATIONS,
                self._user_id,
            )
            if not final_response:
                final_response = (
                    "I've completed the tool calls I could, but reached the processing "
                    "limit. Here is my best response based on what I gathered."
                )

        # ── Post-processing ──────────────────────────────────────────────────
        await self._update_memory(user_message, final_response)
        await self._extract_tasks(messages)

        return final_response

    # ------------------------------------------------------------------
    # _run_streaming
    # ------------------------------------------------------------------

    async def _run_streaming(
        self,
        user_message: str,
        conversation_history: list[dict[str, Any]],
        conversation_id: int | None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming agent loop.

        Executes tool calls synchronously (non-streaming) until the model
        produces a final answer, then streams that final answer.
        """
        # Run the tool loop non-streaming to get the final messages list
        messages = await self._prepare_messages(
            user_message, conversation_history, conversation_id
        )
        tools = self._tool_registry.to_openai_tools()

        # Execute tool-call loop (non-streaming phase)
        for iteration in range(_MAX_TOOL_ITERATIONS):
            if not tools:
                break

            try:
                response = await self._client.chat_completion_with_tools(
                    messages=messages,
                    tools=tools,
                    model=settings.OPENAI_MODEL,
                    temperature=0.7,
                )
            except Exception as exc:
                logger.exception("Streaming agent: OpenAI error iteration %d: %s", iteration, exc)
                async def _err_gen() -> AsyncGenerator[str, None]:
                    yield "I encountered an error. Please try again."
                return _err_gen()

            choice = response.choices[0]
            finish_reason = choice.finish_reason

            if finish_reason in ("stop", "length"):
                # No more tools — stream this final content
                content = choice.message.content or ""
                messages.append({"role": "assistant", "content": content})
                break

            if finish_reason == "tool_calls":
                tool_calls = choice.message.tool_calls or []
                if not tool_calls:
                    break

                assistant_msg = {
                    "role": "assistant",
                    "content": choice.message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
                messages.append(assistant_msg)
                tool_results = await self._process_tool_calls(tool_calls)
                for tr in tool_results:
                    messages.append(tr)
                continue

            break

        # Now stream the final answer
        return self._stream_final_answer(messages, user_message)

    async def _stream_final_answer(
        self,
        messages: list[dict[str, Any]],
        user_message: str,
    ) -> AsyncGenerator[str, None]:
        """Yield streaming chunks for the final assistant response."""
        collected: list[str] = []
        try:
            gen = await self._client.chat_completion(
                messages=messages,
                stream=True,
                model=settings.OPENAI_MODEL,
                temperature=0.7,
            )
            async for chunk in gen:  # type: ignore[union-attr]
                collected.append(chunk)
                yield chunk
        except Exception as exc:
            logger.exception("_stream_final_answer error: %s", exc)
            yield "\n[Error: could not complete response]"

        # Post-process after streaming completes
        full_response = "".join(collected)
        await self._update_memory(user_message, full_response)

    # ------------------------------------------------------------------
    # _prepare_messages
    # ------------------------------------------------------------------

    async def _prepare_messages(
        self,
        user_message: str,
        conversation_history: list[dict[str, Any]],
        conversation_id: int | None,
    ) -> list[dict[str, Any]]:
        """
        Build the initial messages list for the first OpenAI call.

        If a ContextManager is available, delegates to it.  Otherwise falls
        back to a simple inline assembly.
        """
        if self._context_manager is not None and conversation_id is not None:
            try:
                return await self._context_manager.build_context(
                    conversation_id=conversation_id,
                    new_message=user_message,
                    user=self._user,
                )
            except Exception as exc:
                logger.warning("_prepare_messages: context manager failed: %s", exc)

        # ── Inline fallback ────────────────────────────────────────────────
        memory_context = ""
        if self._memory_manager is not None:
            try:
                memory_context = await self._memory_manager.get_relevant_context(
                    query=user_message,
                    max_tokens=600,
                )
            except Exception:
                pass

        from app.ai.prompts.system_prompts import (
            JARVIS_SYSTEM_PROMPT,
            build_system_prompt,
        )

        current_time = datetime.now(timezone.utc).strftime("%A, %B %d %Y at %H:%M UTC")

        if self._user is not None:
            system_prompt = build_system_prompt(
                user=self._user,
                memories=memory_context,
                current_time=current_time,
            )
        else:
            system_prompt = JARVIS_SYSTEM_PROMPT.format(
                user_name="User",
                current_time=current_time,
                user_context="",
                memories=memory_context or "No memories stored yet.",
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        # Append provided history
        for msg in conversation_history:
            if msg.get("role") in ("user", "assistant", "system", "tool"):
                messages.append(msg)

        # Append the new user message
        messages.append({"role": "user", "content": user_message})

        return messages

    # ------------------------------------------------------------------
    # _build_system_prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self, user_context: str = "") -> str:
        """
        Build the personalised system prompt for the current user.

        Args:
            user_context: Optional additional context string.

        Returns:
            Rendered system prompt string.
        """
        current_time = datetime.now(timezone.utc).strftime("%A, %B %d %Y at %H:%M UTC")

        if self._user is not None:
            from app.ai.prompts.system_prompts import build_system_prompt

            return build_system_prompt(
                user=self._user,
                memories=user_context,
                current_time=current_time,
            )

        from app.ai.prompts.system_prompts import JARVIS_SYSTEM_PROMPT

        return JARVIS_SYSTEM_PROMPT.format(
            user_name="User",
            current_time=current_time,
            user_context="",
            memories=user_context or "No memories stored yet.",
        )

    # ------------------------------------------------------------------
    # _process_tool_calls
    # ------------------------------------------------------------------

    async def _process_tool_calls(
        self,
        tool_calls: list[Any],
    ) -> list[dict[str, Any]]:
        """
        Execute a batch of tool calls concurrently and return result messages.

        Args:
            tool_calls: List of tool call objects from the OpenAI response.

        Returns:
            List of ``{"role": "tool", ...}`` messages ready to append to
            the conversation.
        """
        async def _execute_one(tc: Any) -> dict[str, Any]:
            tool_name: str = tc.function.name
            raw_args: str = tc.function.arguments or "{}"
            call_id: str = tc.id

            try:
                params: dict[str, Any] = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "_process_tool_calls: JSON parse error for '%s': %s",
                    tool_name,
                    exc,
                )
                params = {}

            logger.info(
                "Executing tool: name=%s call_id=%s params=%s",
                tool_name,
                call_id,
                params,
            )

            result = await self._tool_registry.execute(tool_name, params)

            # Convert result dict to a JSON string for the tool message
            try:
                result_str = json.dumps(result, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                result_str = str(result)

            return {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": result_str,
            }

        # Run all tool calls in parallel
        tasks = [_execute_one(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        tool_messages: list[dict[str, Any]] = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.exception(
                    "_process_tool_calls: tool[%d] raised: %s", i, res
                )
                tc = tool_calls[i]
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": json.dumps({"error": str(res)}),
                    }
                )
            else:
                tool_messages.append(res)  # type: ignore[arg-type]

        return tool_messages

    # ------------------------------------------------------------------
    # _update_memory
    # ------------------------------------------------------------------

    async def _update_memory(
        self,
        user_message: str,
        ai_response: str,
    ) -> None:
        """
        Trigger asynchronous memory extraction from the current exchange.

        Runs in the background — failure does not affect the user response.

        Args:
            user_message: The user's input text.
            ai_response: JARVIS's reply text.
        """
        if self._memory_manager is None:
            return

        conversation_snippet = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ai_response},
        ]

        try:
            await self._memory_manager.extract_and_store_memories(conversation_snippet)
        except Exception as exc:
            # Non-fatal: memory extraction failure should not surface to the user
            logger.warning("_update_memory: extraction failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # _extract_tasks
    # ------------------------------------------------------------------

    async def _extract_tasks(
        self,
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Use AI to find actionable tasks mentioned in the conversation that
        weren't already created via the create_task tool, then persist them.

        Args:
            conversation: Full messages list from the current agent run.

        Returns:
            List of extracted task dicts (for logging/testing).
        """
        from app.ai.prompts.system_prompts import TASK_EXTRACTION_PROMPT

        # Only extract if there are enough messages and no tool-created tasks
        # to avoid duplicate creation
        user_assistant = [
            m for m in conversation
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        if len(user_assistant) < 2:
            return []

        # Check if a create_task tool was already called in this turn
        for msg in conversation:
            if msg.get("role") == "tool" and msg.get("name") == "create_task":
                # Task was already created via tool — don't extract duplicates
                return []

        conv_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in user_assistant
        )

        prompt = TASK_EXTRACTION_PROMPT.format(conversation=conv_text)

        try:
            raw = await self._client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract tasks from conversations. "
                            "Return a JSON object with a 'tasks' array."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.warning("_extract_tasks: AI call failed: %s", exc)
            return []

        try:
            parsed = json.loads(raw)  # type: ignore[arg-type]
            tasks: list[dict[str, Any]] = (
                parsed if isinstance(parsed, list) else parsed.get("tasks", [])
            )
        except (json.JSONDecodeError, TypeError):
            return []

        # Persist each extracted task
        created: list[dict[str, Any]] = []
        for task_data in tasks:
            title = (task_data.get("title") or "").strip()
            if not title:
                continue
            try:
                from app.models.task import Task  # type: ignore[import]

                due_date = None
                raw_due = task_data.get("due_date")
                if raw_due:
                    try:
                        due_date = datetime.fromisoformat(raw_due)
                    except ValueError:
                        pass

                task = Task(
                    user_id=self._user_id,
                    title=title,
                    description=task_data.get("description", ""),
                    priority=task_data.get("priority", "medium"),
                    due_date=due_date,
                    tags=task_data.get("tags", []),
                    status="pending",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                self._db.add(task)
                created.append(task_data)
            except Exception as exc:
                logger.warning("_extract_tasks: failed to persist task: %s", exc)

        if created:
            try:
                await self._db.flush()
                logger.info(
                    "_extract_tasks: auto-created %d task(s) for user=%d",
                    len(created),
                    self._user_id,
                )
            except Exception as exc:
                logger.warning("_extract_tasks: flush failed: %s", exc)

        return created
