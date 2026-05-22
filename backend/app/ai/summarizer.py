"""
Conversation and task summarisation for JARVIS.

Provides:
- ConversationSummarizer.summarize(): compress a long message list into a summary
- ConversationSummarizer.summarize_for_memory(): extract key facts as a list
- ConversationSummarizer.daily_summary(): full daily report for a user
- ConversationSummarizer.generate_task_report(): narrative task status report
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.client import OpenAIClient
from app.ai.prompts.system_prompts import SUMMARIZATION_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConversationSummarizer
# ---------------------------------------------------------------------------


class ConversationSummarizer:
    """
    Generates summaries and structured extracts from JARVIS conversations.

    All public methods are async and use the OpenAI Chat API.
    """

    def __init__(self, openai_client: OpenAIClient) -> None:
        self._client = openai_client

    # ------------------------------------------------------------------
    # summarize
    # ------------------------------------------------------------------

    async def summarize(
        self,
        messages: list[dict[str, Any]],
        max_length: int = 500,
    ) -> str:
        """
        Summarise a list of chat messages into a single concise paragraph.

        Args:
            messages: OpenAI-formatted chat messages (role + content).
            max_length: Approximate maximum word count for the summary.

        Returns:
            Summary string, or empty string on failure.
        """
        if not messages:
            return ""

        # Filter out system messages and tool result messages for summarisation
        user_assistant_msgs = [
            m for m in messages
            if m.get("role") in ("user", "assistant")
            and m.get("content")
        ]

        if not user_assistant_msgs:
            return ""

        conversation_text = self._format_messages_for_summary(user_assistant_msgs)

        prompt = SUMMARIZATION_PROMPT.format(
            max_length=max_length,
            conversation=conversation_text,
        )

        try:
            summary = await self._client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise summarisation assistant. "
                            "Return only the summary text — no preamble."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=600,
            )
            summary = (summary or "").strip()
            logger.debug(
                "summarize: %d messages → %d chars summary",
                len(messages),
                len(summary),
            )
            return summary
        except Exception as exc:
            logger.exception("summarize failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # summarize_for_memory
    # ------------------------------------------------------------------

    async def summarize_for_memory(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Extract a list of key facts from a conversation for memory storage.

        Returns:
            List of dicts with keys: ``content``, ``memory_type``,
            ``importance``, ``tags``.
        """
        if not messages:
            return []

        from app.ai.prompts.system_prompts import MEMORY_EXTRACTION_PROMPT

        user_assistant_msgs = [
            m for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        if not user_assistant_msgs:
            return []

        conversation_text = self._format_messages_for_summary(user_assistant_msgs)
        prompt = MEMORY_EXTRACTION_PROMPT.format(conversation=conversation_text)

        try:
            raw = await self._client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract structured memory facts from conversations. "
                            "Return only a valid JSON object with a 'memories' array."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.exception("summarize_for_memory AI call failed: %s", exc)
            return []

        try:
            parsed = json.loads(raw)  # type: ignore[arg-type]
            if isinstance(parsed, list):
                facts = parsed
            elif isinstance(parsed, dict):
                facts = parsed.get("memories", [])
            else:
                facts = []
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("summarize_for_memory: JSON parse error: %s", exc)
            return []

        # Validate and normalise entries
        validated: list[dict[str, Any]] = []
        for item in facts:
            if not isinstance(item, dict):
                continue
            content = (item.get("content") or "").strip()
            if not content:
                continue
            validated.append(
                {
                    "content": content,
                    "memory_type": item.get("memory_type", "general"),
                    "importance": float(item.get("importance", 0.5)),
                    "tags": item.get("tags", []) if isinstance(item.get("tags"), list) else [],
                }
            )

        logger.debug("summarize_for_memory: extracted %d facts", len(validated))
        return validated

    # ------------------------------------------------------------------
    # daily_summary
    # ------------------------------------------------------------------

    async def daily_summary(
        self,
        user_id: int,
        db_session: AsyncSession,
    ) -> str:
        """
        Generate a comprehensive daily summary for a user.

        Aggregates:
        - All conversations from today
        - Pending and completed tasks
        - Scheduled / upcoming reminders

        Args:
            user_id: The user's DB ID.
            db_session: An active async DB session.

        Returns:
            A Markdown-formatted daily summary string.
        """
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        today_end = today_start + timedelta(days=1)

        # ── Gather data ───────────────────────────────────────────────────────
        tasks_data: list[dict[str, Any]] = []
        messages_data: list[dict[str, Any]] = []
        events_data: list[dict[str, Any]] = []

        try:
            from app.models.task import Task  # type: ignore[import]

            stmt = select(Task).where(
                Task.user_id == user_id,
                Task.status.in_(["pending", "in_progress", "completed"]),
            ).limit(50)
            result = await db_session.execute(stmt)
            db_tasks = result.scalars().all()
            tasks_data = [
                {
                    "title": t.title,
                    "status": t.status,
                    "priority": t.priority,
                    "due_date": t.due_date.isoformat() if t.due_date else None,
                    "description": t.description or "",
                }
                for t in db_tasks
            ]
        except Exception as exc:
            logger.warning("daily_summary: could not fetch tasks: %s", exc)

        try:
            from app.models.conversation import Conversation  # type: ignore[import]
            from app.models.message import Message  # type: ignore[import]

            conv_stmt = select(Conversation).where(
                Conversation.user_id == user_id,
                Conversation.created_at >= today_start,
            )
            conv_result = await db_session.execute(conv_stmt)
            convs = conv_result.scalars().all()
            conv_ids = [c.id for c in convs]

            if conv_ids:
                msg_stmt = (
                    select(Message)
                    .where(
                        Message.conversation_id.in_(conv_ids),
                        Message.role.in_(["user", "assistant"]),
                    )
                    .order_by(Message.created_at.desc())
                    .limit(100)
                )
                msg_result = await db_session.execute(msg_stmt)
                db_msgs = msg_result.scalars().all()
                messages_data = [
                    {
                        "sender": m.role,
                        "text": (m.content or "")[:200],
                        "date": m.created_at.isoformat() if m.created_at else None,
                    }
                    for m in db_msgs
                ]
        except Exception as exc:
            logger.warning("daily_summary: could not fetch messages: %s", exc)

        # ── Build the report via template ──────────────────────────────────────
        from app.ai.prompts.templates import daily_summary_template

        template_output = daily_summary_template(
            tasks=tasks_data,
            messages=messages_data,
            events=events_data,
            date=datetime.now(timezone.utc).strftime("%A, %B %d %Y"),
        )

        # ── AI narrative layer ────────────────────────────────────────────────
        if not tasks_data and not messages_data:
            return template_output

        try:
            narrative_prompt = (
                "You are JARVIS. The following is a structured daily summary. "
                "Add a brief 2-3 sentence narrative at the top that highlights the most "
                "important items for today in your characteristic tone. "
                "Do NOT restate everything — just add the opening narrative paragraph.\n\n"
                f"{template_output}"
            )
            narrative = await self._client.chat_completion(
                messages=[
                    {"role": "user", "content": narrative_prompt},
                ],
                temperature=0.4,
                max_tokens=200,
            )
            if narrative:
                return f"{narrative.strip()}\n\n---\n\n{template_output}"
        except Exception as exc:
            logger.warning("daily_summary: narrative generation failed: %s", exc)

        return template_output

    # ------------------------------------------------------------------
    # generate_task_report
    # ------------------------------------------------------------------

    async def generate_task_report(self, tasks: list[Any]) -> str:
        """
        Generate a human-readable narrative report about a list of tasks.

        Args:
            tasks: List of Task ORM objects or dicts with task fields.

        Returns:
            Narrative report string.
        """
        if not tasks:
            return "No tasks to report."

        # Serialise tasks to a compact text format
        task_lines: list[str] = []
        for t in tasks:
            if isinstance(t, dict):
                title = t.get("title", "Untitled")
                status = t.get("status", "unknown")
                priority = t.get("priority", "medium")
                due = t.get("due_date", "")
            else:
                title = getattr(t, "title", "Untitled")
                status = getattr(t, "status", "unknown")
                priority = getattr(t, "priority", "medium")
                due_dt = getattr(t, "due_date", None)
                due = due_dt.isoformat() if due_dt else ""

            line = f"- [{status.upper()}] {title} (priority: {priority})"
            if due:
                line += f" | due: {due}"
            task_lines.append(line)

        task_text = "\n".join(task_lines)
        prompt = (
            "You are JARVIS. Summarise the following task list into a brief, "
            "clear report (max 150 words). Highlight urgent/overdue items. "
            "Use bullet points. Be concise.\n\n"
            f"Tasks:\n{task_text}\n\nReport:"
        )

        try:
            report = await self._client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are JARVIS, a concise task reporting AI.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            return (report or "").strip()
        except Exception as exc:
            logger.exception("generate_task_report failed: %s", exc)
            return f"Task summary:\n{task_text}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_messages_for_summary(
        messages: list[dict[str, Any]],
    ) -> str:
        """Format messages as a readable conversation transcript."""
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle content arrays (vision messages etc.)
                content = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
