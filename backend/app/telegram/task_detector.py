"""
AI-powered task extraction from Telegram messages.

TelegramTaskDetector uses an OpenAI chat model to identify action items,
commitments, deadlines, questions, and other intent signals in raw Telegram
message text.  It also supports bulk processing of a full chat's recent
history.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

from backend.app.core.exceptions import AIException

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.app.ai.client import OpenAIClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent constants
# ---------------------------------------------------------------------------

INTENT_TASK = "task"
INTENT_REMINDER = "reminder"
INTENT_QUESTION = "question"
INTENT_INFORMATION = "information"
INTENT_SOCIAL = "social"
INTENT_OTHER = "other"

_VALID_INTENTS = frozenset(
    {INTENT_TASK, INTENT_REMINDER, INTENT_QUESTION, INTENT_INFORMATION, INTENT_SOCIAL, INTENT_OTHER}
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_TASK_DETECTION_SYSTEM = """You are a smart personal assistant that analyses Telegram messages to identify actionable items.

Your job is to detect:
1. Tasks / to-dos (things someone needs to do)
2. Commitments (promises made by any party)
3. Reminders (things to remember or follow up on)
4. Deadlines (explicit or implied due dates)

Return ONLY a valid JSON array.  Each element is an object with:
- "title": str — concise task title (max 100 chars)
- "description": str — fuller context from the message
- "priority": "low" | "medium" | "high" | "urgent"
- "due_date": str | null — ISO-8601 date/datetime if detectable, else null
- "source_text": str — the excerpt that prompted this task
- "confidence": float — 0.0–1.0 confidence that this is a genuine task

Return [] if no actionable items are found.  Output MUST be valid JSON, nothing else."""

_INTENT_CLASSIFICATION_SYSTEM = """Classify the intent of a Telegram message.

Return exactly one of: task, reminder, question, information, social, other

Rules:
- task: message contains an action item, to-do, or request to do something
- reminder: message is about remembering something or a follow-up
- question: message is primarily a question
- information: message shares factual information
- social: casual chat, greetings, reactions
- other: anything that doesn't fit above

Output ONLY the single word label, lowercase, no punctuation."""

_DEADLINE_EXTRACTION_SYSTEM = """Extract all date/time references from the text.

Return a JSON array of ISO-8601 datetime strings (UTC).
Use the current date as anchor: {now_iso}.
Return [] if no dates are found.
Output MUST be valid JSON only."""


class TelegramTaskDetector:
    """
    Extracts actionable tasks and classifies intents in Telegram messages
    using an OpenAI LLM.

    Args:
        openai_client: The OpenAIClient singleton.
        db_session:    Async SQLAlchemy session for persisting detected tasks.
    """

    def __init__(self, openai_client: OpenAIClient, db_session: AsyncSession) -> None:
        self._ai = openai_client
        self._db = db_session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def detect_tasks(
        self,
        messages: list[dict[str, Any]],
        user_id: UUID,
    ) -> list[dict[str, Any]]:
        """
        Analyse a list of message dicts and return detected tasks.

        Args:
            messages: List of message dicts (must contain ``"text"`` key).
            user_id:  UUID of the owning user (attached to detected tasks).

        Returns:
            List of task dicts ready for TaskService.create_task().
        """
        if not messages:
            return []

        # Filter to messages with meaningful text
        texts = [
            m.get("text", "").strip()
            for m in messages
            if m.get("text", "").strip()
        ]
        if not texts:
            return []

        prompt = self._build_task_detection_prompt(texts)
        raw_tasks = await self._call_ai_for_tasks(prompt)

        # Attach user_id and source metadata
        for task in raw_tasks:
            task["user_id"] = str(user_id)
            task.setdefault("priority", "medium")
            task.setdefault("confidence", 0.5)

        logger.info(
            "detect_tasks: analysed=%d messages, found=%d tasks",
            len(texts),
            len(raw_tasks),
        )
        return raw_tasks

    async def detect_single_message(
        self,
        message: str,
        context: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Detect tasks in a single message string with optional surrounding context.

        Args:
            message: The message text to analyse.
            context: Optional list of surrounding messages for context.

        Returns:
            List of task dicts (may be empty).
        """
        if not message.strip():
            return []

        context = context or []
        texts: list[str] = context[-5:] + [message]  # up to 6 messages of context
        prompt = self._build_task_detection_prompt(texts)
        tasks = await self._call_ai_for_tasks(prompt)
        logger.debug(
            "detect_single_message: found=%d tasks in message len=%d",
            len(tasks), len(message),
        )
        return tasks

    async def extract_deadlines(self, text: str) -> list[datetime]:
        """
        Extract explicit or implied deadline dates from a text string.

        Args:
            text: Raw message text.

        Returns:
            List of UTC-aware datetime objects (deduplicated, sorted ascending).
        """
        if not text.strip():
            return []

        now_iso = datetime.now(timezone.utc).isoformat()
        system = _DEADLINE_EXTRACTION_SYSTEM.format(now_iso=now_iso)

        try:
            response = await self._ai.chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            raw: str = response if isinstance(response, str) else ""
            parsed: list[str] = json.loads(self._strip_code_fence(raw))
            deadlines: list[datetime] = []
            for dt_str in parsed:
                try:
                    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    deadlines.append(dt)
                except ValueError:
                    continue
            deadlines = sorted(set(deadlines))
            logger.debug("extract_deadlines: found=%d dates in text", len(deadlines))
            return deadlines
        except Exception as exc:  # noqa: BLE001
            logger.warning("extract_deadlines failed: %s", exc)
            return []

    async def classify_message_intent(self, message: str) -> str:
        """
        Classify the intent of a single message.

        Args:
            message: Raw message text.

        Returns:
            One of: task, reminder, question, information, social, other.
        """
        if not message.strip():
            return INTENT_OTHER

        # Fast heuristics before hitting the LLM
        heuristic = self._heuristic_intent(message)
        if heuristic:
            return heuristic

        try:
            response = await self._ai.chat_completion(
                messages=[
                    {"role": "system", "content": _INTENT_CLASSIFICATION_SYSTEM},
                    {"role": "user", "content": message[:1000]},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            label = (response or "").strip().lower().rstrip(".")
            return label if label in _VALID_INTENTS else INTENT_OTHER
        except Exception as exc:  # noqa: BLE001
            logger.warning("classify_message_intent failed: %s", exc)
            return INTENT_OTHER

    async def batch_process_chat(
        self,
        chat_id: int | str,
        user_id: UUID,
        days_back: int = 7,
    ) -> list[dict[str, Any]]:
        """
        Retrieve and analyse recent messages from a chat to surface missed tasks.

        Args:
            chat_id:   The Telegram chat to analyse.
            user_id:   Owner user UUID.
            days_back: How many days of history to examine (default 7).

        Returns:
            All detected tasks from the analysed message batch.
        """
        # Import here to avoid circular imports
        from backend.app.telegram.client import get_telegram_client

        client = await get_telegram_client()
        since = datetime.now(timezone.utc) - timedelta(days=days_back)

        # Fetch messages in batches of 200 to stay within API limits
        all_tasks: list[dict[str, Any]] = []
        batch_size = 200
        offset_id = 0

        while True:
            messages = await client.get_messages(
                chat_id=chat_id,
                limit=batch_size,
                offset_id=offset_id,
            )
            if not messages:
                break

            # Filter by date
            recent = [
                m for m in messages
                if m.get("date") and datetime.fromisoformat(
                    m["date"].replace("Z", "+00:00")
                ) >= since
            ]
            if not recent:
                break

            batch_tasks = await self.detect_tasks(recent, user_id)
            all_tasks.extend(batch_tasks)

            if len(messages) < batch_size:
                break
            offset_id = messages[-1]["id"]

        logger.info(
            "batch_process_chat: chat=%s days=%d found=%d tasks",
            chat_id, days_back, len(all_tasks),
        )
        return all_tasks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_task_detection_prompt(self, messages: list[str]) -> str:
        """
        Construct the user-facing prompt for task detection.

        Args:
            messages: List of message strings to include (newest last).

        Returns:
            Formatted prompt string.
        """
        numbered = "\n".join(
            f"[{i + 1}] {msg}" for i, msg in enumerate(messages)
        )
        return (
            f"Analyse the following Telegram messages and extract all actionable tasks, "
            f"commitments, or reminders:\n\n{numbered}"
        )

    async def _call_ai_for_tasks(self, prompt: str) -> list[dict[str, Any]]:
        """
        Call the AI model and parse the task JSON response.

        Args:
            prompt: The user-turn prompt.

        Returns:
            List of parsed task dicts (empty list on any failure).
        """
        try:
            response = await self._ai.chat_completion(
                messages=[
                    {"role": "system", "content": _TASK_DETECTION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            raw: str = response if isinstance(response, str) else ""

            # The model may return {"tasks": [...]} or directly [...]
            cleaned = self._strip_code_fence(raw)
            parsed = json.loads(cleaned)

            if isinstance(parsed, list):
                tasks = parsed
            elif isinstance(parsed, dict):
                # Try common wrapper keys
                for key in ("tasks", "items", "results", "data"):
                    if isinstance(parsed.get(key), list):
                        tasks = parsed[key]
                        break
                else:
                    tasks = []
            else:
                tasks = []

            # Validate and sanitise
            valid_tasks: list[dict[str, Any]] = []
            for item in tasks:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                if not title:
                    continue
                valid_tasks.append(
                    {
                        "title": title[:200],
                        "description": str(item.get("description", ""))[:1000],
                        "priority": self._normalise_priority(item.get("priority", "medium")),
                        "due_date": item.get("due_date"),
                        "source_text": str(item.get("source_text", ""))[:500],
                        "confidence": float(item.get("confidence", 0.5)),
                    }
                )
            return valid_tasks

        except json.JSONDecodeError as exc:
            logger.warning("Task detection JSON parse error: %s", exc)
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning("Task detection AI call failed: %s", exc)
            return []

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Remove markdown code fences from AI response text."""
        text = text.strip()
        # Remove ```json ... ``` or ``` ... ```
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _normalise_priority(priority: Any) -> str:
        """Map any priority string to one of: low, medium, high, urgent."""
        mapping = {
            "low": "low",
            "normal": "medium",
            "medium": "medium",
            "high": "high",
            "critical": "urgent",
            "urgent": "urgent",
            "asap": "urgent",
        }
        return mapping.get(str(priority).lower().strip(), "medium")

    @staticmethod
    def _heuristic_intent(message: str) -> str | None:
        """
        Fast regex-based intent pre-classifier to avoid unnecessary LLM calls.

        Returns an intent string or None if heuristic doesn't apply.
        """
        lower = message.lower()
        question_patterns = [r"\?$", r"\bwhat\b", r"\bhow\b", r"\bwhen\b", r"\bwhere\b",
                             r"\bwhy\b", r"\bwho\b", r"\bcan you\b", r"\bcould you\b"]
        task_patterns = [r"\bplease\b", r"\bcould you\b", r"\bcan you\b", r"\bneed to\b",
                        r"\bshould\b", r"\btodo\b", r"\bto-do\b", r"\bremind\b", r"\bdeadline\b"]
        social_patterns = [r"^(hi|hey|hello|sup|yo|thanks|thank you|ok|okay|bye|lol|haha)\b"]

        for pat in question_patterns:
            if re.search(pat, lower):
                return INTENT_QUESTION
        for pat in social_patterns:
            if re.search(pat, lower):
                return INTENT_SOCIAL
        for pat in task_patterns:
            if re.search(pat, lower):
                return INTENT_TASK
        return None
