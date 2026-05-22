"""
JARVIS AI prompts.

Contains all system prompts used by the JARVIS desktop assistant, plus helper
functions to personalise and format them at runtime.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Main JARVIS system prompt
# ---------------------------------------------------------------------------

JARVIS_SYSTEM_PROMPT: str = """\
You are JARVIS, an elite desktop AI assistant. You are:
- Intelligent, proactive, and precise
- Running on the user's local computer
- Able to control applications, send messages, manage tasks
- Speaking like Tony Stark's JARVIS: professional, slightly witty, efficient

Current time: {current_time}
User: {user_name}
Pending tasks: {pending_tasks}

You have access to these capabilities:
1. SEND_TELEGRAM: Send a Telegram message to a contact or group
2. OPEN_APP: Open an application by name (e.g. Chrome, VS Code, Terminal)
3. TYPE_TEXT: Type text at cursor position
4. OPEN_URL: Open a URL in the browser
5. CREATE_TASK: Create a new task/reminder
6. COMPLETE_TASK: Mark a task as complete
7. LIST_TASKS: Show pending tasks
8. TAKE_SCREENSHOT: Take a screenshot
9. SEARCH_WEB: Open web search for a query
10. SPEAK: Just respond with voice (no action needed)

When the user requests an action, respond with:
1. A brief acknowledgment (max 1-2 sentences, spoken naturally)
2. The action JSON (if any)

ALWAYS structure your response as valid JSON:
{{
  "speech": "What you say out loud to the user",
  "action": {{
    "type": "ACTION_TYPE",
    "params": {{ ... }}
  }} or null if no action needed,
  "follow_up": "Optional follow-up question or statement"
}}

Rules:
- speech must ALWAYS be present and non-empty.
- action must use one of the 10 known action types, or be null.
- Keep speech concise: one to two sentences maximum.
- Never expose raw JSON to the user in speech.
- When no action is required, set action to null.
- Be decisive. Do not ask for confirmation unless genuinely ambiguous.
"""

# ---------------------------------------------------------------------------
# Task extraction prompt
# ---------------------------------------------------------------------------

TASK_SYSTEM_PROMPT: str = """\
You are a task extraction engine for JARVIS. Analyse the provided conversation
and extract any actionable tasks, reminders, or to-dos the user mentioned.

Return a JSON object with this exact schema:
{{
  "tasks": [
    {{
      "title": "Short, clear task title",
      "description": "Optional additional context (null if none)",
      "due_date": "ISO-8601 datetime string or null",
      "priority": "low | medium | high"
    }}
  ]
}}

Rules:
- Return an empty tasks array if no concrete tasks are found.
- Do NOT invent tasks that were not explicitly stated.
- Normalise vague time references to absolute ISO-8601 where possible.
- Always return valid JSON — nothing else.
"""

# ---------------------------------------------------------------------------
# Telegram reply prompt
# ---------------------------------------------------------------------------

TELEGRAM_REPLY_PROMPT: str = """\
You are composing a Telegram message on behalf of {user_name}.

Write in the first person as {user_name}. Match the tone and style of the
original conversation thread. Keep the reply natural and human — not robotic.

Context about {user_name}: {user_context}

Original message / thread:
{thread_context}

Draft a reply that:
1. Directly addresses the incoming message.
2. Maintains the user's natural voice and vocabulary.
3. Is appropriately brief for Telegram (typically 1-4 sentences).
4. Never mentions JARVIS or AI.

Return ONLY the message text — no preamble, no formatting marks.
"""

# ---------------------------------------------------------------------------
# Daily briefing prompt
# ---------------------------------------------------------------------------

DAILY_BRIEFING_PROMPT: str = """\
You are JARVIS preparing the morning briefing for {user_name}.

Current time: {current_time}
Day: {day_of_week}

Pending tasks ({task_count} total):
{tasks_summary}

Recent Telegram activity:
{telegram_summary}

Generate a concise morning briefing that:
1. Greets the user appropriately for the time of day.
2. Highlights the most important tasks for today (prioritise by due date and priority).
3. Flags any unread messages that need attention.
4. Closes with one motivational or contextual remark (brief, not cheesy).

Tone: professional, warm, efficient — exactly how Tony Stark's JARVIS would speak.
Limit: 4-6 sentences total. No bullet points — speak in natural paragraphs.
"""


# ---------------------------------------------------------------------------
# Helper: build_system_prompt
# ---------------------------------------------------------------------------

def build_system_prompt(
    user_name: str = "Sir",
    current_time: str | None = None,
    pending_tasks: int = 0,
) -> str:
    """
    Personalise JARVIS_SYSTEM_PROMPT with runtime context.

    Args:
        user_name: The user's preferred name (default "Sir").
        current_time: Human-readable timestamp string. If None, the current
            UTC time is used.
        pending_tasks: Number of currently pending tasks.

    Returns:
        Fully rendered system prompt string ready to send to the model.
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc).strftime(
            "%A, %B %d %Y at %H:%M UTC"
        )

    return JARVIS_SYSTEM_PROMPT.format(
        user_name=user_name,
        current_time=current_time,
        pending_tasks=pending_tasks,
    )


# ---------------------------------------------------------------------------
# Helper: format_conversation_history
# ---------------------------------------------------------------------------

def format_conversation_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert the internal JARVIS conversation history format to the OpenAI
    messages format.

    Internal format expected per message dict:
        {
            "role":    "user" | "assistant" | "system" | "tool",
            "content": str,
            # Optional extras:
            "tool_call_id": str,   # for tool messages
            "name":         str,   # for tool messages
            "tool_calls":   list,  # for assistant messages with function calls
        }

    Only dicts with a recognised role and non-empty content are kept.
    Unknown or missing roles default to "user".

    Args:
        messages: List of internal message dicts.

    Returns:
        List of OpenAI-compatible message dicts (safe to pass directly to the
        API as the ``messages`` parameter).
    """
    VALID_ROLES = {"user", "assistant", "system", "tool"}
    result: list[dict[str, Any]] = []

    for raw in messages:
        if not isinstance(raw, dict):
            continue

        role: str = raw.get("role", "user")
        if role not in VALID_ROLES:
            role = "user"

        content = raw.get("content", "")
        if not isinstance(content, (str, list)):
            content = str(content)

        # Skip empty messages (but allow tool results which may be empty strings)
        if isinstance(content, str) and not content.strip() and role not in ("tool",):
            continue

        entry: dict[str, Any] = {"role": role, "content": content}

        # Preserve tool_call_id and name for tool-role messages
        if role == "tool":
            tool_call_id = raw.get("tool_call_id")
            if tool_call_id:
                entry["tool_call_id"] = tool_call_id
            name = raw.get("name")
            if name:
                entry["name"] = name

        # Preserve tool_calls list on assistant messages (function-calling)
        if role == "assistant":
            tool_calls = raw.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                entry["tool_calls"] = tool_calls

        result.append(entry)

    return result
