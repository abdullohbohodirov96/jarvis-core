"""
String templates for JARVIS-generated documents and notifications.

These are plain Python string functions (no Jinja2 dependency required).
They produce clean Markdown output suitable for Telegram, email, or in-app
display.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _priority_emoji(priority: str) -> str:
    return {"urgent": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")


def _status_label(status: str) -> str:
    return {"pending": "Pending", "in_progress": "In Progress", "completed": "Done", "cancelled": "Cancelled"}.get(status, status.title())


def _fmt_date(dt: Any) -> str:
    """Format a datetime object or ISO string to a readable date."""
    if dt is None:
        return "No date"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return dt
    try:
        return dt.strftime("%B %d, %Y")
    except AttributeError:
        return str(dt)


def _fmt_datetime(dt: Any) -> str:
    """Format a datetime object or ISO string to a readable date+time."""
    if dt is None:
        return "—"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return dt
    try:
        return dt.strftime("%b %d %Y at %H:%M UTC")
    except AttributeError:
        return str(dt)


# ---------------------------------------------------------------------------
# daily_summary_template
# ---------------------------------------------------------------------------


def daily_summary_template(
    tasks: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    events: list[dict[str, Any]],
    user_name: str = "User",
    date: str | None = None,
) -> str:
    """
    Generate a daily summary document.

    Args:
        tasks: List of task dicts (title, status, priority, due_date).
        messages: List of recent Telegram message dicts (sender, text, date).
        events: List of calendar event dicts (title, start, end).
        user_name: User's display name.
        date: Date string for the summary header. Defaults to today.

    Returns:
        Markdown-formatted daily summary string.
    """
    date_str = date or datetime.now(timezone.utc).strftime("%A, %B %d %Y")

    lines: list[str] = [
        f"# Daily Summary — {date_str}",
        f"*Prepared for {user_name} by JARVIS*",
        "",
    ]

    # ── Tasks ────────────────────────────────────────────────────────────────
    lines.append("## Tasks")
    if not tasks:
        lines.append("_No tasks for today._")
    else:
        pending = [t for t in tasks if t.get("status") not in ("completed", "cancelled")]
        completed = [t for t in tasks if t.get("status") == "completed"]

        if pending:
            lines.append(f"\n**Open ({len(pending)})**")
            for task in sorted(pending, key=lambda t: (
                {"urgent": 0, "high": 1, "medium": 2, "low": 3}.get(t.get("priority", "medium"), 2),
            )):
                emoji = _priority_emoji(task.get("priority", "medium"))
                due = _fmt_date(task.get("due_date"))
                due_str = f" — due {due}" if task.get("due_date") else ""
                lines.append(f"- {emoji} **{task['title']}**{due_str}")
                if task.get("description"):
                    desc_preview = task["description"][:80]
                    lines.append(f"  _{desc_preview}_")

        if completed:
            lines.append(f"\n**Completed today ({len(completed)})**")
            for task in completed:
                lines.append(f"- ✅ ~~{task['title']}~~")

    lines.append("")

    # ── Events ───────────────────────────────────────────────────────────────
    lines.append("## Upcoming Events")
    if not events:
        lines.append("_No events scheduled._")
    else:
        for event in events:
            start = _fmt_datetime(event.get("start"))
            end_str = ""
            if event.get("end"):
                end_str = f" → {_fmt_datetime(event['end'])}"
            lines.append(f"- 📅 **{event['title']}** — {start}{end_str}")
            if event.get("location"):
                lines.append(f"  📍 {event['location']}")

    lines.append("")

    # ── Telegram messages ─────────────────────────────────────────────────────
    lines.append("## Recent Messages")
    if not messages:
        lines.append("_No unread messages._")
    else:
        for msg in messages[:5]:  # cap at 5 in the summary
            sender = msg.get("sender", "Unknown")
            text = msg.get("text", "")
            if len(text) > 100:
                text = text[:97] + "…"
            date_m = _fmt_datetime(msg.get("date"))
            lines.append(f"- **{sender}** ({date_m}): {text}")
        if len(messages) > 5:
            lines.append(f"  _…and {len(messages) - 5} more_")

    lines.append("")
    lines.append("---")
    lines.append(f"*Generated at {datetime.now(timezone.utc).strftime('%H:%M UTC')} by JARVIS*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# task_reminder_template
# ---------------------------------------------------------------------------


def task_reminder_template(
    task: dict[str, Any],
    user_name: str = "User",
    minutes_until_due: int | None = None,
) -> str:
    """
    Generate a task reminder notification.

    Args:
        task: Task dict (id, title, description, priority, due_date, status).
        user_name: User's display name.
        minutes_until_due: Minutes until the task is due (for urgency framing).

    Returns:
        Formatted reminder string.
    """
    priority = task.get("priority", "medium")
    emoji = _priority_emoji(priority)
    title = task.get("title", "Untitled Task")
    description = task.get("description", "")
    due_date = task.get("due_date")

    lines: list[str] = [f"⏰ **Task Reminder** for {user_name}", ""]

    # Urgency framing
    if minutes_until_due is not None:
        if minutes_until_due <= 0:
            urgency = "This task is **overdue**!"
        elif minutes_until_due < 30:
            urgency = f"Due in **{minutes_until_due} minutes**."
        elif minutes_until_due < 120:
            hours = minutes_until_due // 60
            mins = minutes_until_due % 60
            urgency = f"Due in **{hours}h {mins}m**." if mins else f"Due in **{hours} hour{'s' if hours > 1 else ''}**."
        else:
            urgency = f"Due **{_fmt_datetime(due_date)}**."
    elif due_date:
        urgency = f"Due **{_fmt_datetime(due_date)}**."
    else:
        urgency = "No specific due date."

    lines += [
        f"{emoji} **{title}**",
        urgency,
    ]

    if description:
        preview = description[:150] + ("…" if len(description) > 150 else "")
        lines += ["", f"_{preview}_"]

    lines += [
        "",
        f"Priority: **{priority.title()}**",
        f"Status: {_status_label(task.get('status', 'pending'))}",
    ]

    if task.get("tags"):
        tags_str = ", ".join(f"#{t}" for t in task["tags"])
        lines.append(f"Tags: {tags_str}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# weekly_report_template
# ---------------------------------------------------------------------------


def weekly_report_template(
    stats: dict[str, Any],
    user_name: str = "User",
    week_start: str | None = None,
    week_end: str | None = None,
) -> str:
    """
    Generate a weekly productivity report.

    Args:
        stats: Dict with keys:
            - tasks_created (int)
            - tasks_completed (int)
            - tasks_overdue (int)
            - messages_sent (int)
            - messages_received (int)
            - reminders_set (int)
            - top_tags (list[dict] with 'tag' and 'count')
            - completion_rate (float, 0–1)
            - busiest_day (str)
            - completed_tasks (list[dict] with 'title')
            - pending_tasks (list[dict] with 'title', 'priority')
        user_name: User's display name.
        week_start: ISO date string for the start of the week.
        week_end: ISO date string for the end of the week.

    Returns:
        Markdown-formatted weekly report.
    """
    now = datetime.now(timezone.utc)
    week_str = ""
    if week_start and week_end:
        week_str = f"{_fmt_date(week_start)} — {_fmt_date(week_end)}"
    else:
        week_str = now.strftime("Week of %B %d, %Y")

    tasks_created: int = stats.get("tasks_created", 0)
    tasks_completed: int = stats.get("tasks_completed", 0)
    tasks_overdue: int = stats.get("tasks_overdue", 0)
    messages_sent: int = stats.get("messages_sent", 0)
    messages_received: int = stats.get("messages_received", 0)
    reminders_set: int = stats.get("reminders_set", 0)
    completion_rate: float = stats.get("completion_rate", 0.0)
    busiest_day: str = stats.get("busiest_day", "—")
    top_tags: list[dict[str, Any]] = stats.get("top_tags", [])
    completed_tasks: list[dict[str, Any]] = stats.get("completed_tasks", [])
    pending_tasks: list[dict[str, Any]] = stats.get("pending_tasks", [])

    rate_pct = int(completion_rate * 100)
    # Choose a progress bar character based on completion rate
    if rate_pct >= 80:
        performance = "Excellent"
        perf_emoji = "🏆"
    elif rate_pct >= 60:
        performance = "Good"
        perf_emoji = "👍"
    elif rate_pct >= 40:
        performance = "Fair"
        perf_emoji = "📈"
    else:
        performance = "Needs Improvement"
        perf_emoji = "💪"

    lines: list[str] = [
        f"# Weekly Report — {week_str}",
        f"*{user_name}'s productivity summary*",
        "",
        "## Overview",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Tasks Created | {tasks_created} |",
        f"| Tasks Completed | {tasks_completed} |",
        f"| Tasks Overdue | {tasks_overdue} |",
        f"| Completion Rate | {rate_pct}% |",
        f"| Messages Sent | {messages_sent} |",
        f"| Messages Received | {messages_received} |",
        f"| Reminders Set | {reminders_set} |",
        f"| Busiest Day | {busiest_day} |",
        "",
        f"## Performance: {perf_emoji} {performance}",
        "",
    ]

    # Progress bar (20 chars wide)
    filled = int(rate_pct / 5)
    bar = "█" * filled + "░" * (20 - filled)
    lines.append(f"`{bar}` {rate_pct}% completion rate")
    lines.append("")

    # Top tags
    if top_tags:
        lines.append("## Most Active Areas")
        for tag_info in top_tags[:5]:
            tag = tag_info.get("tag", "")
            count = tag_info.get("count", 0)
            lines.append(f"- `#{tag}` — {count} task{'s' if count != 1 else ''}")
        lines.append("")

    # Completed tasks
    if completed_tasks:
        lines.append("## Completed This Week ✅")
        for t in completed_tasks[:10]:
            lines.append(f"- ~~{t.get('title', 'Untitled')}~~")
        if len(completed_tasks) > 10:
            lines.append(f"  _…and {len(completed_tasks) - 10} more_")
        lines.append("")

    # Pending / carry-over tasks
    if pending_tasks:
        lines.append("## Carrying Over to Next Week")
        for t in sorted(
            pending_tasks[:8],
            key=lambda x: {"urgent": 0, "high": 1, "medium": 2, "low": 3}.get(x.get("priority", "medium"), 2),
        ):
            emoji = _priority_emoji(t.get("priority", "medium"))
            lines.append(f"- {emoji} {t.get('title', 'Untitled')}")
        lines.append("")

    lines += [
        "---",
        f"*Report generated {now.strftime('%B %d %Y at %H:%M UTC')} by JARVIS*",
    ]

    return "\n".join(lines)
