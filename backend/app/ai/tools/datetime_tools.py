"""
DateTime and reminder tools for the JARVIS agent.

Provides tools for current time lookups, date arithmetic, and reminder creation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.ai.tools.base import BaseTool

logger = logging.getLogger(__name__)

# Supported relative unit names normalised to timedelta kwargs
_UNIT_MAP: dict[str, str] = {
    "second": "seconds",
    "seconds": "seconds",
    "minute": "minutes",
    "minutes": "minutes",
    "hour": "hours",
    "hours": "hours",
    "day": "days",
    "days": "days",
    "week": "weeks",
    "weeks": "weeks",
}


def _get_tz(tz_name: str | None) -> timezone | ZoneInfo:
    """Return a timezone object for the given IANA timezone name or UTC."""
    if not tz_name:
        return timezone.utc
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, Exception):
        logger.warning("Unknown timezone '%s', falling back to UTC.", tz_name)
        return timezone.utc


# ---------------------------------------------------------------------------
# GetCurrentTimeTool
# ---------------------------------------------------------------------------


class GetCurrentTimeTool(BaseTool):
    """Return the current date and time."""

    name = "get_current_time"
    description = (
        "Get the current date and time. Optionally specify a timezone. "
        "Use this whenever the user asks what time it is, what day it is, "
        "or any time-relative calculation needs a starting point."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": (
                    "IANA timezone name (e.g. 'America/New_York', 'Europe/London', "
                    "'Asia/Tashkent'). Defaults to UTC."
                ),
            },
            "format": {
                "type": "string",
                "description": (
                    "Output format: 'iso' for ISO 8601, 'human' for readable string, "
                    "'unix' for Unix timestamp. Defaults to 'human'."
                ),
                "enum": ["iso", "human", "unix"],
            },
        },
        "required": [],
    }

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        tz_name: str | None = params.get("timezone")
        fmt: str = params.get("format", "human")

        tz = _get_tz(tz_name)
        now = datetime.now(tz=tz)

        if fmt == "iso":
            time_str = now.isoformat()
        elif fmt == "unix":
            time_str = str(int(now.timestamp()))
        else:
            time_str = now.strftime("%A, %B %d %Y at %H:%M:%S %Z")

        return {
            "success": True,
            "current_time": time_str,
            "timezone": tz_name or "UTC",
            "iso": now.isoformat(),
            "unix_timestamp": int(now.timestamp()),
            "day_of_week": now.strftime("%A"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
        }


# ---------------------------------------------------------------------------
# SetReminderTool
# ---------------------------------------------------------------------------


class SetReminderTool(BaseTool):
    """Create a reminder for the user at a specific time."""

    name = "set_reminder"
    description = (
        "Create a reminder that will notify the user at a specified time. "
        "Use this when the user asks to be reminded of something at a future time."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short title or subject of the reminder.",
            },
            "message": {
                "type": "string",
                "description": "Detailed reminder message the user will receive.",
            },
            "remind_at": {
                "type": "string",
                "description": (
                    "ISO 8601 datetime for when to send the reminder "
                    "(e.g. '2024-12-31T09:00:00Z'). Must be in the future."
                ),
            },
            "recurrence": {
                "type": "string",
                "enum": ["none", "daily", "weekly", "monthly"],
                "description": "How often the reminder repeats. Defaults to 'none'.",
            },
            "channel": {
                "type": "string",
                "enum": ["app", "telegram", "both"],
                "description": "How to deliver the reminder. Defaults to 'app'.",
            },
        },
        "required": ["title", "remind_at"],
    }

    def __init__(self, user_id: int, db_session: AsyncSession) -> None:
        self._user_id = user_id
        self._db = db_session

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        title: str = params["title"]
        message: str = params.get("message", title)
        remind_at_str: str = params["remind_at"]
        recurrence: str = params.get("recurrence", "none")
        channel: str = params.get("channel", "app")

        # Parse and validate remind_at
        try:
            remind_at = datetime.fromisoformat(remind_at_str)
            if remind_at.tzinfo is None:
                remind_at = remind_at.replace(tzinfo=timezone.utc)
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid datetime format: '{remind_at_str}'. Use ISO 8601.",
            }

        now = datetime.now(timezone.utc)
        if remind_at <= now:
            return {
                "success": False,
                "error": "Reminder time must be in the future.",
            }

        delta = remind_at - now
        human_delta = _format_timedelta(delta)

        try:
            from backend.app.models.reminder import Reminder  # type: ignore[import]

            reminder = Reminder(
                user_id=self._user_id,
                title=title,
                message=message,
                remind_at=remind_at,
                recurrence=recurrence,
                channel=channel,
                status="pending",
                created_at=now,
            )
            self._db.add(reminder)
            await self._db.flush()
            await self._db.refresh(reminder)

            logger.info(
                "SetReminderTool: created reminder id=%s at %s",
                reminder.id,
                remind_at.isoformat(),
            )
            return {
                "success": True,
                "reminder_id": reminder.id,
                "title": title,
                "remind_at": remind_at.isoformat(),
                "recurrence": recurrence,
                "channel": channel,
                "message": f"Reminder '{title}' set for {remind_at.strftime('%B %d at %H:%M UTC')} ({human_delta}).",
            }
        except ImportError:
            logger.warning("SetReminderTool: Reminder model unavailable")
            return {
                "success": True,
                "reminder_id": None,
                "title": title,
                "remind_at": remind_at.isoformat(),
                "recurrence": recurrence,
                "channel": channel,
                "message": (
                    f"Reminder '{title}' noted for "
                    f"{remind_at.strftime('%B %d at %H:%M UTC')} ({human_delta}). "
                    "Persistence unavailable."
                ),
            }
        except Exception as exc:
            logger.exception("SetReminderTool failed: %s", exc)
            return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# CalculateDateTool
# ---------------------------------------------------------------------------


class CalculateDateTool(BaseTool):
    """Perform date and time arithmetic."""

    name = "calculate_date"
    description = (
        "Perform date arithmetic: add or subtract time from a date, "
        "calculate the difference between two dates, or find what day "
        "of the week a date falls on. Use this for any date-related calculations."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "difference", "day_of_week", "format"],
                "description": (
                    "'add'/'subtract': add/subtract time from a date. "
                    "'difference': compute days between two dates. "
                    "'day_of_week': find the weekday of a date. "
                    "'format': convert a date to a human-readable string."
                ),
            },
            "date": {
                "type": "string",
                "description": (
                    "Base date in ISO 8601 format or 'now' for current time."
                ),
            },
            "date2": {
                "type": "string",
                "description": "Second date (required for 'difference' operation).",
            },
            "amount": {
                "type": "number",
                "description": "Amount of time to add/subtract (used with 'add'/'subtract').",
            },
            "unit": {
                "type": "string",
                "enum": ["seconds", "minutes", "hours", "days", "weeks"],
                "description": "Time unit for 'add'/'subtract' operations.",
            },
            "timezone": {
                "type": "string",
                "description": "IANA timezone for result display. Defaults to UTC.",
            },
            "output_format": {
                "type": "string",
                "description": (
                    "strftime format string for the output, e.g. '%Y-%m-%d'. "
                    "Defaults to ISO 8601."
                ),
            },
        },
        "required": ["operation"],
    }

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        operation: str = params["operation"]
        tz = _get_tz(params.get("timezone"))
        output_fmt: str | None = params.get("output_format")

        def _parse_date(s: str | None) -> datetime | None:
            if not s:
                return None
            if s.lower() == "now":
                return datetime.now(timezone.utc)
            try:
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                return None

        def _fmt(dt: datetime) -> str:
            dt_local = dt.astimezone(tz)
            if output_fmt:
                return dt_local.strftime(output_fmt)
            return dt_local.isoformat()

        # ── add / subtract ────────────────────────────────────────────────────
        if operation in ("add", "subtract"):
            base_str: str = params.get("date", "now")
            base = _parse_date(base_str)
            if base is None:
                return {"success": False, "error": f"Cannot parse date: '{base_str}'"}

            amount: float = float(params.get("amount", 1))
            unit: str = _UNIT_MAP.get(params.get("unit", "days"), "days")

            if operation == "subtract":
                amount = -amount

            delta = timedelta(**{unit: amount})
            result_dt = base + delta

            return {
                "success": True,
                "operation": operation,
                "input_date": base.isoformat(),
                "result_date": _fmt(result_dt),
                "result_iso": result_dt.isoformat(),
                "delta": f"{abs(amount)} {unit}",
            }

        # ── difference ───────────────────────────────────────────────────────
        if operation == "difference":
            date1_str: str = params.get("date", "now")
            date2_str: str | None = params.get("date2")

            d1 = _parse_date(date1_str)
            d2 = _parse_date(date2_str)

            if d1 is None:
                return {"success": False, "error": f"Cannot parse 'date': '{date1_str}'"}
            if d2 is None:
                return {"success": False, "error": f"Cannot parse 'date2': '{date2_str}'"}

            diff = d2 - d1
            total_seconds = int(diff.total_seconds())
            abs_seconds = abs(total_seconds)

            return {
                "success": True,
                "operation": "difference",
                "date1": d1.isoformat(),
                "date2": d2.isoformat(),
                "total_seconds": total_seconds,
                "total_minutes": total_seconds // 60,
                "total_hours": total_seconds // 3600,
                "total_days": diff.days,
                "human": _format_timedelta(timedelta(seconds=abs_seconds)),
                "direction": "future" if total_seconds >= 0 else "past",
            }

        # ── day_of_week ───────────────────────────────────────────────────────
        if operation == "day_of_week":
            date_str: str = params.get("date", "now")
            dt = _parse_date(date_str)
            if dt is None:
                return {"success": False, "error": f"Cannot parse date: '{date_str}'"}

            local_dt = dt.astimezone(tz)
            return {
                "success": True,
                "operation": "day_of_week",
                "date": local_dt.strftime("%Y-%m-%d"),
                "day_of_week": local_dt.strftime("%A"),
                "day_number": local_dt.weekday(),  # Monday=0, Sunday=6
                "iso_weekday": local_dt.isoweekday(),  # Monday=1, Sunday=7
                "week_number": local_dt.strftime("%U"),
            }

        # ── format ────────────────────────────────────────────────────────────
        if operation == "format":
            date_str_f: str = params.get("date", "now")
            dt = _parse_date(date_str_f)
            if dt is None:
                return {"success": False, "error": f"Cannot parse date: '{date_str_f}'"}

            fmt_str = output_fmt or "%A, %B %d %Y at %H:%M %Z"
            local_dt = dt.astimezone(tz)
            return {
                "success": True,
                "operation": "format",
                "input": date_str_f,
                "formatted": local_dt.strftime(fmt_str),
                "iso": local_dt.isoformat(),
            }

        return {"success": False, "error": f"Unknown operation: '{operation}'"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_timedelta(delta: timedelta) -> str:
    """Return a human-readable description of a timedelta."""
    total_seconds = int(abs(delta.total_seconds()))

    if total_seconds < 60:
        return f"{total_seconds} second{'s' if total_seconds != 1 else ''}"
    if total_seconds < 3600:
        minutes = total_seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    if total_seconds < 86400:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        h_str = f"{hours} hour{'s' if hours != 1 else ''}"
        if minutes:
            h_str += f" {minutes} minute{'s' if minutes != 1 else ''}"
        return h_str

    days = total_seconds // 86400
    if days < 7:
        return f"{days} day{'s' if days != 1 else ''}"
    if days < 30:
        weeks = days // 7
        rem_days = days % 7
        w_str = f"{weeks} week{'s' if weeks != 1 else ''}"
        if rem_days:
            w_str += f" {rem_days} day{'s' if rem_days != 1 else ''}"
        return w_str
    if days < 365:
        months = days // 30
        return f"about {months} month{'s' if months != 1 else ''}"

    years = days // 365
    return f"about {years} year{'s' if years != 1 else ''}"
