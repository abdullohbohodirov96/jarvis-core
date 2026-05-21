"""
Validation utilities for JARVIS.

Provides pure functions (no external HTTP calls, no DB) that validate and
sanitise individual values before they enter the domain layer.
"""

from __future__ import annotations

import html
import re
import unicodedata
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Phone number
# ---------------------------------------------------------------------------

# Accepts international format: optional +, 7-15 digits (ITU-T E.164)
_PHONE_RE = re.compile(r"^\+?[1-9]\d{6,14}$")


def validate_phone_number(phone: str) -> bool:
    """Return True if *phone* looks like a valid international phone number.

    The validation is intentionally lenient (E.164 format check only) and
    does not perform carrier lookups.

    Args:
        phone: Phone number string, optionally starting with '+'.

    Returns:
        ``True`` if the number passes the basic format check.

    Examples::

        >>> validate_phone_number("+14155552671")
        True
        >>> validate_phone_number("not-a-phone")
        False
    """
    if not isinstance(phone, str):
        return False
    stripped = re.sub(r"[\s\-().]+", "", phone)
    return bool(_PHONE_RE.match(stripped))


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

# RFC-5321 simplified regex — good enough for most practical purposes.
_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$",
    re.IGNORECASE,
)


def validate_email(email: str) -> bool:
    """Return True if *email* is a plausibly valid email address.

    Args:
        email: Email address string.

    Returns:
        ``True`` if the basic structure is valid.

    Examples::

        >>> validate_email("user@example.com")
        True
        >>> validate_email("not-an-email")
        False
    """
    if not isinstance(email, str):
        return False
    if len(email) > 254:
        return False
    return bool(_EMAIL_RE.match(email.strip()))


# ---------------------------------------------------------------------------
# Text sanitisation
# ---------------------------------------------------------------------------

# Characters that are dangerous in SQL/HTML/shell contexts
_DANGEROUS_CHARS_RE = re.compile(r"[<>\"\\';\x00-\x1f\x7f]")


def sanitize_text(text: str, max_length: int = 10_000) -> str:
    """Clean *text* by removing control characters and HTML-escaping specials.

    Steps performed:
      1. Unicode normalise to NFC.
      2. Remove ASCII control characters (0x00–0x1F, 0x7F).
      3. Strip leading/trailing whitespace.
      4. HTML-escape ``<``, ``>``, ``&``, ``"``, ``'``.
      5. Truncate to *max_length*.

    Args:
        text:       Input string.
        max_length: Maximum length of the returned string.

    Returns:
        Sanitised string.
    """
    if not isinstance(text, str):
        text = str(text)

    # NFC normalisation ensures consistent codepoints
    text = unicodedata.normalize("NFC", text)

    # Remove ASCII control characters except tab (0x09), LF (0x0A), CR (0x0D)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    text = text.strip()

    # HTML-escape to neutralise XSS vectors
    text = html.escape(text, quote=True)

    return text[:max_length]


def strip_html(text: str) -> str:
    """Remove HTML tags from *text* without escaping the remaining content.

    Args:
        text: HTML-formatted string.

    Returns:
        Plain text with tags removed.
    """
    return re.sub(r"<[^>]+>", "", text)


# ---------------------------------------------------------------------------
# Task priority
# ---------------------------------------------------------------------------

_VALID_PRIORITIES = {"low", "medium", "high", "urgent"}


def validate_task_priority(priority: str) -> bool:
    """Return True if *priority* is one of the recognised task priority levels.

    Args:
        priority: Priority string (case-insensitive).

    Returns:
        ``True`` if valid.

    Examples::

        >>> validate_task_priority("high")
        True
        >>> validate_task_priority("URGENT")
        True
        >>> validate_task_priority("super-high")
        False
    """
    if not isinstance(priority, str):
        return False
    return priority.lower() in _VALID_PRIORITIES


# ---------------------------------------------------------------------------
# Datetime string
# ---------------------------------------------------------------------------

_DATETIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%S%z",   # ISO with tz offset
    "%Y-%m-%dT%H:%M:%SZ",    # ISO UTC Z suffix
    "%Y-%m-%dT%H:%M:%S",     # ISO no tz (assumed UTC)
    "%Y-%m-%d %H:%M:%S",     # SQL-style
    "%Y-%m-%d",              # date only
    "%d/%m/%Y %H:%M",        # European datetime
    "%d/%m/%Y",              # European date
    "%m/%d/%Y %H:%M",        # US datetime
    "%m/%d/%Y",              # US date
]


def validate_datetime_string(dt_str: str) -> Optional[datetime]:
    """Try to parse *dt_str* into a timezone-aware UTC datetime.

    Tries several common formats; returns ``None`` if none match.

    Args:
        dt_str: A datetime or date string in one of several formats.

    Returns:
        Timezone-aware datetime (UTC), or ``None`` if unparseable.

    Examples::

        >>> validate_datetime_string("2024-01-15T10:30:00")
        datetime.datetime(2024, 1, 15, 10, 30, tzinfo=datetime.timezone.utc)

        >>> validate_datetime_string("garbage") is None
        True
    """
    if not isinstance(dt_str, str):
        return None

    dt_str = dt_str.strip()

    # Try fromisoformat first (handles many ISO variants including +HH:MM offsets)
    try:
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        pass

    for fmt in _DATETIME_FORMATS:
        try:
            dt = datetime.strptime(dt_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            continue

    return None


# ---------------------------------------------------------------------------
# URL
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"^(https?|ftp)://"
    r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}"
    r"(?::\d+)?"
    r"(?:/[^\s]*)?$",
    re.IGNORECASE,
)


def validate_url(url: str) -> bool:
    """Return True if *url* looks like a well-formed HTTP/HTTPS/FTP URL.

    Args:
        url: URL string.

    Returns:
        ``True`` if valid.
    """
    if not isinstance(url, str):
        return False
    return bool(_URL_RE.match(url.strip()))


# ---------------------------------------------------------------------------
# Numeric range
# ---------------------------------------------------------------------------


def validate_positive_integer(value: int, min_val: int = 1, max_val: int = 2**31 - 1) -> bool:
    """Return True if *value* is an integer in [*min_val*, *max_val*].

    Args:
        value:   Value to check.
        min_val: Lower bound (inclusive).
        max_val: Upper bound (inclusive).
    """
    return isinstance(value, int) and not isinstance(value, bool) and min_val <= value <= max_val
