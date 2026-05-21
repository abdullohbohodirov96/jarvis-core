"""
General-purpose utility functions used across JARVIS.
"""

from __future__ import annotations

import asyncio
import functools
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


def generate_id() -> str:
    """Return a new UUID4 string (no dashes are removed — standard format).

    Example::

        >>> id_ = generate_id()
        >>> len(id_)
        36
    """
    return str(uuid.uuid4())


def generate_short_id(length: int = 8) -> str:
    """Return a compact, URL-safe random ID of *length* characters.

    Uses hex characters (0-9, a-f) for simplicity.

    Args:
        length: Desired length of the output string (default 8).
    """
    return uuid.uuid4().hex[:length]


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def now_utc() -> datetime:
    """Return the current UTC datetime as a timezone-aware object."""
    return datetime.now(tz=timezone.utc)


def format_datetime(dt: datetime, fmt: str = "%Y-%m-%dT%H:%M:%SZ") -> str:
    """Format *dt* as a string.

    Args:
        dt:  Datetime object (naive datetimes are assumed UTC).
        fmt: strftime format string.  Defaults to ISO-8601 UTC.

    Returns:
        Formatted datetime string.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime(fmt)


def parse_datetime(dt_str: str) -> Optional[datetime]:
    """Attempt to parse an ISO-8601 datetime string.

    Returns ``None`` if parsing fails instead of raising.

    Args:
        dt_str: ISO-8601 formatted datetime string.
    """
    try:
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def truncate_text(text: str, max_len: int, suffix: str = "…") -> str:
    """Truncate *text* to at most *max_len* characters.

    A *suffix* (default: ellipsis) is appended when truncation occurs.

    Args:
        text:    Input string.
        max_len: Maximum allowed length (including the suffix).
        suffix:  String appended on truncation.

    Returns:
        Original or truncated string.

    Example::

        >>> truncate_text("Hello, World!", 8)
        'Hello, …'
    """
    if len(text) <= max_len:
        return text
    cut = max_len - len(suffix)
    if cut < 0:
        return suffix[:max_len]
    return text[:cut] + suffix


def sanitize_html(text: str) -> str:
    """Strip HTML tags from *text* using a simple regex.

    For production use consider the ``bleach`` library for proper sanitisation.
    """
    return re.sub(r"<[^>]+>", "", text)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_JSON_INLINE_RE = re.compile(r"(\{[\s\S]*\}|\[[\s\S]*\])")


def extract_json_from_text(text: str) -> Any:
    """Extract and parse a JSON object or array from *text*.

    Tries (in order):
      1. Direct JSON parse of the whole string.
      2. Fenced code-block (```json ... ```).
      3. First ``{...}`` or ``[...]`` block in the string.

    Args:
        text: Input string that may contain embedded JSON.

    Returns:
        Parsed Python object (dict, list, etc.) or ``None`` if nothing found.
    """
    # 1. Direct
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Fenced code block
    block_match = _JSON_BLOCK_RE.search(text)
    if block_match:
        try:
            return json.loads(block_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Inline JSON
    inline_match = _JSON_INLINE_RE.search(text)
    if inline_match:
        try:
            return json.loads(inline_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Parse *text* as JSON, returning ``None`` on failure.

    Unlike ``json.loads``, this never raises an exception.

    Args:
        text: JSON-encoded string.

    Returns:
        Parsed dict, or ``None``.
    """
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------


def chunk_list(lst: List[Any], size: int) -> List[List[Any]]:
    """Split *lst* into sublists of at most *size* elements.

    Args:
        lst:  Input list.
        size: Maximum sublist length (must be >= 1).

    Returns:
        List of sublists.

    Example::

        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    if size < 1:
        raise ValueError("chunk size must be at least 1")
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def flatten(nested: List[List[Any]]) -> List[Any]:
    """Flatten a single level of nesting."""
    return [item for sublist in nested for item in sublist]


# ---------------------------------------------------------------------------
# Async retry decorator
# ---------------------------------------------------------------------------


def retry_async(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """Decorator that retries an async function on failure.

    Args:
        retries:    Maximum number of retry attempts (not counting the first call).
        delay:      Initial wait between retries in seconds.
        backoff:    Multiplier applied to *delay* after each failure.
        exceptions: Tuple of exception types to catch.

    Returns:
        Decorated coroutine function.

    Example::

        @retry_async(retries=3, delay=0.5)
        async def call_external_api():
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exc: Optional[Exception] = None
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:  # type: ignore[misc]
                    last_exc = exc
                    if attempt < retries:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Async generator helpers
# ---------------------------------------------------------------------------


async def aenumerate(
    aiter: AsyncGenerator[Any, None], start: int = 0
) -> AsyncGenerator[tuple, None]:
    """Async version of ``enumerate``."""
    i = start
    async for item in aiter:
        yield i, item
        i += 1
