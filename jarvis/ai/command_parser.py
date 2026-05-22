"""
Command parser for JARVIS AI responses.

Parses the structured JSON that the AI model returns and validates action
types against the known action registry.  Falls back gracefully when the
model returns plain text instead of JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Known actions registry
# ---------------------------------------------------------------------------

# Maps each action type to a description and the required/optional param schema.
# "required" lists keys that MUST be present; "optional" lists keys that MAY appear.
KNOWN_ACTIONS: dict[str, dict[str, Any]] = {
    "SEND_TELEGRAM": {
        "description": "Send a Telegram message to a contact or group.",
        "required": ["chat_id_or_name", "message"],
        "optional": [],
    },
    "OPEN_APP": {
        "description": "Open an application by name (e.g. Chrome, VS Code, Terminal).",
        "required": ["app_name"],
        "optional": [],
    },
    "TYPE_TEXT": {
        "description": "Type text at the current cursor position.",
        "required": ["text"],
        "optional": [],
    },
    "OPEN_URL": {
        "description": "Open a URL in the default web browser.",
        "required": ["url"],
        "optional": [],
    },
    "CREATE_TASK": {
        "description": "Create a new task or reminder.",
        "required": ["title"],
        "optional": ["description", "due_date", "priority"],
    },
    "COMPLETE_TASK": {
        "description": "Mark a task as complete.",
        "required": [],
        "optional": ["task_id", "title"],
    },
    "LIST_TASKS": {
        "description": "List all pending tasks.",
        "required": [],
        "optional": [],
    },
    "TAKE_SCREENSHOT": {
        "description": "Capture a screenshot of the current screen.",
        "required": [],
        "optional": ["filename"],
    },
    "SEARCH_WEB": {
        "description": "Open a web search for the given query.",
        "required": ["query"],
        "optional": [],
    },
    "SPEAK": {
        "description": "Respond with voice only — no desktop action.",
        "required": [],
        "optional": [],
    },
}


# ---------------------------------------------------------------------------
# ParsedResponse dataclass
# ---------------------------------------------------------------------------


@dataclass
class ParsedResponse:
    """Structured representation of a JARVIS AI response."""

    speech: str
    """What JARVIS says out loud to the user."""

    action_type: str | None
    """The validated action type string, or None if no action is requested."""

    action_params: dict[str, Any]
    """Parameters for the action (empty dict when action_type is None)."""

    follow_up: str | None
    """Optional follow-up statement or question from JARVIS."""

    raw: str
    """The raw response text as returned by the model."""

    is_fallback: bool = field(default=False)
    """True when the response could not be parsed as JSON and a fallback was used."""

    validation_errors: list[str] = field(default_factory=list)
    """List of non-fatal validation warnings (e.g. missing optional params)."""


# ---------------------------------------------------------------------------
# CommandParser
# ---------------------------------------------------------------------------


class CommandParser:
    """
    Parse AI JSON responses into ``ParsedResponse`` objects.

    Handles:
    - Valid JSON with expected fields.
    - JSON missing optional fields (graceful defaults).
    - Unknown action types (set action_type to None with a warning).
    - Plain text responses (treated as speech-only fallback).
    """

    # ------------------------------------------------------------------
    # parse
    # ------------------------------------------------------------------

    def parse(self, response_text: str) -> ParsedResponse:
        """
        Parse a raw model response string.

        Strategy:
        1. Try ``json.loads`` on the full response.
        2. If that fails, scan for the first ``{`` … last ``}`` substring
           and try again (model sometimes wraps JSON in prose).
        3. If JSON is found, extract ``speech``, ``action``, and
           ``follow_up`` fields.
        4. Validate action type against ``KNOWN_ACTIONS``.
        5. If all JSON attempts fail, treat the entire text as speech.

        Args:
            response_text: Raw string returned by the model.

        Returns:
            ``ParsedResponse`` instance.
        """
        raw = response_text.strip()

        # ── Attempt JSON parse ────────────────────────────────────────────────
        parsed_json: dict[str, Any] | None = None

        # Try the full string first
        parsed_json = self._try_parse_json(raw)

        # If that fails, try to extract a JSON object substring
        if parsed_json is None:
            parsed_json = self._extract_json_substring(raw)

        if parsed_json is not None:
            return self._build_from_json(parsed_json, raw)

        # ── Fallback: treat as plain speech ──────────────────────────────────
        logger.warning(
            "CommandParser: could not parse JSON from response (len={}). "
            "Using plain-text fallback.",
            len(raw),
        )
        speech = raw if raw else "Understood."
        return ParsedResponse(
            speech=speech,
            action_type=None,
            action_params={},
            follow_up=None,
            raw=raw,
            is_fallback=True,
        )

    # ------------------------------------------------------------------
    # is_valid_action
    # ------------------------------------------------------------------

    def is_valid_action(self, action_type: str) -> bool:
        """
        Return True if ``action_type`` is a recognised JARVIS action.

        Args:
            action_type: The action type string to check.

        Returns:
            Boolean.
        """
        return action_type in KNOWN_ACTIONS

    # ------------------------------------------------------------------
    # get_action_description
    # ------------------------------------------------------------------

    def get_action_description(self, action_type: str) -> str:
        """
        Return the human-readable description for an action type.

        Args:
            action_type: A known action type string.

        Returns:
            Description string, or an empty string if ``action_type`` is
            unknown.
        """
        entry = KNOWN_ACTIONS.get(action_type)
        if entry is None:
            return ""
        return entry.get("description", "")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_parse_json(self, text: str) -> dict[str, Any] | None:
        """Attempt a direct json.loads parse. Returns the dict or None."""
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _extract_json_substring(self, text: str) -> dict[str, Any] | None:
        """
        Look for the outermost ``{...}`` block in ``text`` and try to parse it.

        Handles cases where the model wraps JSON in prose or code fences.
        """
        # Strip markdown code fences if present
        stripped = text
        for fence in ("```json", "```JSON", "```"):
            if fence in stripped:
                stripped = stripped.split(fence, 1)[-1]
                if "```" in stripped:
                    stripped = stripped.rsplit("```", 1)[0]
                stripped = stripped.strip()
                break

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        candidate = stripped[start : end + 1]
        return self._try_parse_json(candidate)

    def _build_from_json(
        self, data: dict[str, Any], raw: str
    ) -> ParsedResponse:
        """
        Build a ``ParsedResponse`` from a parsed JSON dict.

        Handles missing or malformed fields defensively.
        """
        errors: list[str] = []

        # ── Extract speech ────────────────────────────────────────────────────
        speech_raw = data.get("speech", "")
        if not isinstance(speech_raw, str) or not speech_raw.strip():
            errors.append("'speech' field missing or empty; using fallback.")
            speech = "Understood."
        else:
            speech = speech_raw.strip()

        # ── Extract follow_up ─────────────────────────────────────────────────
        follow_up_raw = data.get("follow_up")
        follow_up: str | None = None
        if isinstance(follow_up_raw, str) and follow_up_raw.strip():
            follow_up = follow_up_raw.strip()

        # ── Extract action ────────────────────────────────────────────────────
        action_data = data.get("action")
        action_type: str | None = None
        action_params: dict[str, Any] = {}

        if action_data is not None and isinstance(action_data, dict):
            raw_type = action_data.get("type", "")
            if isinstance(raw_type, str) and raw_type.strip():
                candidate_type = raw_type.strip().upper()
                if self.is_valid_action(candidate_type):
                    action_type = candidate_type
                    raw_params = action_data.get("params", {})
                    action_params = raw_params if isinstance(raw_params, dict) else {}
                    # Validate required params
                    required = KNOWN_ACTIONS[candidate_type].get("required", [])
                    missing = [k for k in required if k not in action_params]
                    if missing:
                        errors.append(
                            f"Action '{candidate_type}' is missing required "
                            f"params: {missing}."
                        )
                else:
                    errors.append(
                        f"Unknown action type '{raw_type}'; ignoring action."
                    )
            # else: type field empty/missing — no action
        # action_data == null or missing → no action

        if errors:
            for err in errors:
                logger.warning("CommandParser: {}.", err)

        return ParsedResponse(
            speech=speech,
            action_type=action_type,
            action_params=action_params,
            follow_up=follow_up,
            raw=raw,
            is_fallback=False,
            validation_errors=errors,
        )
