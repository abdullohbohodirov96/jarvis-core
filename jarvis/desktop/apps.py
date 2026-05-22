"""
JARVIS known applications registry.

Provides:
  APP_REGISTRY  — canonical map of logical app name → per-platform commands.
  get_app_command()  — resolve the command to launch an app on a given platform.
  resolve_app_name() — fuzzy-match user input to a known app name.
  list_known_apps()  — return sorted list of all registered app names.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Optional

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

APP_REGISTRY: dict[str, dict[str, str]] = {
    "chrome": {
        "linux": "google-chrome",
        "mac": "Google Chrome",
        "windows": "chrome",
    },
    "firefox": {
        "linux": "firefox",
        "mac": "Firefox",
        "windows": "firefox",
    },
    "vscode": {
        "linux": "code",
        "mac": "Visual Studio Code",
        "windows": "code",
    },
    "terminal": {
        "linux": "gnome-terminal",
        "mac": "Terminal",
        "windows": "cmd",
    },
    "files": {
        "linux": "nautilus",
        "mac": "Finder",
        "windows": "explorer",
    },
    "spotify": {
        "linux": "spotify",
        "mac": "Spotify",
        "windows": "Spotify",
    },
    "telegram": {
        "linux": "telegram-desktop",
        "mac": "Telegram",
        "windows": "Telegram",
    },
    "slack": {
        "linux": "slack",
        "mac": "Slack",
        "windows": "slack",
    },
    "discord": {
        "linux": "discord",
        "mac": "Discord",
        "windows": "Discord",
    },
    "zoom": {
        "linux": "zoom",
        "mac": "zoom.us",
        "windows": "Zoom",
    },
    "gimp": {
        "linux": "gimp",
        "mac": "GIMP",
        "windows": "gimp",
    },
    "vlc": {
        "linux": "vlc",
        "mac": "VLC",
        "windows": "vlc",
    },
    "thunderbird": {
        "linux": "thunderbird",
        "mac": "Thunderbird",
        "windows": "thunderbird",
    },
    "libreoffice": {
        "linux": "libreoffice",
        "mac": "LibreOffice",
        "windows": "soffice",
    },
    "calculator": {
        "linux": "gnome-calculator",
        "mac": "Calculator",
        "windows": "calc",
    },
    "notepad": {
        "linux": "gedit",
        "mac": "TextEdit",
        "windows": "notepad",
    },
    "settings": {
        "linux": "gnome-control-center",
        "mac": "System Preferences",
        "windows": "ms-settings:",
    },
    "steam": {
        "linux": "steam",
        "mac": "Steam",
        "windows": "steam",
    },
    "obs": {
        "linux": "obs",
        "mac": "OBS",
        "windows": "obs64",
    },
    "brave": {
        "linux": "brave-browser",
        "mac": "Brave Browser",
        "windows": "brave",
    },
    "edge": {
        "linux": "microsoft-edge",
        "mac": "Microsoft Edge",
        "windows": "msedge",
    },
    "pycharm": {
        "linux": "pycharm",
        "mac": "PyCharm",
        "windows": "pycharm64",
    },
    "intellij": {
        "linux": "idea",
        "mac": "IntelliJ IDEA",
        "windows": "idea64",
    },
    "postman": {
        "linux": "postman",
        "mac": "Postman",
        "windows": "postman",
    },
}

# Alias table maps common alternate names → canonical registry key.
_ALIASES: dict[str, str] = {
    "google chrome": "chrome",
    "google-chrome": "chrome",
    "chromium": "chrome",
    "vs code": "vscode",
    "visual studio code": "vscode",
    "sublime": "vscode",
    "sublime text": "vscode",
    "bash": "terminal",
    "zsh": "terminal",
    "console": "terminal",
    "file manager": "files",
    "finder": "files",
    "nautilus": "files",
    "explorer": "files",
    "tg": "telegram",
    "tdesktop": "telegram",
    "obs studio": "obs",
    "microsoft edge": "edge",
    "intellij idea": "intellij",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_app_command(app_name: str, platform: str) -> Optional[str]:
    """
    Return the shell command / app name for *app_name* on *platform*.

    Parameters
    ----------
    app_name:
        Canonical key from APP_REGISTRY (e.g. ``"chrome"``).
    platform:
        One of ``"linux"``, ``"mac"``, or ``"windows"``.

    Returns
    -------
    str | None
        The platform-specific command string, or ``None`` if not found.
    """
    entry = APP_REGISTRY.get(app_name.lower())
    if entry is None:
        return None
    return entry.get(platform.lower())


def resolve_app_name(user_input: str) -> Optional[str]:
    """
    Fuzzy-match *user_input* to a known canonical app name.

    Strategy:
    1. Exact match on canonical key.
    2. Exact match on alias table.
    3. Substring match (canonical key contained in input or vice-versa).
    4. SequenceMatcher ratio ≥ 0.6 on canonical key.

    Returns the canonical key (e.g. ``"chrome"``), or ``None``.
    """
    normalised = user_input.lower().strip()

    # 1. Exact canonical match.
    if normalised in APP_REGISTRY:
        return normalised

    # 2. Alias match.
    if normalised in _ALIASES:
        return _ALIASES[normalised]

    # 3. Substring — also check each alias.
    for alias, canonical in _ALIASES.items():
        if alias in normalised or normalised in alias:
            return canonical

    for key in APP_REGISTRY:
        if key in normalised or normalised in key:
            return key

    # 4. Fuzzy ratio.
    best_key: Optional[str] = None
    best_ratio: float = 0.0
    for key in APP_REGISTRY:
        ratio = SequenceMatcher(None, normalised, key).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_key = key

    if best_ratio >= 0.6 and best_key is not None:
        return best_key

    return None


def list_known_apps() -> list[str]:
    """Return a sorted list of all canonical app names in the registry."""
    return sorted(APP_REGISTRY.keys())
