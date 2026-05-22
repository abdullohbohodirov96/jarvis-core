"""
JARVIS desktop automation controller.

Uses pyautogui for mouse/keyboard control and subprocess for launching apps.
All public methods are async (they offload blocking calls to a thread-pool
executor so the event loop is never blocked).
"""

from __future__ import annotations

import asyncio
import os
import platform
import subprocess
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import settings
from core.logger import get_logger
from desktop.apps import APP_REGISTRY, get_app_command, resolve_app_name

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLATFORM_MAP = {
    "Linux": "linux",
    "Darwin": "mac",
    "Windows": "windows",
}


def _detect_platform() -> str:
    return _PLATFORM_MAP.get(platform.system(), "linux")


def _run_in_executor(func, *args):
    """Schedule a blocking *func* on the default thread-pool executor."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, func, *args)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ActionResult:
    success: bool
    action_type: str
    result: str  # human-readable description
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class DesktopController:
    """
    Cross-platform desktop automation.

    All methods are coroutines; blocking OS calls are dispatched to a
    thread-pool executor.
    """

    def __init__(self) -> None:
        self.is_enabled: bool = settings.DESKTOP_AUTOMATION_ENABLED
        self._platform: str = _detect_platform()
        self._screenshot_dir = Path(settings.SCREENSHOT_DIR)
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "DesktopController initialised (platform={p}, enabled={e})",
            p=self._platform,
            e=self.is_enabled,
        )

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------

    def _check_enabled(self) -> bool:
        if not self.is_enabled:
            log.warning("Desktop automation is disabled (DESKTOP_AUTOMATION_ENABLED=False).")
        return self.is_enabled

    # ------------------------------------------------------------------
    # App launching
    # ------------------------------------------------------------------

    async def open_app(self, app_name: str) -> bool:
        """
        Open an application by name.

        Resolves the name through the APP_REGISTRY and launches it using
        the appropriate platform mechanism.
        """
        if not self._check_enabled():
            return False

        canonical = resolve_app_name(app_name)
        if canonical is None:
            # Try the raw name as a command.
            canonical_cmd = app_name
            log.warning(
                "App '{name}' not in registry — trying raw command.", name=app_name
            )
        else:
            canonical_cmd = get_app_command(canonical, self._platform) or canonical

        try:
            await _run_in_executor(self._launch_app_command, canonical_cmd)
            log.info("Opened app '{name}' → '{cmd}'", name=app_name, cmd=canonical_cmd)
            return True
        except Exception as exc:
            log.error("open_app error for '{name}': {exc}", name=app_name, exc=exc)
            return False

    def _launch_app_command(self, cmd: str) -> None:
        """Blocking helper — run by executor."""
        sys_platform = self._platform
        if sys_platform == "linux":
            # Try xdg-open first; fall back to direct subprocess.
            try:
                subprocess.Popen(
                    ["xdg-open", cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                subprocess.Popen(
                    cmd.split(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        elif sys_platform == "mac":
            subprocess.Popen(
                ["open", "-a", cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif sys_platform == "windows":
            subprocess.Popen(
                ["start", cmd],
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            raise RuntimeError(f"Unsupported platform: {sys_platform}")

    # ------------------------------------------------------------------
    # URL / browser
    # ------------------------------------------------------------------

    async def open_url(self, url: str, browser: str = "default") -> bool:
        """Open *url* in the system default browser (or a named browser)."""
        if not self._check_enabled():
            return False

        def _open() -> bool:
            if browser == "default":
                return webbrowser.open(url)
            controller = webbrowser.get(browser)
            return controller.open(url)

        try:
            result = await _run_in_executor(_open)
            log.info("Opened URL: {url}", url=url)
            return bool(result)
        except Exception as exc:
            log.error("open_url error: {exc}", exc=exc)
            return False

    # ------------------------------------------------------------------
    # Keyboard / typing
    # ------------------------------------------------------------------

    async def type_text(self, text: str, interval: float = 0.05) -> bool:
        """Type *text* at the current cursor position."""
        if not self._check_enabled():
            return False

        def _type() -> None:
            import pyautogui  # local import: only needed at call time

            # pyautogui.typewrite does not handle unicode; use write for safety.
            pyautogui.write(text, interval=interval)

        try:
            await _run_in_executor(_type)
            return True
        except Exception as exc:
            log.error("type_text error: {exc}", exc=exc)
            return False

    async def press_key(self, key: str) -> bool:
        """Press a single keyboard key (e.g. ``'enter'``, ``'esc'``)."""
        if not self._check_enabled():
            return False

        def _press() -> None:
            import pyautogui
            pyautogui.press(key)

        try:
            await _run_in_executor(_press)
            return True
        except Exception as exc:
            log.error("press_key error for '{key}': {exc}", key=key, exc=exc)
            return False

    async def hotkey(self, *keys: str) -> bool:
        """Simulate a keyboard shortcut (e.g. ``hotkey('ctrl', 'c')``)."""
        if not self._check_enabled():
            return False

        def _hotkey() -> None:
            import pyautogui
            pyautogui.hotkey(*keys)

        try:
            await _run_in_executor(_hotkey)
            return True
        except Exception as exc:
            log.error("hotkey error {keys}: {exc}", keys=keys, exc=exc)
            return False

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    async def click(self, x: int, y: int, button: str = "left") -> bool:
        """Click at screen coordinates (*x*, *y*)."""
        if not self._check_enabled():
            return False

        def _click() -> None:
            import pyautogui
            pyautogui.click(x, y, button=button)

        try:
            await _run_in_executor(_click)
            return True
        except Exception as exc:
            log.error("click error at ({x},{y}): {exc}", x=x, y=y, exc=exc)
            return False

    # ------------------------------------------------------------------
    # Screen
    # ------------------------------------------------------------------

    async def get_screen_size(self) -> tuple[int, int]:
        """Return the current screen resolution as ``(width, height)``."""

        def _size() -> tuple[int, int]:
            import pyautogui
            return pyautogui.size()

        try:
            return await _run_in_executor(_size)
        except Exception as exc:
            log.error("get_screen_size error: {exc}", exc=exc)
            return (1920, 1080)  # sensible fallback

    async def take_screenshot(self, filename: Optional[str] = None) -> str:
        """
        Capture the entire screen and save to ``settings.SCREENSHOT_DIR``.

        Returns the absolute path of the saved file.
        """
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"screenshot_{ts}.png"

        out_path = self._screenshot_dir / filename

        def _capture() -> None:
            import pyautogui
            img = pyautogui.screenshot()
            img.save(str(out_path))

        try:
            await _run_in_executor(_capture)
            log.info("Screenshot saved: {path}", path=out_path)
            return str(out_path)
        except Exception as exc:
            log.error("take_screenshot error: {exc}", exc=exc)
            return ""

    # ------------------------------------------------------------------
    # Web search
    # ------------------------------------------------------------------

    async def search_web(self, query: str) -> bool:
        """Open the default browser with a Google search for *query*."""
        url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        return await self.open_url(url)

    # ------------------------------------------------------------------
    # Generic action dispatcher
    # ------------------------------------------------------------------

    #: Map action_type string → (method_name, required_param_keys)
    _ACTION_DISPATCH: dict[str, tuple[str, list[str]]] = {
        "OPEN_APP": ("open_app", ["app_name"]),
        "OPEN_URL": ("open_url", ["url"]),
        "TYPE_TEXT": ("type_text", ["text"]),
        "TAKE_SCREENSHOT": ("take_screenshot", []),
        "SEARCH_WEB": ("search_web", ["query"]),
        "PRESS_KEY": ("press_key", ["key"]),
        "HOTKEY": ("hotkey", []),  # keys passed as *args via special handling
        "CLICK": ("click", ["x", "y"]),
    }

    async def execute_action(self, action_type: str, params: dict) -> ActionResult:
        """
        Dispatch to the right method based on *action_type*.

        Supported action_type values:
          OPEN_APP, OPEN_URL, TYPE_TEXT, TAKE_SCREENSHOT, SEARCH_WEB,
          PRESS_KEY, HOTKEY, CLICK

        *params* should carry the arguments required by the target method.
        Extra keys are silently ignored.
        """
        utype = action_type.upper()
        dispatch_info = self._ACTION_DISPATCH.get(utype)

        if dispatch_info is None:
            err = f"Unknown action_type: '{action_type}'"
            log.warning(err)
            return ActionResult(
                success=False,
                action_type=action_type,
                result="",
                error=err,
            )

        method_name, required_keys = dispatch_info
        method = getattr(self, method_name)

        try:
            # Special handling for actions that use *args.
            if utype == "HOTKEY":
                keys = params.get("keys", [])
                if isinstance(keys, str):
                    keys = [k.strip() for k in keys.split(",")]
                result_val = await method(*keys)
            elif utype == "TAKE_SCREENSHOT":
                result_val = await method(params.get("filename"))
            else:
                kwargs = {k: params[k] for k in required_keys if k in params}
                # Also pass optional kwargs that the method accepts.
                if utype == "TYPE_TEXT" and "interval" in params:
                    kwargs["interval"] = float(params["interval"])
                if utype == "OPEN_URL" and "browser" in params:
                    kwargs["browser"] = params["browser"]
                if utype == "CLICK" and "button" in params:
                    kwargs["button"] = params["button"]
                result_val = await method(**kwargs)

            if isinstance(result_val, bool):
                success = result_val
                human_result = "Success" if success else "Failed"
            elif isinstance(result_val, str):
                success = bool(result_val)
                human_result = result_val or "No output"
            elif isinstance(result_val, tuple):
                success = True
                human_result = f"Result: {result_val}"
            else:
                success = True
                human_result = str(result_val)

            return ActionResult(
                success=success,
                action_type=action_type,
                result=human_result,
                error=None,
            )

        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            log.error(
                "execute_action failed for '{at}': {exc}", at=action_type, exc=exc
            )
            return ActionResult(
                success=False,
                action_type=action_type,
                result="",
                error=err_msg,
            )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_controller_instance: Optional[DesktopController] = None


def get_controller() -> DesktopController:
    """Return the process-level DesktopController singleton."""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = DesktopController()
    return _controller_instance
