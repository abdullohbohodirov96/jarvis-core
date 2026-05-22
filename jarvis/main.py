#!/usr/bin/env python3
"""
JARVIS - Desktop AI Assistant
==============================

Run modes
---------
    python main.py                 # Full mode: voice + Telegram + API
    python main.py --no-voice      # Disable microphone / wake-word detection
    python main.py --no-telegram   # Disable Telegram integration
    python main.py --api-only      # REST API only (implies --no-voice)
    python main.py --debug         # Verbose logging
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure jarvis/ package root is on sys.path before any jarvis imports.
# This makes ``from config.settings import settings`` work when running
# ``python jarvis/main.py`` from the repo root.
# ---------------------------------------------------------------------------
_JARVIS_ROOT = Path(__file__).parent
if str(_JARVIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_JARVIS_ROOT))

# ---------------------------------------------------------------------------
# Now we can import from jarvis packages.
# ---------------------------------------------------------------------------

from assistant.runner import JarvisRunner  # noqa: E402
from config.settings import settings  # noqa: E402
from core.logger import get_logger, setup_logging  # noqa: E402

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    """Parse command-line arguments and return the namespace."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="JARVIS — Desktop AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py                   Start with all features enabled
  python main.py --no-voice        Text/API mode only (no microphone)
  python main.py --api-only        Run only the REST API server
  python main.py --debug           Enable verbose debug logging
        """,
    )

    parser.add_argument(
        "--no-voice",
        action="store_true",
        default=False,
        help="Disable microphone, wake-word detection and voice playback",
    )
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        default=False,
        help="Disable Telegram integration",
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        default=False,
        help="Run only the FastAPI REST server (implies --no-voice)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable DEBUG log level and extra diagnostics",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------


def _print_banner() -> None:
    """Print the JARVIS ASCII-art banner with startup info."""
    banner = f"""
  ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
  ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
  ██║███████║██████╔╝██║   ██║██║███████╗
  ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
  ██║██║  ██║██║  ██║ ╚████╔╝ ██║███████║
  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝

  Desktop AI Assistant  v{settings.VERSION}
  API  : http://localhost:{settings.API_PORT}
  Docs : http://localhost:{settings.API_PORT}/docs
"""
    print(banner, flush=True)


# ---------------------------------------------------------------------------
# Main coroutine
# ---------------------------------------------------------------------------


async def main(args) -> None:
    """
    Bootstrap the JARVIS runner and block until shutdown.

    Signal handlers (SIGINT / SIGTERM) trigger a graceful shutdown via
    ``runner.shutdown()``.
    """
    runner = JarvisRunner()

    # ── Signal handling ──────────────────────────────────────────────────────
    loop = asyncio.get_running_loop()

    def _request_shutdown():
        log.info("Shutdown signal received.")
        if runner.is_running:
            asyncio.create_task(runner.shutdown(), name="signal-shutdown")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except (NotImplementedError, RuntimeError):
            # Windows does not support add_signal_handler; fall back to the
            # default behaviour (KeyboardInterrupt on SIGINT).
            pass

    # ── Apply CLI overrides to settings ─────────────────────────────────────
    # No-voice: disable wake word + TTS
    if args.no_voice or args.api_only:
        settings.DESKTOP_AUTOMATION_ENABLED = settings.DESKTOP_AUTOMATION_ENABLED
        # We flag voice as disabled by setting wake word to empty string so the
        # runner skips the voice path gracefully.
        object.__setattr__(settings, "WAKE_WORD", "")
        log.info("Voice / wake-word detection disabled.")

    # No-telegram: zero out Telegram credentials so the runner skips them
    if args.no_telegram:
        object.__setattr__(settings, "TELEGRAM_API_ID", 0)
        object.__setattr__(settings, "TELEGRAM_API_HASH", "")
        log.info("Telegram integration disabled.")

    # ── Run ──────────────────────────────────────────────────────────────────
    try:
        await runner.run()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received — shutting down.")
        await runner.shutdown()
    except Exception as exc:
        log.exception("Unhandled exception in main loop: {exc}", exc=exc)
        await runner.shutdown()
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()

    # Apply debug flag early so setup_logging() picks up the right level
    if args.debug:
        import os

        os.environ["DEBUG"] = "true"
        os.environ["LOG_LEVEL"] = "DEBUG"
        # Re-initialise settings so the env vars are picked up
        from config.settings import get_settings

        get_settings.cache_clear()
        # Reload logging after settings change
        setup_logging()
        log.debug("Debug mode enabled.")

    _print_banner()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        # Swallow the final KeyboardInterrupt so Python doesn't print a traceback.
        print("\nGoodbye, sir.", flush=True)
