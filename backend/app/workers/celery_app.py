"""
Celery application factory for JARVIS background workers.

Creates and configures the Celery app with:
- Redis broker + result backend (from settings)
- JSON serialisation throughout
- Beat schedule for all periodic tasks
- Sensible worker defaults for reliability

get_celery_app() returns the process-level singleton.
"""

from __future__ import annotations

import logging

from celery import Celery
from celery.schedules import crontab

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_celery_app: Celery | None = None


def _create_celery_app() -> Celery:
    """Build and return a fully configured Celery application instance."""
    from backend.app.core.config import get_settings

    settings = get_settings()

    app = Celery(
        "jarvis",
        broker=settings.CELERY_BROKER_URL,
        backend=settings.CELERY_RESULT_BACKEND,
        include=[
            "backend.app.workers.tasks.reminder_tasks",
            "backend.app.workers.tasks.telegram_tasks",
            "backend.app.workers.tasks.ai_tasks",
            "backend.app.workers.tasks.summary_tasks",
        ],
    )

    # ------------------------------------------------------------------
    # Core configuration
    # ------------------------------------------------------------------
    app.conf.update(
        # Serialisation
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        # Timezone
        timezone="UTC",
        enable_utc=True,
        # Reliability
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        # Result TTL: keep results for 24 h
        result_expires=86400,
        # Retry configuration
        task_max_retries=3,
        task_default_retry_delay=60,  # seconds
        # Concurrency & routing
        worker_concurrency=4,
        # Broker connection retry on startup
        broker_connection_retry_on_startup=True,
        # Beat schedule for periodic tasks
        beat_schedule={
            # ── Reminder tasks ─────────────────────────────────────────
            "check-reminders-every-60s": {
                "task": "backend.app.workers.tasks.reminder_tasks.check_and_send_reminders",
                "schedule": 60.0,
                "options": {"queue": "reminders"},
            },
            "check-overdue-tasks-every-5min": {
                "task": "backend.app.workers.tasks.reminder_tasks.check_overdue_tasks",
                "schedule": 300.0,
                "options": {"queue": "reminders"},
            },
            "send-daily-task-summary-9am": {
                "task": "backend.app.workers.tasks.reminder_tasks.send_daily_task_summary",
                "schedule": crontab(hour=9, minute=0),
                "options": {"queue": "summaries"},
            },
            # ── Telegram tasks ─────────────────────────────────────────
            "process-scheduled-telegram-every-30s": {
                "task": "backend.app.workers.tasks.telegram_tasks.process_scheduled_messages",
                "schedule": 30.0,
                "options": {"queue": "telegram"},
            },
            # ── AI / memory tasks ──────────────────────────────────────
            "memory-consolidation-midnight": {
                "task": "backend.app.workers.tasks.ai_tasks.consolidate_user_memories",
                "schedule": crontab(hour=0, minute=0),
                "kwargs": {"user_id": "all"},
                "options": {"queue": "ai"},
            },
            "cleanup-expired-memories-midnight": {
                "task": "backend.app.workers.tasks.ai_tasks.cleanup_expired_memories",
                "schedule": crontab(hour=0, minute=30),
                "options": {"queue": "ai"},
            },
            # ── Summary tasks ──────────────────────────────────────────
            "send-morning-briefing-8am": {
                "task": "backend.app.workers.tasks.summary_tasks.send_morning_briefing",
                "schedule": crontab(hour=8, minute=0),
                "kwargs": {"user_id": "all"},
                "options": {"queue": "summaries"},
            },
            "generate-weekly-report-sunday-10am": {
                "task": "backend.app.workers.tasks.summary_tasks.generate_weekly_report",
                "schedule": crontab(hour=10, minute=0, day_of_week=0),
                "kwargs": {"user_id": "all"},
                "options": {"queue": "summaries"},
            },
            # ── Cleanup ────────────────────────────────────────────────
            "cleanup-old-data-weekly": {
                "task": "backend.app.workers.tasks.ai_tasks.cleanup_expired_memories",
                "schedule": crontab(hour=3, minute=0, day_of_week=0),
                "options": {"queue": "ai"},
            },
        },
        # Queue definitions
        task_queues={
            "default": {"exchange": "default", "routing_key": "default"},
            "reminders": {"exchange": "reminders", "routing_key": "reminders"},
            "telegram": {"exchange": "telegram", "routing_key": "telegram"},
            "ai": {"exchange": "ai", "routing_key": "ai"},
            "summaries": {"exchange": "summaries", "routing_key": "summaries"},
        },
        task_default_queue="default",
        task_routes={
            "backend.app.workers.tasks.reminder_tasks.*": {"queue": "reminders"},
            "backend.app.workers.tasks.telegram_tasks.*": {"queue": "telegram"},
            "backend.app.workers.tasks.ai_tasks.*": {"queue": "ai"},
            "backend.app.workers.tasks.summary_tasks.*": {"queue": "summaries"},
        },
    )

    logger.info(
        "Celery app created: broker=%s backend=%s",
        settings.CELERY_BROKER_URL,
        settings.CELERY_RESULT_BACKEND,
    )
    return app


def get_celery_app() -> Celery:
    """
    Return the process-level Celery application singleton.

    Thread-safe at module level; the GIL guards the initial assignment.
    """
    global _celery_app
    if _celery_app is None:
        _celery_app = _create_celery_app()
    return _celery_app


# ---------------------------------------------------------------------------
# Module-level ``celery`` symbol required by the ``celery`` CLI
# ---------------------------------------------------------------------------

# When workers are launched via `celery -A backend.app.workers.celery_app worker`
# the CLI looks for a module-level Celery instance.  We expose it lazily so
# the settings are loaded on first access.

celery = get_celery_app()
