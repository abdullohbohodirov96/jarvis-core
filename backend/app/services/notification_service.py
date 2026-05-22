"""
Notification service for JARVIS.

NotificationService dispatches user-facing notifications via Telegram
(with hooks for future push notification channels).  It handles:

- Task reminders (with smart formatting)
- Daily briefing summaries
- Ad-hoc alerts
- Due-task push notifications
- Voice notification stubs (ElevenLabs integration point)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

from app.core.exceptions import TelegramException

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.telegram.client import TelegramClientManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Message templates
# ---------------------------------------------------------------------------

_REMINDER_TEMPLATE = """\
{priority_emoji} **Task Reminder**

📝 {title}
{description_line}\
{due_line}\
📊 Priority: {priority}
"""

_DAILY_SUMMARY_TEMPLATE = """\
🌅 **Daily JARVIS Summary** — {date}

📊 **Task Stats**
• Total: {total}
• To Do: {todo}
• In Progress: {in_progress}
• Done: {done}

{overdue_section}\
{upcoming_section}\
"""

_ALERT_TEMPLATE = {
    "normal": "ℹ️ {message}",
    "warning": "⚠️ **Warning** | {message}",
    "high": "🚨 **Alert** | {message}",
    "critical": "🆘 **CRITICAL** | {message}",
}

_PRIORITY_EMOJI = {
    "low": "🟢",
    "medium": "🟡",
    "high": "🟠",
    "urgent": "🔴",
}


class NotificationService:
    """
    Sends notifications through available channels (Telegram + future push).

    Args:
        telegram_client: Connected TelegramClientManager instance.
        db_session:      Async SQLAlchemy session (for loading user data).
    """

    def __init__(
        self,
        telegram_client: TelegramClientManager,
        db_session: AsyncSession,
    ) -> None:
        self._client = telegram_client
        self._db = db_session

    # ------------------------------------------------------------------
    # Task reminders
    # ------------------------------------------------------------------

    async def send_reminder(
        self,
        task: Any,
        user_telegram_id: int,
    ) -> bool:
        """
        Send a task reminder to the user's Telegram account.

        Args:
            task:             Task-like object (dataclass or ORM model with
                              title, description, due_date, priority fields).
            user_telegram_id: Telegram user ID (numeric) to send to.

        Returns:
            True if sent successfully.
        """
        title = getattr(task, "title", "Untitled Task")
        description = getattr(task, "description", "") or ""
        due_date: datetime | None = getattr(task, "due_date", None)
        priority_val: str = getattr(
            getattr(task, "priority", "medium"), "value",
            getattr(task, "priority", "medium")
        )

        priority_emoji = _PRIORITY_EMOJI.get(priority_val, "⚪")

        description_line = f"📄 {description[:200]}\n" if description.strip() else ""
        if due_date:
            formatted_due = due_date.strftime("%Y-%m-%d %H:%M UTC")
            due_line = f"⏰ Due: {formatted_due}\n"
        else:
            due_line = ""

        message = _REMINDER_TEMPLATE.format(
            priority_emoji=priority_emoji,
            title=title,
            description_line=description_line,
            due_line=due_line,
            priority=priority_val.capitalize(),
        ).strip()

        try:
            success = await self._client.send_message(
                chat_id_or_username=user_telegram_id,
                text=message,
                parse_mode="md",
            )
            if success:
                logger.info(
                    "send_reminder: task=%s user_tg=%d",
                    getattr(task, "id", "?"), user_telegram_id,
                )
            return success
        except TelegramException as exc:
            logger.error(
                "send_reminder failed for user_tg=%d: %s", user_telegram_id, exc
            )
            return False

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    async def send_daily_summary(self, user_id: UUID) -> bool:
        """
        Generate and send a daily task summary to the user.

        Loads user Telegram ID from DB and builds the summary from task data.

        Args:
            user_id: UUID of the JARVIS user.

        Returns:
            True if sent successfully.
        """
        user_tg_id = await self._get_user_telegram_id(user_id)
        if not user_tg_id:
            logger.warning(
                "send_daily_summary: no Telegram ID for user=%s", user_id
            )
            return False

        from app.services.task_service import TaskService

        task_service = TaskService(db_session=self._db, user_id=user_id)
        summary = await task_service.get_daily_summary()

        counts = summary.get("counts", {})
        overdue = summary.get("overdue", [])
        upcoming = summary.get("upcoming_24h", [])

        # Build overdue section
        overdue_section = ""
        if overdue:
            overdue_lines = [
                f"  • {t['title']} (due {t['due_date'][:10] if t.get('due_date') else 'N/A'})"
                for t in overdue[:5]
            ]
            overdue_section = (
                f"⚠️ **Overdue ({len(overdue)})**\n"
                + "\n".join(overdue_lines)
                + "\n\n"
            )

        # Build upcoming section
        upcoming_section = ""
        if upcoming:
            upcoming_lines = [
                f"  • {t['title']} (due {t['due_date'][11:16] if t.get('due_date') else 'TBD'} UTC)"
                for t in upcoming[:5]
            ]
            upcoming_section = (
                f"📅 **Due in 24h ({len(upcoming)})**\n"
                + "\n".join(upcoming_lines)
            )

        today = datetime.now(timezone.utc).strftime("%A, %B %d %Y")
        message = _DAILY_SUMMARY_TEMPLATE.format(
            date=today,
            total=counts.get("total", 0),
            todo=counts.get("todo", 0),
            in_progress=counts.get("in_progress", 0),
            done=counts.get("done", 0),
            overdue_section=overdue_section,
            upcoming_section=upcoming_section,
        ).strip()

        try:
            success = await self._client.send_message(
                chat_id_or_username=user_tg_id,
                text=message,
                parse_mode="md",
            )
            logger.info("send_daily_summary: user=%s tg=%s", user_id, user_tg_id)
            return success
        except TelegramException as exc:
            logger.error("send_daily_summary failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    async def send_alert(
        self,
        user_id: UUID,
        message: str,
        priority: str = "normal",
    ) -> bool:
        """
        Send a one-off alert message to the user.

        Args:
            user_id:  JARVIS user UUID.
            message:  Alert text (plain or Markdown).
            priority: "normal" | "warning" | "high" | "critical".

        Returns:
            True if sent successfully.
        """
        user_tg_id = await self._get_user_telegram_id(user_id)
        if not user_tg_id:
            logger.warning("send_alert: no Telegram ID for user=%s", user_id)
            return False

        template = _ALERT_TEMPLATE.get(priority, _ALERT_TEMPLATE["normal"])
        formatted = template.format(message=message)

        try:
            success = await self._client.send_message(
                chat_id_or_username=user_tg_id,
                text=formatted,
                parse_mode="md",
            )
            logger.info(
                "send_alert: user=%s priority=%s", user_id, priority
            )
            return success
        except TelegramException as exc:
            logger.error("send_alert failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Task due notification
    # ------------------------------------------------------------------

    async def notify_task_due(self, task: Any) -> bool:
        """
        Send a "task is now due" notification.

        Looks up the user's Telegram ID from the task's user_id field.

        Args:
            task: Task-like object with ``user_id`` and task fields.

        Returns:
            True if sent successfully.
        """
        user_id: UUID | None = getattr(task, "user_id", None)
        if not user_id:
            logger.warning("notify_task_due: task has no user_id")
            return False

        user_tg_id = await self._get_user_telegram_id(user_id)
        if not user_tg_id:
            logger.warning(
                "notify_task_due: no Telegram ID for user=%s", user_id
            )
            return False

        title = getattr(task, "title", "Unnamed task")
        priority_raw = getattr(task, "priority", "medium")
        priority_val = getattr(priority_raw, "value", str(priority_raw))
        emoji = _PRIORITY_EMOJI.get(priority_val, "⚪")

        message = (
            f"{emoji} **Task Due Now**\n\n"
            f"📝 {title}\n"
            f"🕐 Due: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )

        try:
            success = await self._client.send_message(
                chat_id_or_username=user_tg_id,
                text=message,
                parse_mode="md",
            )
            logger.info(
                "notify_task_due: task=%s user_tg=%s",
                getattr(task, "id", "?"), user_tg_id,
            )
            return success
        except TelegramException as exc:
            logger.error("notify_task_due failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Voice notification (ElevenLabs stub)
    # ------------------------------------------------------------------

    async def send_voice_notification(
        self,
        user_id: UUID,
        text: str,
    ) -> bool:
        """
        Generate a voice notification (TTS) and send it as a Telegram voice message.

        Currently a best-effort implementation: falls back to text message if
        ElevenLabs is unavailable.

        Args:
            user_id: JARVIS user UUID.
            text:    Text to synthesise into speech.

        Returns:
            True if a notification (voice or text) was sent.
        """
        from app.core.config import get_settings

        settings = get_settings()
        user_tg_id = await self._get_user_telegram_id(user_id)
        if not user_tg_id:
            return False

        audio_bytes: bytes | None = None

        if settings.ELEVENLABS_API_KEY:
            try:
                audio_bytes = await self._synthesise_voice(text, settings)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "send_voice_notification: TTS failed, falling back to text: %s", exc
                )

        if audio_bytes:
            try:
                import io

                buf = io.BytesIO(audio_bytes)
                buf.name = "notification.ogg"
                await self._client.raw.send_file(
                    entity=user_tg_id,
                    file=buf,
                    voice_note=True,
                )
                logger.info(
                    "send_voice_notification: voice sent to user=%s tg=%s",
                    user_id, user_tg_id,
                )
                return True
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "send_voice_notification: send_file failed, falling back: %s", exc
                )

        # Text fallback
        return await self.send_alert(user_id, text, priority="normal")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get_user_telegram_id(self, user_id: UUID) -> int | None:
        """
        Look up the Telegram user ID for a JARVIS user from the database.

        Returns None if the user has no linked Telegram account.
        """
        from sqlalchemy import text as sa_text

        try:
            result = await self._db.execute(
                sa_text(
                    "SELECT telegram_id FROM users WHERE id = :user_id AND is_deleted = false"
                ),
                {"user_id": str(user_id)},
            )
            row = result.fetchone()
            if row and row[0]:
                return int(row[0])
        except Exception as exc:  # noqa: BLE001
            logger.warning("_get_user_telegram_id failed: %s", exc)
        return None

    @staticmethod
    async def _synthesise_voice(text: str, settings: Any) -> bytes:
        """
        Call ElevenLabs API to synthesise text into audio bytes (OGG/Opus).

        Args:
            text:     Text to synthesise.
            settings: Application settings (ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID).

        Returns:
            Raw audio bytes.

        Raises:
            Exception: On API error.
        """
        import asyncio

        import aiohttp

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{settings.ELEVENLABS_VOICE_ID}"
        headers = {
            "xi-api-key": settings.ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text[:500],  # ElevenLabs has a character limit per request
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                return await resp.read()
