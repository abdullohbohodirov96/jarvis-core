"""
Full voice conversation pipeline for JARVIS.

Provides:
- VoiceResponse:     Typed result of a full voice interaction round-trip.
- VoiceCommandResult: Typed result of a parsed + executed voice command.
- VoicePipeline:    audio → STT → AI → TTS pipeline with streaming support.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VoiceResponse:
    """Full round-trip result: audio in, voice out."""

    transcript: str
    response_text: str
    audio_bytes: bytes
    processing_time: float
    command_executed: bool = False

    def __repr__(self) -> str:
        return (
            f"<VoiceResponse transcript={self.transcript!r:.40} "
            f"time={self.processing_time:.2f}s "
            f"audio_len={len(self.audio_bytes)}>"
        )


@dataclass
class VoiceCommandResult:
    """Result of parsing and executing a voice command."""

    command_type: str
    success: bool
    result: dict[str, Any] = field(default_factory=dict)
    response_text: str = ""

    def __repr__(self) -> str:
        return (
            f"<VoiceCommandResult type={self.command_type!r} "
            f"success={self.success}>"
        )


# ---------------------------------------------------------------------------
# Command patterns
# ---------------------------------------------------------------------------

_COMMAND_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("set_reminder", re.compile(
        r"(?:set|add|create)\s+(?:a\s+)?reminder\s+(?:for\s+)?(.+)",
        re.IGNORECASE,
    )),
    ("send_telegram", re.compile(
        r"(?:send|message|text)\s+(.+?)\s+(?:on\s+telegram\s*)?(?:saying\s+)?(.+)",
        re.IGNORECASE,
    )),
    ("list_tasks", re.compile(
        r"(?:what(?:'s|\s+are|\s+is)\s+(?:my\s+)?|show(?:\s+me)?\s+(?:my\s+)?|list(?:\s+my)?\s+)tasks?",
        re.IGNORECASE,
    )),
    ("create_task", re.compile(
        r"(?:add|create|make)\s+(?:a\s+)?task\s+(?:to\s+|for\s+)?(.+)",
        re.IGNORECASE,
    )),
    ("complete_task", re.compile(
        r"(?:mark|complete|finish|done)\s+task\s+(.+)",
        re.IGNORECASE,
    )),
    ("check_messages", re.compile(
        r"(?:check|read|show)\s+(?:my\s+)?(?:telegram\s+)?messages?",
        re.IGNORECASE,
    )),
    ("play_music", re.compile(
        r"play\s+(?:some\s+)?(.+?)(?:\s+music|\s+song)?$",
        re.IGNORECASE,
    )),
    ("set_timer", re.compile(
        r"(?:set|start)\s+(?:a\s+)?timer\s+(?:for\s+)?(\d+)\s+(second|minute|hour)s?",
        re.IGNORECASE,
    )),
    ("weather", re.compile(
        r"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?weather(?:\s+(?:like|today|now|forecast))?",
        re.IGNORECASE,
    )),
    ("stop", re.compile(
        r"^(?:stop|cancel|nevermind|never\s+mind|abort)$",
        re.IGNORECASE,
    )),
]


# ---------------------------------------------------------------------------
# VoicePipeline
# ---------------------------------------------------------------------------


class VoicePipeline:
    """
    Orchestrates the full voice interaction lifecycle:

        1. Preprocess audio
        2. Speech-to-Text (WhisperSTT)
        3. Command detection (optional fast path)
        4. AI response generation
        5. Text-to-Speech (TTSManager)

    Supports both blocking (process_voice_input) and streaming
    (stream_response) modes.
    """

    def __init__(
        self,
        stt: Any,
        tts: Any,
        agent: Any,
        user_id: str,
    ) -> None:
        """
        Args:
            stt:     WhisperSTT instance (from backend.app.voice.stt).
            tts:     TTSManager instance (from backend.app.voice.tts).
            agent:   AI agent / conversation manager with an
                     ``ask(user_id, text)`` coroutine that returns a str.
            user_id: JARVIS user identifier string.
        """
        self._stt = stt
        self._tts = tts
        self._agent = agent
        self._user_id = user_id
        self._start_time: float = 0.0

        logger.info("VoicePipeline init: user_id=%s", user_id)

    # ------------------------------------------------------------------
    # Primary pipeline
    # ------------------------------------------------------------------

    async def process_voice_input(
        self,
        audio_bytes: bytes,
    ) -> VoiceResponse:
        """
        Complete round-trip: audio bytes → transcript → AI response → audio.

        Args:
            audio_bytes: Raw WAV audio bytes from microphone or upload.

        Returns:
            VoiceResponse with transcript, AI response text and audio.
        """
        start = time.perf_counter()

        # 1. Transcribe
        transcription = await self._stt.transcribe_bytes(audio_bytes)
        transcript = transcription.text.strip()
        logger.debug("Transcribed: %r", transcript)

        if not transcript:
            response_text = "I didn't catch that. Could you repeat?"
            audio = await self._tts.speak(response_text)
            return VoiceResponse(
                transcript="",
                response_text=response_text,
                audio_bytes=audio,
                processing_time=time.perf_counter() - start,
                command_executed=False,
            )

        # 2. Check for a voice command (fast path)
        cmd_result = await self.handle_command(transcript)

        if cmd_result.success and cmd_result.command_type != "unknown":
            response_text = cmd_result.response_text or "Done."
            audio = await self._tts.speak(response_text)
            elapsed = time.perf_counter() - start
            return VoiceResponse(
                transcript=transcript,
                response_text=response_text,
                audio_bytes=audio,
                processing_time=elapsed,
                command_executed=True,
            )

        # 3. AI conversation path
        response_text = await self._call_agent(transcript)
        audio = await self._tts.speak(response_text)
        elapsed = time.perf_counter() - start

        return VoiceResponse(
            transcript=transcript,
            response_text=response_text,
            audio_bytes=audio,
            processing_time=elapsed,
            command_executed=False,
        )

    async def process_voice_file(
        self,
        file_path: str,
    ) -> VoiceResponse:
        """
        Process audio from a file on disk.

        Args:
            file_path: Absolute path to the audio file.

        Returns:
            VoiceResponse.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        audio_bytes = path.read_bytes()

        # Convert non-WAV formats via AudioProcessor
        suffix = path.suffix.lstrip(".").lower()
        if suffix != "wav":
            from backend.app.voice.audio_processor import AudioProcessor
            audio_bytes = AudioProcessor.convert_to_wav(audio_bytes, suffix)

        return await self.process_voice_input(audio_bytes)

    # ------------------------------------------------------------------
    # Streaming pipeline
    # ------------------------------------------------------------------

    async def stream_response(
        self,
        audio_bytes: bytes,
    ) -> AsyncGenerator[bytes, None]:
        """
        Streaming pipeline: transcribe → AI (streaming) → TTS (streaming).

        Yields TTS audio byte chunks as they are generated.  The AI response
        is synthesised sentence-by-sentence to minimise first-audio latency.

        Args:
            audio_bytes: WAV audio bytes to transcribe.

        Yields:
            MP3 audio byte chunks.
        """
        # 1. Transcribe
        transcription = await self._stt.transcribe_bytes(audio_bytes)
        transcript = transcription.text.strip()

        if not transcript:
            silence_response = "I didn't catch that."
            async for chunk in self._tts.speak_stream(silence_response):
                yield chunk
            return

        # 2. Attempt command detection first
        cmd_result = await self.handle_command(transcript)
        if cmd_result.success and cmd_result.command_type != "unknown":
            async for chunk in self._tts.speak_stream(cmd_result.response_text):
                yield chunk
            return

        # 3. Stream AI response, synthesise sentence by sentence
        sentence_buffer = ""
        sentence_end = re.compile(r"(?<=[.!?])\s+")

        try:
            response_stream = await self._call_agent_stream(transcript)
            async for text_chunk in response_stream:
                sentence_buffer += text_chunk

                # Flush complete sentences
                parts = sentence_end.split(sentence_buffer)
                for sentence in parts[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        async for audio_chunk in self._tts.speak_stream(sentence):
                            yield audio_chunk

                sentence_buffer = parts[-1]

            # Flush remaining
            if sentence_buffer.strip():
                async for audio_chunk in self._tts.speak_stream(sentence_buffer.strip()):
                    yield audio_chunk

        except Exception as exc:  # noqa: BLE001
            logger.error("stream_response AI call failed: %s", exc)
            error_msg = "Sorry, I encountered an error generating a response."
            async for chunk in self._tts.speak_stream(error_msg):
                yield chunk

    # ------------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------------

    async def handle_command(self, text: str) -> VoiceCommandResult:
        """
        Parse and attempt to execute a voice command.

        Recognised command types:
            set_reminder, send_telegram, list_tasks, create_task,
            complete_task, check_messages, play_music, set_timer,
            weather, stop

        Returns:
            VoiceCommandResult; command_type="unknown" if no pattern matched.
        """
        parsed = self._parse_command(text)

        if parsed is None:
            return VoiceCommandResult(
                command_type="unknown",
                success=False,
                response_text="",
            )

        cmd_type = parsed["type"]
        groups = parsed.get("groups", ())

        try:
            if cmd_type == "set_reminder":
                reminder_text = groups[0] if groups else text
                result = await self._cmd_set_reminder(reminder_text)
                return VoiceCommandResult(
                    command_type=cmd_type,
                    success=True,
                    result=result,
                    response_text=f"Reminder set: {reminder_text}",
                )

            elif cmd_type == "send_telegram":
                recipient = groups[0] if groups else ""
                message = groups[1] if len(groups) > 1 else ""
                result = await self._cmd_send_telegram(recipient, message)
                return VoiceCommandResult(
                    command_type=cmd_type,
                    success=result.get("sent", False),
                    result=result,
                    response_text=(
                        f"Message sent to {recipient}."
                        if result.get("sent")
                        else f"Couldn't send message to {recipient}."
                    ),
                )

            elif cmd_type == "list_tasks":
                result = await self._cmd_list_tasks()
                tasks: list[str] = result.get("tasks", [])
                if tasks:
                    task_list = ", ".join(tasks[:5])
                    response = f"You have {len(tasks)} tasks: {task_list}."
                else:
                    response = "You have no pending tasks."
                return VoiceCommandResult(
                    command_type=cmd_type,
                    success=True,
                    result=result,
                    response_text=response,
                )

            elif cmd_type == "create_task":
                task_text = groups[0] if groups else text
                result = await self._cmd_create_task(task_text)
                return VoiceCommandResult(
                    command_type=cmd_type,
                    success=True,
                    result=result,
                    response_text=f"Task created: {task_text}",
                )

            elif cmd_type == "set_timer":
                amount = int(groups[0]) if groups else 1
                unit = groups[1].lower() if len(groups) > 1 else "minute"
                result = await self._cmd_set_timer(amount, unit)
                return VoiceCommandResult(
                    command_type=cmd_type,
                    success=True,
                    result=result,
                    response_text=f"Timer set for {amount} {unit}s.",
                )

            elif cmd_type == "weather":
                result = await self._cmd_get_weather()
                return VoiceCommandResult(
                    command_type=cmd_type,
                    success=True,
                    result=result,
                    response_text=result.get(
                        "description",
                        "Weather information unavailable.",
                    ),
                )

            elif cmd_type == "check_messages":
                result = await self._cmd_check_messages()
                count = result.get("unread_count", 0)
                response = (
                    f"You have {count} unread messages."
                    if count
                    else "No unread messages."
                )
                return VoiceCommandResult(
                    command_type=cmd_type,
                    success=True,
                    result=result,
                    response_text=response,
                )

            elif cmd_type == "stop":
                return VoiceCommandResult(
                    command_type=cmd_type,
                    success=True,
                    result={},
                    response_text="Okay, stopping.",
                )

            else:
                return VoiceCommandResult(
                    command_type=cmd_type,
                    success=False,
                    response_text="Command not implemented yet.",
                )

        except Exception as exc:  # noqa: BLE001
            logger.error("Command execution failed (%s): %s", cmd_type, exc)
            return VoiceCommandResult(
                command_type=cmd_type,
                success=False,
                result={"error": str(exc)},
                response_text=f"Sorry, I couldn't execute that command: {exc}",
            )

    def _parse_command(self, text: str) -> dict[str, Any] | None:
        """
        Attempt to match *text* against known command patterns.

        Returns:
            {"type": str, "groups": tuple} if a pattern matched, else None.
        """
        text = text.strip()
        for cmd_type, pattern in _COMMAND_PATTERNS:
            m = pattern.search(text)
            if m:
                return {"type": cmd_type, "groups": m.groups()}
        return None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_status(self) -> dict[str, Any]:
        """Return runtime status of the pipeline components."""
        return {
            "user_id": self._user_id,
            "stt_engine": type(self._stt).__name__,
            "tts_manager": type(self._tts).__name__,
            "agent": type(self._agent).__name__,
        }

    # ------------------------------------------------------------------
    # Private: AI agent call
    # ------------------------------------------------------------------

    async def _call_agent(self, text: str) -> str:
        """
        Call the AI agent and return the full response string.

        Supports agents with:
        - async ask(user_id, text) -> str
        - async chat(text) -> str
        - async complete(text) -> str
        """
        try:
            if hasattr(self._agent, "ask"):
                return str(await self._agent.ask(self._user_id, text))
            elif hasattr(self._agent, "chat"):
                return str(await self._agent.chat(text))
            elif hasattr(self._agent, "complete"):
                return str(await self._agent.complete(text))
            else:
                return f"Echo: {text}"
        except Exception as exc:  # noqa: BLE001
            logger.error("Agent call failed: %s", exc)
            return "I'm sorry, I encountered an issue processing your request."

    async def _call_agent_stream(
        self,
        text: str,
    ) -> AsyncGenerator[str, None]:
        """Return an async generator that yields AI response text chunks."""
        try:
            if hasattr(self._agent, "ask_stream"):
                async for chunk in self._agent.ask_stream(self._user_id, text):
                    yield str(chunk)
            else:
                # Non-streaming fallback: yield the full response in one shot
                full_response = await self._call_agent(text)
                yield full_response
        except Exception as exc:  # noqa: BLE001
            logger.error("Agent streaming call failed: %s", exc)
            yield "I'm sorry, I encountered an issue processing your request."

    # ------------------------------------------------------------------
    # Private: Command implementations
    # ------------------------------------------------------------------

    async def _cmd_set_reminder(self, reminder_text: str) -> dict[str, Any]:
        """Create a reminder in the task system (best-effort)."""
        try:
            # Attempt to delegate to Celery task queue
            from backend.app.workers.celery_app import get_celery_app
            app = get_celery_app()
            app.send_task(
                "backend.app.workers.tasks.reminder_tasks.send_task_reminder",
                kwargs={"reminder_text": reminder_text, "user_id": self._user_id},
            )
            return {"scheduled": True, "text": reminder_text}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not schedule reminder via Celery: %s", exc)
            return {"scheduled": False, "text": reminder_text, "error": str(exc)}

    async def _cmd_send_telegram(
        self,
        recipient: str,
        message: str,
    ) -> dict[str, Any]:
        """Queue a Telegram message via Celery."""
        try:
            from backend.app.workers.celery_app import get_celery_app
            app = get_celery_app()
            app.send_task(
                "backend.app.workers.tasks.telegram_tasks.send_telegram_message",
                kwargs={
                    "user_id": self._user_id,
                    "recipient": recipient,
                    "text": message,
                },
            )
            return {"sent": True, "recipient": recipient}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not queue Telegram message: %s", exc)
            return {"sent": False, "error": str(exc)}

    async def _cmd_list_tasks(self) -> dict[str, Any]:
        """Return pending tasks for the user (stub returning empty list)."""
        return {"tasks": [], "count": 0}

    async def _cmd_create_task(self, task_text: str) -> dict[str, Any]:
        """Create a task record (stub)."""
        return {"created": True, "text": task_text}

    async def _cmd_set_timer(
        self,
        amount: int,
        unit: str,
    ) -> dict[str, Any]:
        """Schedule a timer via asyncio (fires a log message on expiry)."""
        seconds_map = {"second": 1, "minute": 60, "hour": 3600}
        seconds = amount * seconds_map.get(unit, 60)

        async def _fire() -> None:
            await asyncio.sleep(seconds)
            logger.info("Timer expired: %d %s(s) for user %s", amount, unit, self._user_id)

        asyncio.ensure_future(_fire())
        return {"seconds": seconds, "amount": amount, "unit": unit}

    async def _cmd_get_weather(self) -> dict[str, Any]:
        """Weather stub — integrate a weather API for production use."""
        return {
            "description": "Weather information is not available without an API key.",
            "available": False,
        }

    async def _cmd_check_messages(self) -> dict[str, Any]:
        """Return unread Telegram message count stub."""
        return {"unread_count": 0}
