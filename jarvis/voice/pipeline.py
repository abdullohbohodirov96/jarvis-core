"""
Full voice conversation pipeline for the JARVIS desktop AI assistant.

Orchestrates the complete audio → STT → AI agent → TTS → audio cycle,
including microphone recording with silence detection and continuous wake-word
mode.

Provides:
- VoiceInput:    raw audio descriptor
- VoiceResult:   structured pipeline result
- VoicePipeline: the orchestrator
- get_pipeline(): factory that wires up default STT, TTS, and agent
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SAMPLE_RATE: int = 16_000
_DEFAULT_CHANNELS: int = 1
_DEFAULT_DTYPE: str = "int16"

# Energy threshold used for silence detection (0–32767 scale for 16-bit audio).
# Tune this for your microphone / room noise level.
_DEFAULT_SILENCE_THRESHOLD: float = 500.0

# Seconds of consecutive silence before recording stops
_VOICE_SILENCE_DURATION: float = 1.5

# Block size for microphone reads (samples per callback)
_MIC_BLOCK_SAMPLES: int = 512


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VoiceInput:
    """Container for a raw audio capture."""

    audio_data: bytes    # WAV-encoded bytes
    duration: float      # seconds
    sample_rate: int     # Hz


@dataclass
class VoiceResult:
    """Full result of a pipeline round-trip."""

    transcript: str
    response_text: str
    audio_response: Optional[bytes]          # MP3 bytes or None
    actions_taken: list[str] = field(default_factory=list)
    processing_time: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """True when there was no error."""
        return self.error is None

    def __repr__(self) -> str:
        return (
            f"<VoiceResult ok={self.ok} "
            f"transcript={self.transcript[:40]!r} "
            f"time={self.processing_time:.2f}s>"
        )


# ---------------------------------------------------------------------------
# VoicePipeline
# ---------------------------------------------------------------------------


class VoicePipeline:
    """
    Orchestrates the JARVIS voice interaction loop:

        1. [Optional] Wait for wake word
        2. Record microphone audio until silence
        3. Transcribe via Whisper (STT)
        4. Send transcript to AI agent
        5. Synthesise AI reply (TTS)
        6. Play audio / return result

    All blocking operations (sounddevice reads, Whisper inference, Edge TTS
    synthesis) are dispatched to executor threads so the asyncio event loop
    stays responsive.

    Args:
        stt:      A SpeechToText instance (from ``jarvis.voice.stt``).
        tts:      A TextToSpeech instance (from ``jarvis.voice.tts``).
        ai_agent: Any object with an async ``chat(text: str) -> str`` method
                  (or ``ask(user_id, text) -> str`` / ``run(text) -> str``).
                  A simple echo agent is used if None.
    """

    def __init__(
        self,
        stt: Any,
        tts: Any,
        ai_agent: Any,
    ) -> None:
        self._stt = stt
        self._tts = tts
        self._agent = ai_agent
        self._active: bool = False
        self._stop_event: asyncio.Event = asyncio.Event()
        logger.info("VoicePipeline created.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True while continuous mode is running."""
        return self._active

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def process(self, audio_data: bytes) -> VoiceResult:
        """
        Full pipeline: audio bytes → transcribe → AI agent → TTS → result.

        Args:
            audio_data: WAV audio bytes captured from the microphone.

        Returns:
            VoiceResult with transcript, response text, and MP3 audio.
        """
        start = time.perf_counter()
        actions: list[str] = []

        try:
            # ── 1. Transcribe ────────────────────────────────────────────────
            actions.append("transcribe")
            transcription = await self._stt.transcribe(audio_data)
            transcript = transcription.text.strip()
            logger.info("Transcript: %r", transcript)

            if not transcript:
                response_text = "I didn't catch that. Could you please repeat?"
                audio_response = await self._synthesize_safe(response_text)
                return VoiceResult(
                    transcript="",
                    response_text=response_text,
                    audio_response=audio_response,
                    actions_taken=actions,
                    processing_time=time.perf_counter() - start,
                )

            # ── 2. AI agent ──────────────────────────────────────────────────
            actions.append("ai_agent")
            response_text = await self._call_agent(transcript)
            logger.info("Agent response: %r", response_text[:120])

            # ── 3. TTS ───────────────────────────────────────────────────────
            actions.append("synthesize")
            audio_response = await self._synthesize_safe(response_text)

            return VoiceResult(
                transcript=transcript,
                response_text=response_text,
                audio_response=audio_response,
                actions_taken=actions,
                processing_time=time.perf_counter() - start,
            )

        except Exception as exc:
            logger.error("VoicePipeline.process() error: %s", exc, exc_info=True)
            return VoiceResult(
                transcript="",
                response_text="",
                audio_response=None,
                actions_taken=actions,
                processing_time=time.perf_counter() - start,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Microphone recording
    # ------------------------------------------------------------------

    async def record_until_silence(
        self,
        max_duration: float = 10.0,
        silence_threshold: float = _DEFAULT_SILENCE_THRESHOLD,
        silence_duration: float = _VOICE_SILENCE_DURATION,
    ) -> bytes:
        """
        Open the default microphone, collect audio, and stop when silence is
        detected for *silence_duration* seconds or *max_duration* elapses.

        Uses sounddevice's InputStream with a fixed block size.  The blocking
        stream.read() call is dispatched to an executor thread.

        Args:
            max_duration:      Hard ceiling on recording time (seconds).
            silence_threshold: RMS amplitude below which audio is "silent"
                               (0–32767 for 16-bit samples).
            silence_duration:  Consecutive seconds of silence before stopping.

        Returns:
            WAV-encoded bytes of the captured audio.

        Raises:
            ImportError: If sounddevice is not installed.
            RuntimeError: If the microphone cannot be opened.
        """
        try:
            import sounddevice as sd  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sounddevice is not installed. Run: pip install sounddevice"
            ) from exc

        loop = asyncio.get_running_loop()
        frames: list[np.ndarray] = []
        silent_samples = 0
        required_silent_samples = int(silence_duration * _DEFAULT_SAMPLE_RATE)
        speech_started = False
        deadline = time.monotonic() + max_duration

        # Open the stream in a thread so we don't block the event loop during
        # stream.open().
        def _open_stream() -> Any:
            s = sd.InputStream(
                samplerate=_DEFAULT_SAMPLE_RATE,
                channels=_DEFAULT_CHANNELS,
                dtype=_DEFAULT_DTYPE,
                blocksize=_MIC_BLOCK_SAMPLES,
            )
            s.start()
            return s

        stream = await loop.run_in_executor(None, _open_stream)
        logger.debug("record_until_silence: microphone stream opened.")

        try:
            while time.monotonic() < deadline:
                # Read one block (blocking call → executor)
                data, overflowed = await loop.run_in_executor(
                    None, stream.read, _MIC_BLOCK_SAMPLES
                )
                if overflowed:
                    logger.debug("record_until_silence: audio overflow.")

                chunk: np.ndarray = (
                    data[:, 0] if data.ndim > 1 else data.flatten()
                )
                frames.append(chunk)

                rms = self._rms_energy(chunk)
                is_silent_now = rms < silence_threshold

                if not is_silent_now:
                    speech_started = True
                    silent_samples = 0
                elif speech_started:
                    silent_samples += len(chunk)
                    if silent_samples >= required_silent_samples:
                        logger.debug(
                            "record_until_silence: silence detected after %.2f s.",
                            len(frames) * _MIC_BLOCK_SAMPLES / _DEFAULT_SAMPLE_RATE,
                        )
                        break

        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            logger.debug("record_until_silence: microphone stream closed.")

        if not frames:
            # Return empty WAV
            return self._numpy_to_wav_bytes(np.array([], dtype=np.int16))

        audio = np.concatenate(frames)
        logger.debug(
            "record_until_silence: captured %.2f s of audio.",
            len(audio) / _DEFAULT_SAMPLE_RATE,
        )
        return self._numpy_to_wav_bytes(audio)

    async def listen_and_respond(
        self,
        max_duration: float = 10.0,
    ) -> VoiceResult:
        """
        Open the microphone, record until silence, then run the full pipeline.

        Args:
            max_duration: Maximum recording time in seconds.

        Returns:
            VoiceResult from the pipeline.
        """
        logger.info("listen_and_respond: recording…")
        audio_bytes = await self.record_until_silence(max_duration=max_duration)
        duration = self._wav_duration(audio_bytes)
        logger.info("listen_and_respond: captured %.2f s, processing…", duration)
        return await self.process(audio_bytes)

    # ------------------------------------------------------------------
    # Continuous / wake-word mode
    # ------------------------------------------------------------------

    async def start_continuous_mode(self) -> None:
        """
        Run the full JARVIS interaction loop until ``stop()`` is called:

            1. Wait for the wake word ("jarvis") via WakeWordDetector
            2. Record user command until silence
            3. Run the full pipeline (STT → agent → TTS)
            4. Play the audio response
            5. Publish events via event_bus if available
            6. Repeat

        Publishes events:
            - ``voice.wake_word_detected``
            - ``voice.transcribed``      {"transcript": str}
            - ``voice.response_ready``   {"text": str}
            - ``voice.error``            {"error": str}
        """
        from jarvis.voice.wake_word import get_wake_word_detector

        detector = get_wake_word_detector()
        event_bus = _try_get_event_bus()

        self._active = True
        self._stop_event.clear()
        logger.info("VoicePipeline: continuous mode started.")

        try:
            while not self._stop_event.is_set():
                # ── Wait for wake word ───────────────────────────────────────
                logger.info("Waiting for wake word…")
                try:
                    detected = await asyncio.wait_for(
                        detector.wait_for_wake_word(),
                        timeout=None,   # wait indefinitely
                    )
                except asyncio.CancelledError:
                    break

                if not detected:
                    continue

                logger.info("Wake word detected — listening for command…")
                _publish(event_bus, "voice.wake_word_detected", {})

                # ── Record command ────────────────────────────────────────────
                try:
                    result = await self.listen_and_respond(max_duration=10.0)
                except Exception as exc:
                    logger.error("listen_and_respond failed: %s", exc)
                    _publish(event_bus, "voice.error", {"error": str(exc)})
                    continue

                if result.error:
                    _publish(event_bus, "voice.error", {"error": result.error})
                    continue

                _publish(
                    event_bus,
                    "voice.transcribed",
                    {"transcript": result.transcript},
                )
                _publish(
                    event_bus,
                    "voice.response_ready",
                    {"text": result.response_text},
                )

                # ── Play response (already synthesised inside process()) ──────
                if result.audio_response:
                    try:
                        await self._tts.speak(result.response_text)
                    except Exception as exc:
                        logger.warning(
                            "TTS playback failed (response already synthesised): %s", exc
                        )

        finally:
            self._active = False
            await detector.stop()
            logger.info("VoicePipeline: continuous mode stopped.")

    async def stop(self) -> None:
        """Signal the continuous mode loop to exit after the current turn."""
        self._stop_event.set()
        self._active = False
        logger.info("VoicePipeline: stop requested.")

    # ------------------------------------------------------------------
    # Audio utilities
    # ------------------------------------------------------------------

    def _rms_energy(self, audio_chunk: np.ndarray) -> float:
        """
        Compute the RMS energy of a 16-bit PCM chunk.

        Args:
            audio_chunk: 1-D int16 or float32 array.

        Returns:
            RMS amplitude on a 0–32767 scale.
        """
        arr = audio_chunk.astype(np.float32)
        if arr.size == 0:
            return 0.0
        # If float32 in [-1, 1] range, scale to int16 amplitude
        if arr.max() <= 1.0 and arr.min() >= -1.0:
            arr = arr * 32767.0
        return float(np.sqrt(np.mean(arr ** 2)))

    def _is_silent(
        self,
        audio_chunk: np.ndarray,
        threshold: float = _DEFAULT_SILENCE_THRESHOLD,
    ) -> bool:
        """Return True if the RMS energy of *audio_chunk* is below *threshold*."""
        return self._rms_energy(audio_chunk) < threshold

    @staticmethod
    def _numpy_to_wav_bytes(
        audio: np.ndarray,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
    ) -> bytes:
        """
        Encode a 1-D int16 (or float32) numpy array as mono 16-bit WAV bytes.

        Args:
            audio:       Audio samples (int16 or float32).
            sample_rate: Sample rate in Hz.

        Returns:
            WAV-formatted bytes.
        """
        if audio.dtype != np.int16:
            # Normalise float32 → int16
            if audio.size > 0 and (audio.max() <= 1.0 and audio.min() >= -1.0):
                audio = (audio * 32767).astype(np.int16)
            else:
                audio = np.clip(audio, -32768, 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    @staticmethod
    def _wav_duration(wav_bytes: bytes) -> float:
        """Return the duration of WAV bytes in seconds."""
        try:
            with wave.open(io.BytesIO(wav_bytes)) as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _call_agent(self, text: str) -> str:
        """
        Dispatch *text* to the AI agent and return the response string.

        Supports agents with any of the following async methods:
        - ``chat(text) -> str``
        - ``ask(user_id, text) -> str``  (user_id="jarvis-voice")
        - ``run(text) -> str``
        - ``complete(text) -> str``

        Falls back to an echo if none match.
        """
        try:
            if hasattr(self._agent, "chat"):
                response = await self._agent.chat(text)
            elif hasattr(self._agent, "ask"):
                response = await self._agent.ask("jarvis-voice", text)
            elif hasattr(self._agent, "run"):
                response = await self._agent.run(text)
            elif hasattr(self._agent, "complete"):
                response = await self._agent.complete(text)
            else:
                logger.warning(
                    "Agent has no recognised async method; echoing input."
                )
                response = f"You said: {text}"
            return str(response).strip()
        except Exception as exc:
            logger.error("_call_agent error: %s", exc, exc_info=True)
            return "I'm sorry, I encountered an error processing your request."

    async def _synthesize_safe(self, text: str) -> Optional[bytes]:
        """
        Synthesise *text* and return MP3 bytes, or None on failure.
        Exceptions are logged but not propagated so the pipeline never crashes
        solely due to TTS errors.
        """
        try:
            return await self._tts.synthesize(text)
        except Exception as exc:
            logger.error("TTS synthesis failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Event bus helper (optional)
# ---------------------------------------------------------------------------


def _try_get_event_bus() -> Any:
    """Try to import the JARVIS event bus; return None if unavailable."""
    try:
        from jarvis.core.event_bus import get_event_bus  # type: ignore[import]
        return get_event_bus()
    except ImportError:
        return None


def _publish(event_bus: Any, event: str, data: dict) -> None:
    """Publish an event if the event bus is available."""
    if event_bus is None:
        return
    try:
        if asyncio.iscoroutinefunction(event_bus.publish):
            asyncio.ensure_future(event_bus.publish(event, data))
        else:
            event_bus.publish(event, data)
    except Exception as exc:
        logger.debug("event_bus.publish(%r) failed: %s", event, exc)


# ---------------------------------------------------------------------------
# Default echo agent (used when no agent is provided)
# ---------------------------------------------------------------------------


class _EchoAgent:
    """Fallback agent that echoes the input — useful for testing."""

    async def chat(self, text: str) -> str:  # noqa: D102
        return f"You said: {text}"


# ---------------------------------------------------------------------------
# Factory / singleton
# ---------------------------------------------------------------------------

_pipeline_instance: Optional[VoicePipeline] = None


def get_pipeline(ai_agent: Any = None) -> VoicePipeline:
    """
    Return the process-level VoicePipeline singleton.

    Wires up the default SpeechToText ("base" model) and TextToSpeech
    ("en-US-GuyNeural") singletons.  The *ai_agent* is used only on the first
    call; pass None to fall back to the built-in echo agent.

    Args:
        ai_agent: Optional AI agent instance to wire into the pipeline.

    Returns:
        The singleton VoicePipeline.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        from jarvis.voice.stt import get_stt
        from jarvis.voice.tts import get_tts

        stt = get_stt(model_name="base")
        tts = get_tts()
        agent = ai_agent if ai_agent is not None else _EchoAgent()
        _pipeline_instance = VoicePipeline(stt=stt, tts=tts, ai_agent=agent)
        logger.info("VoicePipeline singleton created.")
    return _pipeline_instance
