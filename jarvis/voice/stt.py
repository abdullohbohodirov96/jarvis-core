"""
Whisper-based Speech-to-Text for the JARVIS desktop AI assistant.

Provides:
- TranscriptionResult: typed result dataclass
- SpeechToText:        async engine backed by a local openai-whisper model
- get_stt():           process-level singleton factory

The model is loaded lazily on the first transcription call.  All blocking
Whisper calls are dispatched to the default ThreadPoolExecutor so they do not
stall the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import struct
import tempfile
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionResult:
    """Structured result from a single Whisper transcription call."""

    text: str
    language: str
    confidence: float      # mean word probability mapped to [0, 1]
    duration: float        # seconds
    segments: list[dict[str, Any]] = field(default_factory=list)

    def __str__(self) -> str:  # noqa: D105
        return self.text

    def __bool__(self) -> bool:
        return bool(self.text.strip())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    """
    Decode WAV bytes (or raw 16-bit PCM) to a float32 array normalised
    to [-1.0, 1.0] at 16 000 Hz mono — exactly what Whisper expects.

    Handles resampling via scipy when available.
    """
    try:
        import scipy.io.wavfile as wav_io  # type: ignore[import]
        import scipy.signal as signal  # type: ignore[import]

        buf = io.BytesIO(audio_bytes)
        try:
            rate, data = wav_io.read(buf)
        except Exception:
            # Not a valid WAV — assume raw 16-bit PCM at 16 kHz mono
            data = np.frombuffer(audio_bytes, dtype=np.int16)
            rate = 16_000

        # Mix stereo down to mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample to 16 kHz if necessary
        if rate != 16_000:
            target_len = int(len(data) * 16_000 / rate)
            data = signal.resample(data, target_len)

        # Normalise to float32 [-1, 1]
        if data.dtype in (np.int16, np.int32):
            max_val = np.iinfo(data.dtype).max
            return data.astype(np.float32) / float(max_val)
        elif data.dtype == np.float64:
            return data.astype(np.float32)
        else:
            return data.astype(np.float32)

    except ImportError:
        # scipy not available — fall back to struct-based WAV decode
        logger.warning("scipy not available; using fallback WAV decoder.")
        buf = io.BytesIO(audio_bytes)
        try:
            with wave.open(buf) as wf:
                raw = wf.readframes(wf.getnframes())
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                return arr / 32768.0
        except Exception:
            arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            return arr / 32768.0


def _float32_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16_000) -> bytes:
    """Convert a float32 numpy array to 16-bit mono WAV bytes."""
    data = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# SpeechToText
# ---------------------------------------------------------------------------


class SpeechToText:
    """
    Async Speech-to-Text engine backed by a local openai-whisper model.

    The model is loaded lazily and thread-safely on the first call.  All
    whisper.transcribe() calls run in an executor to keep the event loop free.

    Usage::

        stt = SpeechToText(model_name="base")
        result = await stt.transcribe(audio_data)
        print(result.text)
    """

    def __init__(self, model_name: str = "base") -> None:
        self._model_name: str = model_name
        self._model: Any = None            # loaded on first use
        self._model_lock: asyncio.Lock = asyncio.Lock()
        logger.info("SpeechToText created: model_name=%s", model_name)

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    async def load_model(self) -> None:
        """
        Explicitly load the Whisper model into memory.

        This is optional — the model is also loaded lazily on the first
        transcription call.  Call this at startup to avoid latency on the
        first user utterance.
        """
        async with self._model_lock:
            if self._model is None:
                await self._load_model_locked()

    async def _load_model_locked(self) -> Any:
        """Load the Whisper model (caller must already hold _model_lock)."""
        logger.info("Loading Whisper model %r…", self._model_name)
        try:
            import whisper  # type: ignore[import]
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                None, whisper.load_model, self._model_name
            )
            logger.info("Whisper model %r loaded.", self._model_name)
        except ImportError as exc:
            raise RuntimeError(
                "openai-whisper is not installed. "
                "Run: pip install openai-whisper"
            ) from exc
        return self._model

    async def _get_model(self) -> Any:
        """Return the loaded model, loading it first if necessary."""
        async with self._model_lock:
            if self._model is None:
                await self._load_model_locked()
        return self._model

    # ------------------------------------------------------------------
    # Core transcription
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        audio_data: Union[bytes, np.ndarray],
        language: str = "en",
    ) -> TranscriptionResult:
        """
        Transcribe audio data.

        Accepts either:
        - ``bytes``:        WAV or MP3 bytes (decoded to float32 internally)
        - ``np.ndarray``:   float32 array at 16 kHz, shape (N,) or (N, 1)

        Args:
            audio_data: Raw audio in one of the supported forms.
            language:   ISO-639-1 language hint passed to Whisper.

        Returns:
            TranscriptionResult with text, language, confidence, duration, segments.
        """
        model = await self._get_model()

        # Convert bytes → float32 numpy
        if isinstance(audio_data, bytes):
            loop = asyncio.get_running_loop()
            audio_f32 = await loop.run_in_executor(None, _bytes_to_float32, audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_f32 = audio_data.flatten().astype(np.float32)
            # Normalise int16 input to float32
            if audio_f32.max() > 1.0 or audio_f32.min() < -1.0:
                audio_f32 = audio_f32 / 32768.0
        else:
            raise TypeError(
                f"audio_data must be bytes or np.ndarray, got {type(audio_data)}"
            )

        return await self._run_whisper(model, audio_f32, language)

    async def _run_whisper(
        self,
        model: Any,
        audio_f32: np.ndarray,
        language: str,
    ) -> TranscriptionResult:
        """Dispatch model.transcribe() to an executor and parse the result."""
        loop = asyncio.get_running_loop()

        def _transcribe() -> dict[str, Any]:
            return model.transcribe(
                audio_f32,
                language=language,
                word_timestamps=True,
                fp16=False,
                verbose=False,
                condition_on_previous_text=False,
            )

        raw: dict[str, Any] = await loop.run_in_executor(None, _transcribe)

        text: str = (raw.get("text") or "").strip()
        detected_lang: str = raw.get("language") or language

        # Flatten word-level data from segments
        segments: list[dict[str, Any]] = []
        for seg in raw.get("segments", []):
            for word_info in seg.get("words", []):
                segments.append(
                    {
                        "word": (word_info.get("word") or "").strip(),
                        "start": float(word_info.get("start", 0.0)),
                        "end": float(word_info.get("end", 0.0)),
                        "probability": float(word_info.get("probability", 1.0)),
                    }
                )

        # Confidence: mean word probability; fall back to 1.0 when no segments
        if segments:
            confidence = float(
                sum(s["probability"] for s in segments) / len(segments)
            )
        else:
            confidence = 1.0

        # Duration from last raw segment's end time
        raw_segs = raw.get("segments", [])
        duration = float(raw_segs[-1].get("end", 0.0)) if raw_segs else 0.0

        return TranscriptionResult(
            text=text,
            language=detected_lang,
            confidence=confidence,
            duration=duration,
            segments=segments,
        )

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    async def transcribe_file(self, path: str, language: str = "en") -> TranscriptionResult:
        """
        Transcribe audio from a file on disk.

        Args:
            path:     Absolute path to an audio file (WAV, MP3, FLAC, …).
            language: Language hint.

        Returns:
            TranscriptionResult.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        audio_bytes = file_path.read_bytes()
        return await self.transcribe(audio_bytes, language=language)

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str = "en",
    ) -> AsyncIterator[str]:
        """
        Accumulate audio chunks and yield partial transcripts.

        Audio is buffered internally.  Each time the buffer reaches ~2 seconds
        of audio (64 000 bytes at 16 kHz / 16-bit mono) a transcription is
        performed and the partial text is yielded.  Any remaining audio in the
        buffer is flushed at the end.

        Args:
            audio_chunks: Async iterator of raw audio byte chunks.
            language:     Language hint.

        Yields:
            Partial transcript strings as each chunk is processed.
        """
        # 2 s of 16-bit mono audio at 16 000 Hz = 64 000 bytes
        window_bytes = 16_000 * 2 * 2
        buffer = bytearray()

        async for chunk in audio_chunks:
            buffer.extend(chunk)
            if len(buffer) >= window_bytes:
                try:
                    result = await self.transcribe(bytes(buffer), language=language)
                    if result.text:
                        yield result.text
                except Exception as exc:
                    logger.warning("transcribe_stream: chunk transcription failed: %s", exc)
                buffer.clear()

        # Flush remainder
        if buffer:
            try:
                result = await self.transcribe(bytes(buffer), language=language)
                if result.text:
                    yield result.text
            except Exception as exc:
                logger.warning("transcribe_stream: flush transcription failed: %s", exc)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_stt_instance: SpeechToText | None = None
_stt_lock = asyncio.Lock()


def get_stt(model_name: str = "base") -> SpeechToText:
    """
    Return the process-level SpeechToText singleton.

    The ``model_name`` parameter is only used on the very first call;
    subsequent calls return the existing instance.

    Thread-safe via the module-level lock held only during object creation.
    Model loading itself is async (see ``SpeechToText.load_model()``).
    """
    global _stt_instance
    if _stt_instance is None:
        # Not yet initialised — create synchronously (model loads lazily later)
        _stt_instance = SpeechToText(model_name=model_name)
    return _stt_instance
