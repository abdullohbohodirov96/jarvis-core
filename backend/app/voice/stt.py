"""
Speech-to-Text using OpenAI Whisper (local model or API).

Provides:
- WhisperSTT: async transcription via local whisper or OpenAI Whisper API
- TranscriptionResult: typed result dataclass
- get_stt_engine(): process-level singleton
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import struct
import tempfile
import wave
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionResult:
    """Structured result from a Whisper transcription call."""

    text: str
    language: str
    confidence: float
    segments: list[dict[str, Any]] = field(default_factory=list)
    duration: float = 0.0

    def __str__(self) -> str:
        return self.text


# ---------------------------------------------------------------------------
# WhisperSTT
# ---------------------------------------------------------------------------


class WhisperSTT:
    """
    Async Speech-to-Text engine backed by either:
    - A local openai-whisper model (use_api=False, default)
    - The OpenAI Whisper HTTP API (use_api=True)

    The local model is loaded lazily on first use to keep startup fast.
    """

    def __init__(
        self,
        model_name: str = "base",
        use_api: bool = False,
    ) -> None:
        self._model_name = model_name
        self._use_api = use_api
        self._model: Any = None          # lazy-loaded local whisper model
        self._model_lock = asyncio.Lock()

        if use_api:
            from openai import AsyncOpenAI
            from backend.app.core.config import get_settings
            settings = get_settings()
            self._openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            self._openai_client = None

        logger.info(
            "WhisperSTT init: model=%s use_api=%s",
            model_name,
            use_api,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _load_model(self) -> Any:
        """Load and cache the local Whisper model (thread-safe)."""
        async with self._model_lock:
            if self._model is None:
                logger.info("Loading local Whisper model '%s' …", self._model_name)
                loop = asyncio.get_running_loop()
                import whisper  # type: ignore[import]

                self._model = await loop.run_in_executor(
                    None, whisper.load_model, self._model_name
                )
                logger.info("Whisper model '%s' loaded.", self._model_name)
        return self._model

    def _preprocess_audio(self, audio_bytes: bytes) -> bytes:
        """
        Normalise raw audio bytes to a 16 kHz mono WAV suitable for Whisper.

        Handles arbitrary PCM / WAV input.  If scipy / soundfile are available
        they are used for resampling; otherwise a simple fallback is applied.
        """
        try:
            import numpy as np
            import scipy.io.wavfile as wav_io  # type: ignore[import]
            import scipy.signal as signal  # type: ignore[import]

            # Parse input WAV
            buf_in = io.BytesIO(audio_bytes)
            try:
                rate, data = wav_io.read(buf_in)
            except Exception:
                # Not a valid WAV; assume raw 16-bit PCM at 16 kHz mono
                data = np.frombuffer(audio_bytes, dtype=np.int16)
                rate = 16000

            # Mono conversion
            if data.ndim > 1:
                data = data.mean(axis=1)

            # Resample to 16 kHz
            if rate != 16000:
                num_samples = int(len(data) * 16000 / rate)
                data = signal.resample(data, num_samples)

            # Normalise to int16 range
            if data.dtype != np.int16:
                max_val = np.max(np.abs(data))
                if max_val > 0:
                    data = (data / max_val * 32767).astype(np.int16)
                else:
                    data = data.astype(np.int16)

            # Write to WAV bytes
            buf_out = io.BytesIO()
            wav_io.write(buf_out, 16000, data)
            return buf_out.getvalue()

        except ImportError:
            logger.warning(
                "scipy not available; returning audio bytes unchanged."
            )
            return audio_bytes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def transcribe_file(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file from disk.

        Args:
            audio_path: Absolute path to the audio file.
            language:   ISO-639-1 language hint. If None, auto-detects.

        Returns:
            TranscriptionResult with text, language, confidence, segments.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio_bytes = path.read_bytes()
        return await self.transcribe_bytes(audio_bytes, language=language)

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe raw audio bytes.

        The bytes are preprocessed to 16 kHz mono WAV before passing to
        Whisper.
        """
        processed = await asyncio.get_running_loop().run_in_executor(
            None, self._preprocess_audio, audio_bytes
        )

        if self._use_api:
            return await self._transcribe_bytes_api(processed, language)
        else:
            return await self._transcribe_bytes_local(processed, language)

    async def _transcribe_bytes_api(
        self,
        audio_bytes: bytes,
        language: str | None,
    ) -> TranscriptionResult:
        """Use the OpenAI Whisper HTTP API."""
        assert self._openai_client is not None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                response = await self._openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language=language,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                )

            segments: list[dict[str, Any]] = []
            if hasattr(response, "words") and response.words:
                segments = [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                    }
                    for w in response.words
                ]

            duration: float = getattr(response, "duration", 0.0) or 0.0

            return TranscriptionResult(
                text=(response.text or "").strip(),
                language=getattr(response, "language", language or "en") or "en",
                confidence=1.0,  # API does not expose per-utterance confidence
                segments=segments,
                duration=duration,
            )
        finally:
            os.unlink(tmp_path)

    async def _transcribe_bytes_local(
        self,
        audio_bytes: bytes,
        language: str | None,
    ) -> TranscriptionResult:
        """Use the locally loaded openai-whisper model."""
        model = await self._load_model()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            loop = asyncio.get_running_loop()

            def _run() -> dict[str, Any]:
                return model.transcribe(  # type: ignore[no-any-return]
                    tmp_path,
                    language=language,
                    word_timestamps=True,
                    verbose=False,
                )

            result: dict[str, Any] = await loop.run_in_executor(None, _run)

            text: str = (result.get("text") or "").strip()
            detected_lang: str = result.get("language") or language

            # Flatten word-level segments
            segments: list[dict[str, Any]] = []
            for seg in result.get("segments", []):
                for word_info in seg.get("words", []):
                    segments.append(
                        {
                            "word": word_info.get("word", "").strip(),
                            "start": word_info.get("start", 0.0),
                            "end": word_info.get("end", 0.0),
                            "probability": word_info.get("probability", 1.0),
                        }
                    )

            # Estimate confidence as mean word probability
            if segments:
                confidence = sum(
                    s.get("probability", 1.0) for s in segments
                ) / len(segments)
            else:
                confidence = 1.0

            # Calculate duration from last segment end time
            duration: float = 0.0
            raw_segs = result.get("segments", [])
            if raw_segs:
                duration = float(raw_segs[-1].get("end", 0.0))

            return TranscriptionResult(
                text=text,
                language=detected_lang,
                confidence=confidence,
                segments=segments,
                duration=duration,
            )
        finally:
            os.unlink(tmp_path)

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
    ) -> AsyncGenerator[str, None]:
        """
        Streaming transcription: collect chunks, transcribe incrementally.

        Yields partial transcript strings as each accumulated chunk is
        processed.  Chunks are processed in ~2-second windows to reduce
        latency.
        """
        buffer = bytearray()
        chunk_size_bytes = 16000 * 2 * 2  # ~2 seconds at 16kHz 16-bit mono

        async for chunk in audio_stream:
            buffer.extend(chunk)
            if len(buffer) >= chunk_size_bytes:
                try:
                    result = await self.transcribe_bytes(bytes(buffer))
                    if result.text:
                        yield result.text
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Stream chunk transcription failed: %s", exc)
                buffer.clear()

        # Flush remaining audio
        if buffer:
            try:
                result = await self.transcribe_bytes(bytes(buffer))
                if result.text:
                    yield result.text
            except Exception as exc:  # noqa: BLE001
                logger.warning("Stream flush transcription failed: %s", exc)

    def detect_language(self, audio_path: str) -> str:
        """
        Detect the spoken language in an audio file.

        Uses local whisper's language detection (pad30s + decode first 30s).
        Runs synchronously; wrap in run_in_executor if called from async code.

        Returns ISO-639-1 language code string (e.g. "en", "de").
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self._use_api:
            # API always auto-detects; return a sentinel that will be resolved
            # after a real transcription call.
            return "auto"

        try:
            import whisper  # type: ignore[import]
            import numpy as np

            # Load model synchronously (caller must be in a thread if async)
            model = whisper.load_model(self._model_name)
            audio = whisper.load_audio(str(path))
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)  # type: ignore[attr-defined]
            _, probs = model.detect_language(mel)  # type: ignore[attr-defined]
            detected: str = max(probs, key=probs.get)  # type: ignore[arg-type]
            return detected
        except Exception as exc:  # noqa: BLE001
            logger.warning("Language detection failed: %s", exc)
            return "en"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_stt_instance: WhisperSTT | None = None


def get_stt_engine() -> WhisperSTT:
    """
    Return the process-level WhisperSTT singleton.

    Configuration is read from environment settings on first call.
    """
    global _stt_instance
    if _stt_instance is None:
        from backend.app.core.config import get_settings
        settings = get_settings()
        use_api = bool(settings.OPENAI_API_KEY)
        _stt_instance = WhisperSTT(
            model_name=settings.WHISPER_MODEL,
            use_api=use_api,
        )
    return _stt_instance
