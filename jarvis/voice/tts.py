"""
Edge-TTS Text-to-Speech for the JARVIS desktop AI assistant.

Uses Microsoft's Edge TTS service (free, no API key) via the ``edge-tts``
library for synthesis and sounddevice / pygame for local playback.

Provides:
- TextToSpeech:  async TTS engine with speak/synthesize/stream helpers
- get_tts():     process-level singleton factory
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import threading
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_VOICE = "en-US-GuyNeural"
_DEFAULT_RATE = "+0%"
_DEFAULT_VOLUME = "+0%"

# Rough maximum chars per synthesis call (Edge TTS can struggle with very long
# strings; split on sentence boundaries before this limit is reached).
_MAX_CHUNK_CHARS = 800

# Sentence boundary splitter
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_sentences(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """
    Split *text* into chunks no longer than *max_chars*.

    Splits first on sentence boundaries (.  !  ?  …), then by word if a
    single sentence exceeds the limit.  Returns at least one element (may
    equal the original text).
    """
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            chunks.append(sentence)
        else:
            # Fall back to word-level splitting
            words = sentence.split()
            current: list[str] = []
            current_len = 0
            for word in words:
                if current_len + len(word) + 1 > max_chars and current:
                    chunks.append(" ".join(current))
                    current = [word]
                    current_len = len(word)
                else:
                    current.append(word)
                    current_len += len(word) + 1
            if current:
                chunks.append(" ".join(current))

    return chunks if chunks else [text]


def _mp3_bytes_to_float32(mp3_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Decode MP3 bytes to a float32 numpy array and sample rate.

    Tries (in order):
    1. soundfile + pydub (if available)
    2. pydub alone
    3. Raw WAV fallback (if the input happens to be WAV)

    Returns:
        (audio_float32, sample_rate)
    """
    # ── attempt 1: pydub MP3 → WAV → scipy/wave decode ──────────────────────
    try:
        from pydub import AudioSegment  # type: ignore[import]
        import scipy.io.wavfile as wav_io  # type: ignore[import]

        seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        seg = seg.set_channels(1)
        wav_buf = io.BytesIO()
        seg.export(wav_buf, format="wav")
        wav_buf.seek(0)
        rate, data = wav_io.read(wav_buf)
        if data.dtype == np.int16:
            return data.astype(np.float32) / 32768.0, rate
        return data.astype(np.float32), rate
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("pydub MP3 decode failed: %s", exc)

    # ── attempt 2: soundfile (supports MP3 via libsndfile / ffmpeg) ──────────
    try:
        import soundfile as sf  # type: ignore[import]
        data, rate = sf.read(io.BytesIO(mp3_bytes), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        return data, rate
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("soundfile MP3 decode failed: %s", exc)

    # ── attempt 3: treat as WAV (fallback) ───────────────────────────────────
    try:
        import scipy.io.wavfile as wav_io  # type: ignore[import]
        rate, data = wav_io.read(io.BytesIO(mp3_bytes))
        if data.dtype == np.int16:
            return data.astype(np.float32) / 32768.0, rate
        return data.astype(np.float32), rate
    except Exception as exc:
        logger.debug("scipy WAV fallback decode failed: %s", exc)

    # ── attempt 4: plain wave module ─────────────────────────────────────────
    try:
        with wave.open(io.BytesIO(mp3_bytes)) as wf:
            raw = wf.readframes(wf.getnframes())
            rate = wf.getframerate()
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            return data, rate
    except Exception as exc:
        raise RuntimeError(
            "Cannot decode audio bytes: no suitable decoder found. "
            "Install pydub (pip install pydub) or soundfile."
        ) from exc


# ---------------------------------------------------------------------------
# TextToSpeech
# ---------------------------------------------------------------------------


class TextToSpeech:
    """
    Async Text-to-Speech engine backed by Microsoft Edge TTS.

    Features:
    - Neural voice synthesis via ``edge-tts`` (no API key needed)
    - Immediate playback via sounddevice (preferred) or pygame
    - Graceful long-text splitting at sentence boundaries
    - Thread-safe ``is_speaking`` state tracking
    - ``stop_speaking()`` to interrupt mid-playback

    Usage::

        tts = TextToSpeech(voice="en-US-GuyNeural")
        await tts.speak("Hello, I am JARVIS.")
    """

    def __init__(
        self,
        voice: str = _DEFAULT_VOICE,
        rate: str = _DEFAULT_RATE,
        volume: str = _DEFAULT_VOLUME,
    ) -> None:
        self._voice: str = voice
        self._rate: str = rate
        self._volume: str = volume

        self._speaking: bool = False
        self._speaking_lock: asyncio.Lock = asyncio.Lock()
        self._stop_requested: threading.Event = threading.Event()

        logger.info(
            "TextToSpeech created: voice=%s rate=%s volume=%s",
            voice, rate, volume,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_speaking(self) -> bool:
        """True while audio is being played back."""
        return self._speaking

    # ------------------------------------------------------------------
    # Core synthesis
    # ------------------------------------------------------------------

    async def synthesize(self, text: str) -> bytes:
        """
        Synthesise *text* and return the MP3 audio as raw bytes.

        Long texts are split at sentence boundaries and each piece is
        synthesised separately; the resulting MP3 chunks are concatenated.

        Args:
            text: Plain text to synthesise.

        Returns:
            MP3 bytes.
        """
        if not text or not text.strip():
            return b""

        try:
            import edge_tts  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "edge-tts is not installed. Run: pip install edge-tts"
            ) from exc

        pieces = _split_sentences(text.strip())
        all_chunks: list[bytes] = []

        for piece in pieces:
            if not piece.strip():
                continue
            communicate = edge_tts.Communicate(
                text=piece,
                voice=self._voice,
                rate=self._rate,
                volume=self._volume,
            )
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio":
                    all_chunks.append(chunk["data"])

        if not all_chunks:
            raise RuntimeError(
                f"edge-tts returned no audio for voice={self._voice!r}, text={text[:60]!r}"
            )
        return b"".join(all_chunks)

    async def synthesize_to_file(self, text: str, path: str) -> str:
        """
        Synthesise *text* and write the MP3 to *path*.

        Args:
            text: Plain text to synthesise.
            path: Destination file path (will be created/overwritten).

        Returns:
            The absolute path to the written file.
        """
        audio_bytes = await self.synthesize(text)
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(audio_bytes)
        logger.debug("synthesize_to_file: wrote %d bytes → %s", len(audio_bytes), dest)
        return str(dest.resolve())

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    async def speak(self, text: str) -> None:
        """
        Synthesise *text* and play the audio through the default output device.

        Blocks the caller until playback finishes (or ``stop_speaking()`` is
        called).  Sets ``is_speaking`` during the playback window.

        Long texts are split at sentence boundaries so the first sentence
        starts playing while later sentences are still being synthesised.

        Args:
            text: Plain text to speak.
        """
        if not text or not text.strip():
            return

        async with self._speaking_lock:
            self._speaking = True
            self._stop_requested.clear()
            try:
                await self._speak_internal(text)
            finally:
                self._speaking = False

    async def _speak_internal(self, text: str) -> None:
        """
        Internal implementation: synthesise sentence-by-sentence and play each
        piece as soon as it is ready so the user hears audio quickly.
        """
        pieces = _split_sentences(text.strip())

        for piece in pieces:
            if not piece.strip():
                continue
            if self._stop_requested.is_set():
                logger.debug("speak: stop requested; aborting playback.")
                break
            try:
                mp3_bytes = await self._synthesize_piece(piece)
                if mp3_bytes and not self._stop_requested.is_set():
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None, self._play_mp3_blocking, mp3_bytes
                    )
            except Exception as exc:
                logger.error("speak: error on piece %r: %s", piece[:40], exc)

    async def _synthesize_piece(self, text: str) -> bytes:
        """Synthesise a single piece (no splitting) and return MP3 bytes."""
        try:
            import edge_tts  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "edge-tts is not installed. Run: pip install edge-tts"
            ) from exc

        communicate = edge_tts.Communicate(
            text=text,
            voice=self._voice,
            rate=self._rate,
            volume=self._volume,
        )
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    def _play_mp3_blocking(self, mp3_bytes: bytes) -> None:
        """
        Decode MP3 bytes and play through sounddevice (blocking).

        Falls back to pygame.mixer if sounddevice is not available.
        """
        if self._stop_requested.is_set():
            return

        try:
            self._play_via_sounddevice(mp3_bytes)
        except ImportError:
            logger.debug("sounddevice unavailable; trying pygame.")
            try:
                self._play_via_pygame(mp3_bytes)
            except ImportError:
                logger.error(
                    "Neither sounddevice nor pygame is installed. "
                    "Cannot play audio. Install sounddevice: pip install sounddevice"
                )
            except Exception as exc:
                logger.error("pygame playback failed: %s", exc)
        except Exception as exc:
            logger.error("sounddevice playback failed: %s", exc)

    def _play_via_sounddevice(self, mp3_bytes: bytes) -> None:
        """Play MP3 bytes using sounddevice (requires numpy + scipy/pydub)."""
        import sounddevice as sd  # type: ignore[import]

        audio_f32, sample_rate = _mp3_bytes_to_float32(mp3_bytes)

        # Ensure mono float32
        if audio_f32.ndim > 1:
            audio_f32 = audio_f32.mean(axis=1)

        sd.play(audio_f32, samplerate=sample_rate)
        # Poll for stop request while audio is playing
        while sd.get_stream().active:
            if self._stop_requested.is_set():
                sd.stop()
                break
            import time
            time.sleep(0.05)
        sd.wait()

    def _play_via_pygame(self, mp3_bytes: bytes) -> None:
        """Play MP3 bytes using pygame.mixer."""
        import pygame  # type: ignore[import]

        if not pygame.mixer.get_init():
            pygame.mixer.init()

        buf = io.BytesIO(mp3_bytes)
        pygame.mixer.music.load(buf, "mp3")
        pygame.mixer.music.play()

        import time
        while pygame.mixer.music.get_busy():
            if self._stop_requested.is_set():
                pygame.mixer.music.stop()
                break
            time.sleep(0.05)

    async def stop_speaking(self) -> None:
        """Interrupt the current playback immediately."""
        self._stop_requested.set()
        logger.debug("stop_speaking: stop requested.")

    # ------------------------------------------------------------------
    # Voice management
    # ------------------------------------------------------------------

    async def set_voice(self, voice: str) -> None:
        """
        Change the active TTS voice.

        Args:
            voice: Edge TTS voice ShortName, e.g. ``"en-GB-RyanNeural"``.
        """
        self._voice = voice
        logger.info("TextToSpeech: voice changed to %s", voice)

    def list_voices(self) -> list[dict[str, Any]]:
        """
        Return a list of available Edge TTS voices (synchronous).

        Each entry contains keys: name, display_name, gender, locale.

        Note: This method creates a temporary event loop when called from
        synchronous code.  From async contexts use ``await list_voices_async()``.

        Returns:
            List of voice descriptor dicts.
        """
        try:
            return asyncio.run(self.list_voices_async())
        except RuntimeError:
            # Already inside a running loop — cannot use asyncio.run()
            logger.warning(
                "list_voices() called from inside a running event loop. "
                "Use `await list_voices_async()` instead."
            )
            return []

    async def list_voices_async(self) -> list[dict[str, Any]]:
        """
        Return a list of available Edge TTS voices (async).

        Returns:
            List of dicts with keys: name, display_name, gender, locale.
        """
        try:
            import edge_tts  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("edge-tts is not installed.") from exc

        voices = await edge_tts.list_voices()
        return [
            {
                "name": v.get("ShortName", ""),
                "display_name": v.get("FriendlyName", ""),
                "gender": v.get("Gender", ""),
                "locale": v.get("Locale", ""),
            }
            for v in voices
        ]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_tts_instance: TextToSpeech | None = None


def get_tts(
    voice: str = _DEFAULT_VOICE,
    rate: str = _DEFAULT_RATE,
    volume: str = _DEFAULT_VOLUME,
) -> TextToSpeech:
    """
    Return the process-level TextToSpeech singleton.

    Parameters are used only on the first call; subsequent calls return the
    existing instance.
    """
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TextToSpeech(voice=voice, rate=rate, volume=volume)
    return _tts_instance
