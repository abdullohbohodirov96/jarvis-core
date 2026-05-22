"""
Wake word detector for JARVIS desktop AI assistant.

Uses Whisper (tiny model) to transcribe 2-second audio windows captured from
the microphone in a background thread and signals when the configured wake
word is detected in the transcript.

No external wake-word SDK required — purely Whisper-based for MVP.
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
import threading
import wave
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHUNK_DURATION_S: float = 2.0      # audio window fed to Whisper
_OVERLAP_DURATION_S: float = 0.5   # overlap between consecutive windows
_DEFAULT_SAMPLE_RATE: int = 16_000
_DEFAULT_CHANNELS: int = 1
_DEFAULT_DTYPE: str = "int16"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = _DEFAULT_SAMPLE_RATE) -> bytes:
    """Convert a 1-D int16 NumPy array to in-memory WAV bytes."""
    data = np.clip(audio, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# WakeWordDetector
# ---------------------------------------------------------------------------


class WakeWordDetector:
    """
    Listens to the default microphone and detects the configured wake word
    using repeated Whisper transcriptions of 2-second audio windows.

    The detector runs in a daemon thread so it does not block the asyncio
    event loop.  The asyncio-facing side exposes ``wait_for_wake_word()``
    which suspends the caller until detection (or cancellation).

    Attributes:
        on_detected: Optional synchronous callback invoked on each detection.
    """

    def __init__(
        self,
        wake_word: str = "jarvis",
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
    ) -> None:
        self._wake_word: str = wake_word.lower().strip()
        self._sample_rate: int = sample_rate

        # State
        self._listening: bool = False
        self._detection_count: int = 0
        self._stop_event: threading.Event = threading.Event()
        self._detected_event: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Asyncio bridge — set by start() so callbacks can signal the loop
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_detected_event: Optional[asyncio.Event] = None

        # Optional callback invoked each time the wake word is detected
        self.on_detected: Optional[Callable[[], None]] = None

        # Lazy-loaded Whisper model for tiny inference
        self._whisper_model = None
        self._model_lock = threading.Lock()

        logger.info(
            "WakeWordDetector created: wake_word=%r sample_rate=%d",
            self._wake_word,
            self._sample_rate,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_listening(self) -> bool:
        """True while the background listener thread is running."""
        return self._listening

    @property
    def detection_count(self) -> int:
        """Number of wake-word detections since start()."""
        return self._detection_count

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start background microphone monitoring in an executor thread."""
        if self._listening:
            logger.debug("WakeWordDetector.start() called but already listening.")
            return

        self._loop = asyncio.get_running_loop()
        self._async_detected_event = asyncio.Event()
        self._stop_event.clear()
        self._detected_event.clear()
        self._listening = True

        self._thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="jarvis-wake-word",
        )
        self._thread.start()
        logger.info("WakeWordDetector started; listening for %r.", self._wake_word)

    async def stop(self) -> None:
        """Stop the background listener thread and release microphone."""
        if not self._listening:
            return

        logger.info("WakeWordDetector stopping…")
        self._stop_event.set()
        self._listening = False

        if self._thread is not None and self._thread.is_alive():
            # Run the join in the executor so we don't block the event loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._thread.join, 5.0)
            self._thread = None

        logger.info("WakeWordDetector stopped.")

    # ------------------------------------------------------------------
    # Asyncio-facing API
    # ------------------------------------------------------------------

    async def wait_for_wake_word(self) -> bool:
        """
        Suspend the caller until the wake word is detected.

        Returns True immediately when detection fires.  If the detector is
        not started yet, it is started automatically.

        The caller should await ``stop()`` when done with the detector.
        """
        if not self._listening:
            await self.start()

        assert self._async_detected_event is not None
        self._async_detected_event.clear()
        await self._async_detected_event.wait()
        return True

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the Whisper tiny model (blocking, thread-safe)."""
        with self._model_lock:
            if self._whisper_model is None:
                try:
                    import whisper  # type: ignore[import]
                    logger.info("WakeWordDetector: loading Whisper tiny model…")
                    self._whisper_model = whisper.load_model("tiny")
                    logger.info("WakeWordDetector: Whisper tiny model loaded.")
                except Exception as exc:
                    logger.error("WakeWordDetector: failed to load Whisper model: %s", exc)
                    raise
        return self._whisper_model

    def _transcribe_chunk(self, audio: np.ndarray) -> str:
        """Run Whisper on a numpy float32 audio array; return transcript text."""
        try:
            import whisper  # type: ignore[import]
            model = self._load_model()
            # Whisper expects float32 normalised to [-1, 1]
            audio_f32 = audio.astype(np.float32) / 32768.0
            # Whisper transcribe accepts a numpy array directly
            result = model.transcribe(
                audio_f32,
                language="en",
                fp16=False,
                verbose=False,
                condition_on_previous_text=False,
            )
            text: str = result.get("text", "").strip()
            return text
        except Exception as exc:
            logger.debug("WakeWordDetector: transcription error: %s", exc)
            return ""

    def _listen_loop(self) -> None:
        """
        Main loop running in the background daemon thread.

        Opens a sounddevice InputStream, accumulates audio in rolling 2-second
        windows (with 0.5 s overlap), runs Whisper on each window, and signals
        the asyncio event loop whenever the wake word is found.
        """
        try:
            import sounddevice as sd  # type: ignore[import]
        except ImportError:
            logger.warning(
                "sounddevice not installed; WakeWordDetector cannot open microphone."
            )
            self._listening = False
            return

        chunk_samples = int(_CHUNK_DURATION_S * self._sample_rate)
        overlap_samples = int(_OVERLAP_DURATION_S * self._sample_rate)
        read_samples = chunk_samples - overlap_samples  # new samples per iteration

        ring_buffer: list[np.ndarray] = []   # accumulated sample arrays

        try:
            stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=_DEFAULT_CHANNELS,
                dtype=_DEFAULT_DTYPE,
                blocksize=read_samples,
            )
            stream.start()
            logger.debug("WakeWordDetector: InputStream opened.")
        except Exception as exc:
            logger.warning(
                "WakeWordDetector: could not open microphone: %s", exc
            )
            self._listening = False
            return

        try:
            while not self._stop_event.is_set():
                try:
                    data, overflowed = stream.read(read_samples)
                    if overflowed:
                        logger.debug("WakeWordDetector: audio buffer overflowed.")
                    chunk: np.ndarray = data[:, 0] if data.ndim > 1 else data.flatten()
                    ring_buffer.append(chunk)

                    # Keep only enough samples for one window
                    total_samples = sum(c.shape[0] for c in ring_buffer)
                    while total_samples > chunk_samples and len(ring_buffer) > 1:
                        removed = ring_buffer.pop(0)
                        total_samples -= removed.shape[0]

                    if total_samples < chunk_samples:
                        # Not enough audio yet
                        continue

                    window = np.concatenate(ring_buffer)[-chunk_samples:]
                    transcript = self._transcribe_chunk(window)

                    if transcript:
                        logger.debug("WakeWordDetector transcript: %r", transcript)

                    if self._wake_word in transcript.lower():
                        self._detection_count += 1
                        logger.info(
                            "Wake word %r detected (count=%d). Transcript: %r",
                            self._wake_word,
                            self._detection_count,
                            transcript,
                        )
                        self._detected_event.set()

                        # Signal asyncio event safely from this thread
                        if self._loop is not None and self._async_detected_event is not None:
                            self._loop.call_soon_threadsafe(
                                self._async_detected_event.set
                            )

                        # Fire synchronous callback if registered
                        if self.on_detected is not None:
                            try:
                                self.on_detected()
                            except Exception as cb_exc:
                                logger.warning(
                                    "WakeWordDetector: on_detected callback raised: %s",
                                    cb_exc,
                                )

                        # Clear the ring buffer so a fresh window starts
                        ring_buffer.clear()

                except Exception as exc:
                    logger.warning("WakeWordDetector: listen loop error: %s", exc)
                    # Brief pause to avoid tight error loops
                    self._stop_event.wait(timeout=0.1)

        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            logger.debug("WakeWordDetector: InputStream closed.")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_detector_instance: Optional[WakeWordDetector] = None
_detector_lock = threading.Lock()


def get_wake_word_detector(
    wake_word: str = "jarvis",
    sample_rate: int = _DEFAULT_SAMPLE_RATE,
) -> WakeWordDetector:
    """
    Return the process-level WakeWordDetector singleton.

    On first call the instance is created with the provided parameters;
    subsequent calls return the existing instance regardless of parameters.
    """
    global _detector_instance
    with _detector_lock:
        if _detector_instance is None:
            _detector_instance = WakeWordDetector(
                wake_word=wake_word,
                sample_rate=sample_rate,
            )
    return _detector_instance
