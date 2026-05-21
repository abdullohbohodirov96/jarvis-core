"""
Real-time microphone listening for JARVIS voice input.

Provides:
- VoiceActivityDetector: energy-based speech detection
- MicrophoneListener:    async recording with silence detection and wake-word
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, AsyncGenerator
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SAMPLE_RATE: int = 16000
_DEFAULT_CHANNELS: int = 1
_DEFAULT_DTYPE: str = "int16"


# ---------------------------------------------------------------------------
# VoiceActivityDetector
# ---------------------------------------------------------------------------


class VoiceActivityDetector:
    """
    Simple energy-based Voice Activity Detector (VAD).

    Uses RMS energy threshold to classify audio chunks as speech or silence.
    For production use, consider replacing with webrtcvad or silero-vad.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Args:
            threshold: Normalised RMS threshold (0.0–1.0) above which a
                       chunk is classified as speech.  0.5 ≈ roughly moderate
                       background noise tolerance.
        """
        # Convert normalised threshold to raw int16 amplitude
        self._raw_threshold: float = threshold * 32767.0

    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        Return True if the energy in *audio_chunk* exceeds the threshold.

        Args:
            audio_chunk: Raw 16-bit PCM bytes.
        """
        if not audio_chunk:
            return False
        arr = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(arr ** 2)))
        return rms > self._raw_threshold

    def detect_endpoints(
        self,
        audio_stream: list[bytes],
    ) -> list[tuple[int, int]]:
        """
        Return (start_frame, end_frame) pairs for speech segments.

        Args:
            audio_stream: Ordered list of fixed-size audio byte chunks.

        Returns:
            List of (start_index, end_index) tuples, one per speech segment.
        """
        endpoints: list[tuple[int, int]] = []
        in_speech = False
        start_idx = 0

        for i, chunk in enumerate(audio_stream):
            is_sp = self.is_speech(chunk)
            if is_sp and not in_speech:
                start_idx = i
                in_speech = True
            elif not is_sp and in_speech:
                endpoints.append((start_idx, i - 1))
                in_speech = False

        if in_speech:
            endpoints.append((start_idx, len(audio_stream) - 1))

        return endpoints


# ---------------------------------------------------------------------------
# MicrophoneListener
# ---------------------------------------------------------------------------


class MicrophoneListener:
    """
    Async microphone recording with silence-based endpoint detection.

    Uses sounddevice for cross-platform audio capture.  All recording runs
    in a dedicated thread pool executor to avoid blocking the event loop.
    """

    def __init__(
        self,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        chunk_duration: float = 0.5,
        silence_threshold: int = 500,
    ) -> None:
        """
        Args:
            sample_rate:        PCM sample rate in Hz.
            chunk_duration:     Duration in seconds of each audio chunk read.
            silence_threshold:  Raw int16 amplitude below which a frame is
                                considered silent (0–32767).
        """
        self._sample_rate = sample_rate
        self._chunk_duration = chunk_duration
        self._silence_threshold = silence_threshold
        self._chunk_samples = int(sample_rate * chunk_duration)

        self._is_listening = False
        self._stream: Any = None  # sounddevice InputStream

        self._vad = VoiceActivityDetector(
            threshold=silence_threshold / 32767.0
        )

        logger.info(
            "MicrophoneListener init: rate=%d chunk=%.2fs threshold=%d",
            sample_rate,
            chunk_duration,
            silence_threshold,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_listening(self) -> None:
        """Open the sounddevice input stream."""
        if self._is_listening:
            return
        try:
            import sounddevice as sd  # type: ignore[import]

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._open_stream)
            self._is_listening = True
            logger.info("MicrophoneListener: stream opened.")
        except ImportError:
            logger.error("sounddevice not installed; microphone unavailable.")
            raise

    def _open_stream(self) -> None:
        import sounddevice as sd  # type: ignore[import]

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=_DEFAULT_CHANNELS,
            dtype=_DEFAULT_DTYPE,
            blocksize=self._chunk_samples,
        )
        self._stream.start()

    async def stop_listening(self) -> None:
        """Close the sounddevice input stream."""
        if not self._is_listening:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._close_stream)
        self._is_listening = False
        logger.info("MicrophoneListener: stream closed.")

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing audio stream: %s", exc)
            self._stream = None

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def _read_chunk(self) -> bytes:
        """Read one chunk from the open stream (blocking)."""
        if self._stream is None:
            raise RuntimeError("Stream not open. Call start_listening() first.")
        data, _ = self._stream.read(self._chunk_samples)
        return data.tobytes()

    def _is_silent(self, audio_chunk: bytes) -> bool:
        """Return True if the chunk is below the silence threshold."""
        arr = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        return float(np.max(np.abs(arr))) < self._silence_threshold

    def _detect_speech_start(self, buffer: list[bytes]) -> bool:
        """Return True if any chunk in *buffer* contains speech."""
        return any(not self._is_silent(c) for c in buffer)

    def _record_until_silence(self, timeout: float = 10.0) -> bytes:
        """
        Record audio until silence is detected or *timeout* elapses.

        Blocks the calling thread — must be called via run_in_executor.

        Returns:
            Raw 16-bit PCM bytes of the recorded audio.
        """
        frames: list[bytes] = []
        silent_chunks = 0
        # Number of silent chunks to declare end-of-speech
        # (~1.0 s of silence)
        max_silent = int(1.0 / self._chunk_duration)

        deadline = time.monotonic() + timeout
        speech_started = False

        while time.monotonic() < deadline:
            chunk = self._read_chunk()
            frames.append(chunk)

            if not self._is_silent(chunk):
                speech_started = True
                silent_chunks = 0
            else:
                if speech_started:
                    silent_chunks += 1
                    if silent_chunks >= max_silent:
                        break

        return b"".join(frames)

    # ------------------------------------------------------------------
    # Public recording API
    # ------------------------------------------------------------------

    async def listen_for_command(
        self,
        timeout: float = 10.0,
    ) -> bytes | None:
        """
        Record a single voice command.

        Opens the stream (if not already open), waits for speech to begin,
        records until silence, then closes the stream.

        Args:
            timeout: Maximum recording time in seconds.

        Returns:
            Raw 16-bit PCM bytes, or None if no speech was detected.
        """
        opened = not self._is_listening
        if opened:
            await self.start_listening()

        try:
            loop = asyncio.get_running_loop()
            audio = await asyncio.wait_for(
                loop.run_in_executor(
                    None, self._record_until_silence, timeout
                ),
                timeout=timeout + 2.0,
            )
            if not audio:
                return None
            # Convert raw PCM to WAV bytes
            from backend.app.voice.audio_processor import AudioProcessor
            return AudioProcessor.numpy_to_bytes(
                np.frombuffer(audio, dtype=np.int16),
                self._sample_rate,
            )
        except asyncio.TimeoutError:
            logger.warning("listen_for_command timed out.")
            return None
        finally:
            if opened:
                await self.stop_listening()

    async def continuous_listen(
        self,
        callback: Callable[[bytes], Any],
    ) -> None:
        """
        Continuously listen and invoke *callback* with each detected utterance.

        Runs until stop_listening() is called.

        Args:
            callback: Async or sync callable that receives WAV audio bytes.
        """
        await self.start_listening()
        logger.info("MicrophoneListener: continuous listening started.")

        try:
            while self._is_listening:
                try:
                    loop = asyncio.get_running_loop()
                    audio = await loop.run_in_executor(
                        None, self._record_until_silence, 10.0
                    )
                    if audio and not self._is_silent(audio[:self._chunk_samples * 2]):
                        from backend.app.voice.audio_processor import AudioProcessor
                        wav_bytes = AudioProcessor.numpy_to_bytes(
                            np.frombuffer(audio, dtype=np.int16),
                            self._sample_rate,
                        )
                        if asyncio.iscoroutinefunction(callback):
                            await callback(wav_bytes)
                        else:
                            callback(wav_bytes)
                except Exception as exc:  # noqa: BLE001
                    logger.error("continuous_listen error: %s", exc)
                    await asyncio.sleep(0.1)
        finally:
            await self.stop_listening()
            logger.info("MicrophoneListener: continuous listening stopped.")

    async def wait_for_wake_word(
        self,
        wake_words: list[str] | None = None,
    ) -> bool:
        """
        Listen continuously until one of *wake_words* is spoken.

        Uses the WhisperSTT engine for detection.  Returns True once a wake
        word is found, False on timeout (60 s default).

        Args:
            wake_words: List of trigger phrases; defaults to ["jarvis",
                        "hey jarvis"].
        """
        if wake_words is None:
            wake_words = ["jarvis", "hey jarvis"]

        wake_words_lower = [w.lower() for w in wake_words]

        from backend.app.voice.stt import get_stt_engine

        stt = get_stt_engine()
        deadline = time.monotonic() + 60.0

        await self.start_listening()
        logger.info(
            "Waiting for wake word(s): %s",
            wake_words,
        )

        try:
            while time.monotonic() < deadline:
                try:
                    loop = asyncio.get_running_loop()
                    chunk = await loop.run_in_executor(None, self._read_chunk)

                    if self._is_silent(chunk):
                        continue

                    # Collect ~2 second window after non-silent frame
                    frames = [chunk]
                    window_end = time.monotonic() + 2.0
                    while time.monotonic() < window_end:
                        frames.append(
                            await loop.run_in_executor(None, self._read_chunk)
                        )

                    audio_data = b"".join(frames)
                    from backend.app.voice.audio_processor import AudioProcessor
                    wav_bytes = AudioProcessor.numpy_to_bytes(
                        np.frombuffer(audio_data, dtype=np.int16),
                        self._sample_rate,
                    )

                    result = await stt.transcribe_bytes(wav_bytes)
                    transcript_lower = result.text.lower().strip()

                    if any(w in transcript_lower for w in wake_words_lower):
                        logger.info(
                            "Wake word detected: '%s'", result.text
                        )
                        return True

                except Exception as exc:  # noqa: BLE001
                    logger.warning("Wake word detection error: %s", exc)
                    await asyncio.sleep(0.1)
        finally:
            await self.stop_listening()

        logger.info("Wake word timeout reached.")
        return False
