"""
Audio processing utilities for the JARVIS voice pipeline.

Provides:
- AudioProcessor: static/class methods for format conversion, normalisation,
  resampling, silence trimming, splitting, merging, and numpy interop.
"""

from __future__ import annotations

import io
import logging
import struct
import wave
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SAMPLE_RATE: int = 16000
_DEFAULT_SAMPLE_WIDTH: int = 2       # 16-bit
_DEFAULT_CHANNELS: int = 1           # mono


# ---------------------------------------------------------------------------
# AudioProcessor
# ---------------------------------------------------------------------------


class AudioProcessor:
    """
    Utility class for audio manipulation.

    All methods are stateless and operate on in-memory byte buffers.
    Input is expected to be WAV bytes unless stated otherwise.
    """

    # ------------------------------------------------------------------
    # Format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def convert_to_wav(audio_bytes: bytes, source_format: str) -> bytes:
        """
        Convert audio bytes from *source_format* to 16-bit mono WAV at
        the source file's native sample rate.

        Supported source formats (requires pydub + ffmpeg):
            "mp3", "ogg", "flac", "m4a", "webm", "opus"

        Falls back to returning audio_bytes unchanged if pydub is not
        available.

        Args:
            audio_bytes:   Raw bytes of the source audio.
            source_format: Lowercase format name without the leading dot.

        Returns:
            WAV bytes.
        """
        source_format = source_format.lstrip(".").lower()

        if source_format == "wav":
            return audio_bytes

        try:
            from pydub import AudioSegment  # type: ignore[import]

            buf = io.BytesIO(audio_bytes)
            segment = AudioSegment.from_file(buf, format=source_format)
            segment = segment.set_channels(1).set_sample_width(2)

            out = io.BytesIO()
            segment.export(out, format="wav")
            return out.getvalue()

        except ImportError:
            logger.warning(
                "pydub not installed; cannot convert %s to WAV. "
                "Returning bytes unchanged.",
                source_format,
            )
            return audio_bytes
        except Exception as exc:  # noqa: BLE001
            logger.error("convert_to_wav failed for format=%s: %s", source_format, exc)
            raise

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_audio(audio_bytes: bytes) -> bytes:
        """
        Normalise audio amplitude so the peak level reaches near 0 dBFS.

        Args:
            audio_bytes: WAV bytes.

        Returns:
            Normalised WAV bytes.
        """
        arr = AudioProcessor.bytes_to_numpy(audio_bytes)
        peak = np.max(np.abs(arr))
        if peak == 0:
            return audio_bytes

        target_peak = 32767.0 * 0.95
        arr = (arr * (target_peak / peak)).astype(np.int16)

        # Recover sample rate from original WAV
        try:
            with wave.open(io.BytesIO(audio_bytes)) as wf:
                rate = wf.getframerate()
        except Exception:  # noqa: BLE001
            rate = _DEFAULT_SAMPLE_RATE

        return AudioProcessor.numpy_to_bytes(arr, sample_rate=rate)

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    @staticmethod
    def resample(audio_bytes: bytes, target_rate: int) -> bytes:
        """
        Resample WAV audio to *target_rate* Hz.

        Requires scipy.  If unavailable, returns audio unchanged.

        Args:
            audio_bytes: WAV bytes at any sample rate.
            target_rate: Desired output sample rate in Hz.

        Returns:
            WAV bytes at *target_rate* Hz.
        """
        try:
            import scipy.io.wavfile as wav_io  # type: ignore[import]
            import scipy.signal as signal  # type: ignore[import]
        except ImportError:
            logger.warning("scipy not installed; skipping resample.")
            return audio_bytes

        buf = io.BytesIO(audio_bytes)
        rate, data = wav_io.read(buf)

        if rate == target_rate:
            return audio_bytes

        if data.ndim > 1:
            data = data.mean(axis=1)

        num_samples = int(len(data) * target_rate / rate)
        resampled = signal.resample(data, num_samples).astype(np.int16)

        out = io.BytesIO()
        wav_io.write(out, target_rate, resampled)
        return out.getvalue()

    # ------------------------------------------------------------------
    # Silence trimming
    # ------------------------------------------------------------------

    @staticmethod
    def trim_silence(audio_bytes: bytes, threshold: int = 500) -> bytes:
        """
        Remove leading and trailing silence from WAV audio.

        Args:
            audio_bytes: WAV bytes.
            threshold:   Amplitude below which a sample is considered silent
                         (0–32767 scale for 16-bit audio).

        Returns:
            WAV bytes with leading/trailing silence removed.
        """
        arr = AudioProcessor.bytes_to_numpy(audio_bytes)
        mask = np.abs(arr) > threshold
        indices = np.where(mask)[0]

        if len(indices) == 0:
            return audio_bytes

        start, end = int(indices[0]), int(indices[-1]) + 1
        trimmed = arr[start:end]

        try:
            with wave.open(io.BytesIO(audio_bytes)) as wf:
                rate = wf.getframerate()
        except Exception:  # noqa: BLE001
            rate = _DEFAULT_SAMPLE_RATE

        return AudioProcessor.numpy_to_bytes(trimmed, sample_rate=rate)

    # ------------------------------------------------------------------
    # Duration
    # ------------------------------------------------------------------

    @staticmethod
    def get_duration(audio_bytes: bytes) -> float:
        """
        Return the duration of WAV audio in seconds.

        Args:
            audio_bytes: WAV bytes.

        Returns:
            Duration in fractional seconds.
        """
        try:
            with wave.open(io.BytesIO(audio_bytes)) as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_duration failed: %s", exc)
            # Estimate from raw byte size assuming 16-bit 16kHz mono
            return len(audio_bytes) / (16000 * 2)

    # ------------------------------------------------------------------
    # Silence splitting
    # ------------------------------------------------------------------

    @staticmethod
    def split_on_silence(
        audio_bytes: bytes,
        min_silence_len: int = 500,
        threshold: int = 500,
    ) -> list[bytes]:
        """
        Split audio on silent intervals.

        Args:
            audio_bytes:      WAV bytes.
            min_silence_len:  Minimum silence duration in milliseconds.
            threshold:        Amplitude threshold for silence detection.

        Returns:
            List of WAV byte segments (non-silent chunks).
        """
        arr = AudioProcessor.bytes_to_numpy(audio_bytes)

        try:
            with wave.open(io.BytesIO(audio_bytes)) as wf:
                rate = wf.getframerate()
        except Exception:  # noqa: BLE001
            rate = _DEFAULT_SAMPLE_RATE

        silence_samples = int(min_silence_len * rate / 1000)
        above_thresh = (np.abs(arr) > threshold).astype(np.int8)

        segments: list[bytes] = []
        in_speech = False
        start = 0
        silent_count = 0

        for i, val in enumerate(above_thresh):
            if val:
                if not in_speech:
                    start = max(0, i - silence_samples // 4)
                    in_speech = True
                silent_count = 0
            else:
                if in_speech:
                    silent_count += 1
                    if silent_count >= silence_samples:
                        end = i - silence_samples + silence_samples // 4
                        chunk = arr[start:end]
                        if len(chunk) > 0:
                            segments.append(
                                AudioProcessor.numpy_to_bytes(chunk, rate)
                            )
                        in_speech = False
                        silent_count = 0

        # Flush trailing segment
        if in_speech:
            chunk = arr[start:]
            if len(chunk) > 0:
                segments.append(AudioProcessor.numpy_to_bytes(chunk, rate))

        return segments if segments else [audio_bytes]

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    @staticmethod
    def merge_audio(audio_segments: list[bytes]) -> bytes:
        """
        Concatenate a list of WAV byte segments into a single WAV.

        All segments must share the same sample rate and bit depth.
        The output uses the parameters from the first segment.

        Args:
            audio_segments: List of WAV bytes to merge.

        Returns:
            Merged WAV bytes.
        """
        if not audio_segments:
            raise ValueError("audio_segments list is empty.")

        if len(audio_segments) == 1:
            return audio_segments[0]

        # Read parameters from first segment
        try:
            with wave.open(io.BytesIO(audio_segments[0])) as wf:
                params = wf.getparams()
                rate = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
        except Exception as exc:  # noqa: BLE001
            logger.error("merge_audio: could not read first segment: %s", exc)
            raise

        out_buf = io.BytesIO()
        with wave.open(out_buf, "wb") as out_wf:
            out_wf.setnchannels(channels)
            out_wf.setsampwidth(sampwidth)
            out_wf.setframerate(rate)
            for seg in audio_segments:
                try:
                    with wave.open(io.BytesIO(seg)) as wf:
                        out_wf.writeframes(wf.readframes(wf.getnframes()))
                except Exception:  # noqa: BLE001
                    # Treat as raw PCM if WAV parsing fails
                    out_wf.writeframes(seg)

        return out_buf.getvalue()

    # ------------------------------------------------------------------
    # NumPy interop
    # ------------------------------------------------------------------

    @staticmethod
    def bytes_to_numpy(audio_bytes: bytes) -> np.ndarray:
        """
        Convert WAV bytes to a 1-D int16 NumPy array.

        Args:
            audio_bytes: WAV bytes.

        Returns:
            Flat int16 array of audio samples.
        """
        try:
            with wave.open(io.BytesIO(audio_bytes)) as wf:
                raw = wf.readframes(wf.getnframes())
                data = np.frombuffer(raw, dtype=np.int16).copy()
                # If stereo, mix down to mono
                if wf.getnchannels() == 2:
                    data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)
                return data
        except Exception:  # noqa: BLE001
            # Fallback: assume raw 16-bit PCM
            return np.frombuffer(audio_bytes, dtype=np.int16).copy()

    @staticmethod
    def numpy_to_bytes(array: np.ndarray, sample_rate: int) -> bytes:
        """
        Convert a NumPy array to mono 16-bit WAV bytes.

        Args:
            array:       1-D numeric array (will be cast to int16).
            sample_rate: Sample rate in Hz.

        Returns:
            WAV bytes.
        """
        data = np.clip(array, -32768, 32767).astype(np.int16)
        out = io.BytesIO()
        with wave.open(out, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(data.tobytes())
        return out.getvalue()
