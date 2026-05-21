"""
Text-to-Speech providers for JARVIS.

Providers:
- EdgeTTS: free, Microsoft Edge TTS via edge-tts library (primary)
- ElevenLabsTTS: premium, realistic voice cloning via ElevenLabs API
- TTSManager: provider router with speak / stream helpers

get_tts_manager() returns the process-level singleton.
"""

from __future__ import annotations

import asyncio
import io
import logging
import tempfile
import os
from collections.abc import AsyncGenerator
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EdgeTTS
# ---------------------------------------------------------------------------


class EdgeTTS:
    """
    Microsoft Edge TTS via the edge-tts Python library.

    Free to use; no API key required.  Provides neural-quality voices
    including multiple languages and speaking styles.
    """

    def __init__(self, voice: str = "en-US-AriaNeural") -> None:
        self._voice = voice
        logger.info("EdgeTTS init: voice=%s", voice)

    async def synthesize(self, text: str) -> bytes:
        """
        Synthesise text to MP3 audio bytes.

        Args:
            text: Plain text string to synthesise.

        Returns:
            MP3 audio as raw bytes.
        """
        import edge_tts  # type: ignore[import]

        communicate = edge_tts.Communicate(text=text, voice=self._voice)
        audio_chunks: list[bytes] = []

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        if not audio_chunks:
            raise RuntimeError(
                f"EdgeTTS returned no audio for voice={self._voice!r}."
            )
        return b"".join(audio_chunks)

    async def synthesize_to_file(self, text: str, output_path: str) -> None:
        """Synthesise text and write MP3 to *output_path*."""
        audio_bytes = await self.synthesize(text)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        logger.debug("EdgeTTS written to file: %s", output_path)

    async def stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream audio chunks as they arrive from Edge TTS.

        Yields:
            Individual MP3 audio byte chunks.
        """
        import edge_tts  # type: ignore[import]

        communicate = edge_tts.Communicate(text=text, voice=self._voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    def list_voices(self) -> list[dict[str, Any]]:
        """
        Return a list of available Edge TTS voices.

        Each entry is a dict with keys: Name, ShortName, Gender, Locale.
        This is a synchronous wrapper; use asyncio.run() if calling from sync code.
        """

        async def _fetch() -> list[dict[str, Any]]:
            import edge_tts  # type: ignore[import]
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

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't call asyncio.run() inside a running loop
                # Return a cached or empty list; caller should use
                # asyncio.ensure_future / await inside async context
                logger.warning(
                    "list_voices() called from a running event loop; "
                    "returning empty list. Use `await list_voices_async()` instead."
                )
                return []
            return loop.run_until_complete(_fetch())
        except RuntimeError:
            return asyncio.run(_fetch())

    async def list_voices_async(self) -> list[dict[str, Any]]:
        """Async variant of list_voices()."""
        import edge_tts  # type: ignore[import]

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

    async def synthesize_ssml(self, ssml: str) -> bytes:
        """
        Synthesise Speech Synthesis Markup Language (SSML) string.

        edge-tts does not natively support raw SSML input; this method strips
        tags and synthesises plain text.  For richer SSML support consider
        Azure Cognitive Services TTS.
        """
        import re
        plain_text = re.sub(r"<[^>]+>", " ", ssml).strip()
        plain_text = re.sub(r"\s+", " ", plain_text)
        return await self.synthesize(plain_text)


# ---------------------------------------------------------------------------
# ElevenLabsTTS
# ---------------------------------------------------------------------------


class ElevenLabsTTS:
    """
    Premium Text-to-Speech via the ElevenLabs REST API.

    Supports streaming, voice cloning, and a wide range of high-quality
    neural voices.
    """

    _BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(self, api_key: str, voice_id: str) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }
        logger.info("ElevenLabsTTS init: voice_id=%s", voice_id)

    async def synthesize(
        self,
        text: str,
        model_id: str = "eleven_turbo_v2",
    ) -> bytes:
        """
        Synthesise text and return MP3 audio bytes.

        Args:
            text:     Input text to synthesise.
            model_id: ElevenLabs model identifier.

        Returns:
            MP3 bytes.
        """
        import aiohttp  # type: ignore[import]

        url = f"{self._BASE_URL}/text-to-speech/{self._voice_id}"
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=self._headers,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"ElevenLabs API error {resp.status}: {body}"
                    )
                return await resp.read()

    async def synthesize_stream(
        self,
        text: str,
        model_id: str = "eleven_turbo_v2",
    ) -> AsyncGenerator[bytes, None]:
        """
        Streaming synthesis: yield MP3 chunks as they stream from the API.
        """
        import aiohttp  # type: ignore[import]

        url = f"{self._BASE_URL}/text-to-speech/{self._voice_id}/stream"
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=self._headers,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"ElevenLabs stream API error {resp.status}: {body}"
                    )
                async for chunk in resp.content.iter_chunked(4096):
                    if chunk:
                        yield chunk

    async def clone_voice(
        self,
        audio_samples: list[bytes],
        name: str,
    ) -> str:
        """
        Clone a voice from a list of audio sample byte blobs.

        Args:
            audio_samples: List of audio bytes (WAV or MP3).
            name:          Name for the cloned voice.

        Returns:
            The newly created voice_id string.
        """
        import aiohttp  # type: ignore[import]

        url = f"{self._BASE_URL}/voices/add"
        headers = {"xi-api-key": self._api_key}

        form = aiohttp.FormData()
        form.add_field("name", name)
        for i, sample in enumerate(audio_samples):
            form.add_field(
                "files",
                io.BytesIO(sample),
                filename=f"sample_{i}.wav",
                content_type="audio/wav",
            )

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form, headers=headers) as resp:
                if resp.status not in (200, 201):
                    body = await resp.text()
                    raise RuntimeError(
                        f"ElevenLabs clone API error {resp.status}: {body}"
                    )
                data = await resp.json()
                voice_id: str = data["voice_id"]
                logger.info("Cloned voice '%s' -> voice_id=%s", name, voice_id)
                return voice_id

    def list_voices(self) -> list[dict[str, Any]]:
        """
        Synchronously return available voices.

        Uses a fresh event loop; do not call from inside an async context.
        """
        return asyncio.run(self.list_voices_async())

    async def list_voices_async(self) -> list[dict[str, Any]]:
        """Async: fetch and return the list of available ElevenLabs voices."""
        import aiohttp  # type: ignore[import]

        url = f"{self._BASE_URL}/voices"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self._headers) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"ElevenLabs voices API error {resp.status}")
                data = await resp.json()
                return [
                    {
                        "voice_id": v["voice_id"],
                        "name": v["name"],
                        "category": v.get("category", ""),
                        "description": v.get("description", ""),
                    }
                    for v in data.get("voices", [])
                ]


# ---------------------------------------------------------------------------
# TTSManager
# ---------------------------------------------------------------------------


class TTSManager:
    """
    Provider-agnostic TTS router.

    Routes synthesise calls to EdgeTTS (default) or ElevenLabsTTS based on
    the *provider* argument and available credentials.
    """

    def __init__(self, provider: str = "edge") -> None:
        self._provider_name = provider
        self._edge: EdgeTTS | None = None
        self._elevenlabs: ElevenLabsTTS | None = None
        self._init_providers()

    def _init_providers(self) -> None:
        from backend.app.core.config import get_settings
        settings = get_settings()

        self._edge = EdgeTTS(voice="en-US-AriaNeural")

        if settings.ELEVENLABS_API_KEY:
            self._elevenlabs = ElevenLabsTTS(
                api_key=settings.ELEVENLABS_API_KEY,
                voice_id=settings.ELEVENLABS_VOICE_ID,
            )
        else:
            logger.info("ElevenLabs API key not set; premium TTS unavailable.")

    def _get_provider(self) -> EdgeTTS | ElevenLabsTTS:
        """Return the active provider instance."""
        if self._provider_name == "elevenlabs" and self._elevenlabs:
            return self._elevenlabs
        if self._edge is None:
            raise RuntimeError("No TTS provider available.")
        return self._edge

    async def speak(self, text: str) -> bytes:
        """
        Synthesise *text* and return audio bytes.

        Routes to EdgeTTS or ElevenLabsTTS depending on provider setting.
        """
        provider = self._get_provider()
        try:
            return await provider.synthesize(text)  # type: ignore[return-value]
        except Exception as exc:
            # Fall back to EdgeTTS on provider failure
            if provider is not self._edge and self._edge:
                logger.warning(
                    "Primary TTS provider failed (%s), falling back to Edge TTS.",
                    exc,
                )
                return await self._edge.synthesize(text)
            raise

    async def speak_and_play(self, text: str) -> None:
        """
        Synthesise *text* and play audio immediately via sounddevice.

        Requires the ``sounddevice`` and ``scipy`` packages.
        """
        audio_bytes = await self.speak(text)
        await asyncio.get_running_loop().run_in_executor(
            None, self._play_audio, audio_bytes
        )

    def _play_audio(self, audio_bytes: bytes) -> None:
        """Play MP3 audio bytes through the system default output device."""
        try:
            import sounddevice as sd  # type: ignore[import]
            import numpy as np
            import scipy.io.wavfile as wav_io  # type: ignore[import]

            # Decode MP3 to WAV using pydub if available
            try:
                from pydub import AudioSegment  # type: ignore[import]

                seg = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                seg = seg.set_frame_rate(44100).set_channels(1)
                wav_buf = io.BytesIO()
                seg.export(wav_buf, format="wav")
                wav_buf.seek(0)
                rate, data = wav_io.read(wav_buf)
            except ImportError:
                # Assume bytes are already WAV
                rate, data = wav_io.read(io.BytesIO(audio_bytes))

            if data.dtype != np.float32:
                data = data.astype(np.float32) / 32768.0

            sd.play(data, samplerate=rate)
            sd.wait()
        except ImportError as exc:
            logger.error("sounddevice or scipy not installed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Audio playback failed: %s", exc)

    async def speak_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Streaming synthesis: yield audio byte chunks as they are generated.
        """
        provider = self._get_provider()
        if isinstance(provider, EdgeTTS):
            async for chunk in provider.stream(text):
                yield chunk
        elif isinstance(provider, ElevenLabsTTS):
            async for chunk in provider.synthesize_stream(text):
                yield chunk
        else:
            # Fallback: synthesise entirely, then yield in one chunk
            audio_bytes = await self.speak(text)
            yield audio_bytes


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_tts_manager_instance: TTSManager | None = None


def get_tts_manager() -> TTSManager:
    """
    Return the process-level TTSManager singleton.

    The provider is determined by the ELEVENLABS_API_KEY setting:
    - If set: "elevenlabs"
    - Otherwise: "edge"
    """
    global _tts_manager_instance
    if _tts_manager_instance is None:
        from backend.app.core.config import get_settings
        settings = get_settings()
        provider = "elevenlabs" if settings.ELEVENLABS_API_KEY else "edge"
        _tts_manager_instance = TTSManager(provider=provider)
    return _tts_manager_instance
