"""
AsyncOpenAI client wrapper for the JARVIS desktop assistant.

Features:
- Chat completions (streaming and non-streaming)
- Embeddings
- Whisper audio transcription
- Content moderation
- Cumulative token usage tracking
- Exponential-backoff retry on RateLimitError (3×) and APIConnectionError (2×)
- Process-level singleton via get_ai_client()
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, RateLimitError, APIConnectionError

from config.settings import settings
from core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

_RATE_LIMIT_MAX_RETRIES: int = 3    # total attempts = 1 + 3 retries
_CONN_MAX_RETRIES: int = 2          # total attempts = 1 + 2 retries
_INITIAL_BACKOFF: float = 1.0       # seconds
_BACKOFF_MULTIPLIER: float = 2.0    # doubles each attempt (1s → 2s → 4s)


async def _with_retry(
    coro_factory: Any,
    *,
    rate_limit_retries: int = _RATE_LIMIT_MAX_RETRIES,
    conn_retries: int = _CONN_MAX_RETRIES,
) -> Any:
    """
    Execute an async coroutine factory with per-error-type retry budgets.

    ``coro_factory`` must be a *callable* that returns a fresh coroutine on
    every call (never pass the coroutine object itself — it can only be
    awaited once).

    Retry schedule (backoff resets between error types):
    - RateLimitError:    up to `rate_limit_retries` retries (1s, 2s, 4s …)
    - APIConnectionError: up to `conn_retries` retries (same schedule)
    """
    rate_limit_attempts = 0
    conn_attempts = 0
    backoff = _INITIAL_BACKOFF

    while True:
        try:
            return await coro_factory()

        except RateLimitError as exc:
            rate_limit_attempts += 1
            if rate_limit_attempts > rate_limit_retries:
                logger.error(
                    "OpenAI rate limit exceeded after {} retries.", rate_limit_retries
                )
                raise
            wait = backoff
            logger.warning(
                "RateLimitError (attempt {}/{}). Retrying in {:.1f}s. {}",
                rate_limit_attempts,
                rate_limit_retries,
                wait,
                exc,
            )
            await asyncio.sleep(wait)
            backoff *= _BACKOFF_MULTIPLIER

        except APIConnectionError as exc:
            conn_attempts += 1
            if conn_attempts > conn_retries:
                logger.error(
                    "APIConnectionError after {} retries.", conn_retries
                )
                raise
            wait = backoff
            logger.warning(
                "APIConnectionError (attempt {}/{}). Retrying in {:.1f}s. {}",
                conn_attempts,
                conn_retries,
                wait,
                exc,
            )
            await asyncio.sleep(wait)
            backoff *= _BACKOFF_MULTIPLIER


# ---------------------------------------------------------------------------
# AIClient
# ---------------------------------------------------------------------------


class AIClient:
    """
    Async wrapper around the OpenAI Python SDK tailored for JARVIS.

    All public methods are coroutines (or return async generators for
    streaming operations).  Create one instance per process via the
    ``get_ai_client()`` singleton helper.
    """

    def __init__(self) -> None:
        self._api_key: str = settings.OPENAI_API_KEY
        self._default_model: str = settings.OPENAI_MODEL
        self._default_max_tokens: int = settings.OPENAI_MAX_TOKENS
        self._default_temperature: float = settings.OPENAI_TEMPERATURE

        self._client: AsyncOpenAI = AsyncOpenAI(
            api_key=self._api_key,
            timeout=60.0,
            max_retries=0,  # Retries are handled by _with_retry()
        )

        # Cumulative token counter for the lifetime of this instance
        self._total_tokens_used: int = 0

        logger.info(
            "AIClient initialised (model={}, max_tokens={}).",
            self._default_model,
            self._default_max_tokens,
        )

    # ------------------------------------------------------------------
    # Public property
    # ------------------------------------------------------------------

    @property
    def total_tokens_used(self) -> int:
        """Cumulative total tokens consumed since this instance was created."""
        return self._total_tokens_used

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """
        Call the OpenAI Chat Completions endpoint.

        Args:
            messages:       OpenAI-format message list.
            model:          Model override (defaults to settings.OPENAI_MODEL).
            temperature:    Sampling temperature override.
            max_tokens:     Completion token limit override.
            response_format: e.g. ``{"type": "json_object"}``.
            stream:         If True, returns an AsyncGenerator[str, None]
                            yielding text delta chunks.

        Returns:
            Full response string when ``stream=False``.
            AsyncGenerator[str, None] when ``stream=True``.
        """
        _model = model or self._default_model
        _temperature = temperature if temperature is not None else self._default_temperature
        _max_tokens = max_tokens or self._default_max_tokens

        kwargs: dict[str, Any] = {
            "model": _model,
            "messages": messages,
            "temperature": _temperature,
            "max_tokens": _max_tokens,
            "stream": stream,
        }
        if response_format:
            kwargs["response_format"] = response_format

        if stream:
            return self._stream_chat(kwargs)

        # ── Non-streaming path ────────────────────────────────────────────────
        async def _call() -> Any:
            return await self._client.chat.completions.create(**kwargs)

        response = await _with_retry(_call)

        if response.usage:
            self._total_tokens_used += response.usage.total_tokens or 0

        content: str = response.choices[0].message.content or ""
        logger.debug(
            "chat: model={} prompt_tokens={} completion_tokens={}",
            _model,
            response.usage.prompt_tokens if response.usage else "?",
            response.usage.completion_tokens if response.usage else "?",
        )
        return content

    async def _stream_chat(
        self, kwargs: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """Internal generator — yields partial text chunks from a streaming call."""

        async def _call() -> Any:
            return await self._client.chat.completions.create(**kwargs)

        stream = await _with_retry(_call)
        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue
            delta = choice.delta
            if delta and delta.content:
                yield delta.content

    # ------------------------------------------------------------------
    # embedding
    # ------------------------------------------------------------------

    async def embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small",
    ) -> list[float]:
        """
        Generate an embedding vector for ``text``.

        Args:
            text:  Input string to embed.
            model: Embedding model (default text-embedding-3-small).

        Returns:
            A list of floats representing the embedding vector.
        """
        async def _call() -> Any:
            return await self._client.embeddings.create(
                model=model,
                input=[text],
                encoding_format="float",
            )

        response = await _with_retry(_call)

        if response.usage:
            self._total_tokens_used += response.usage.total_tokens or 0

        vector: list[float] = response.data[0].embedding
        logger.debug("embedding: model={} dim={}.", model, len(vector))
        return vector

    # ------------------------------------------------------------------
    # transcribe_audio
    # ------------------------------------------------------------------

    async def transcribe_audio(self, file_path: str) -> str:
        """
        Transcribe an audio file using OpenAI Whisper.

        Args:
            file_path: Path to the audio file (mp3, wav, m4a, ogg, etc.).

        Returns:
            Transcribed text string.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        async def _call() -> Any:
            with open(path, "rb") as audio_file:
                return await self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                )

        result = await _with_retry(_call)
        transcript: str = result if isinstance(result, str) else getattr(result, "text", str(result))
        logger.debug("transcribe_audio: file={} chars={}.", file_path, len(transcript))
        return transcript

    # ------------------------------------------------------------------
    # moderate
    # ------------------------------------------------------------------

    async def moderate(self, text: str) -> bool:
        """
        Run ``text`` through the OpenAI moderation endpoint.

        Args:
            text: The content to check.

        Returns:
            ``True`` if the content is safe (not flagged).
            ``False`` if the content was flagged as harmful.
        """
        async def _call() -> Any:
            return await self._client.moderations.create(
                model="omni-moderation-latest",
                input=text,
            )

        response = await _with_retry(_call)
        flagged: bool = response.results[0].flagged
        logger.debug("moderate: flagged={}.", flagged)
        return not flagged  # True → safe

    # ------------------------------------------------------------------
    # health_check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """
        Verify connectivity to the OpenAI API with a minimal request.

        Returns:
            ``True`` if the API is reachable and responding, ``False`` otherwise.
        """
        try:
            await self._client.models.list()
            logger.debug("health_check: OpenAI API is reachable.")
            return True
        except Exception as exc:
            logger.warning("health_check: OpenAI API unreachable — {}.", exc)
            return False


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_ai_client_instance: AIClient | None = None


def get_ai_client() -> AIClient:
    """
    Return the process-level AIClient singleton.

    Creates the instance on first call; subsequent calls return the same object.
    Thread-safe under CPython's GIL for the initial assignment.
    """
    global _ai_client_instance
    if _ai_client_instance is None:
        _ai_client_instance = AIClient()
    return _ai_client_instance
