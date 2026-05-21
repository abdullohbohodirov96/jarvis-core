"""
OpenAI async client wrapper for JARVIS.

Provides a singleton OpenAIClient with:
- Chat completions (streaming and non-streaming)
- Embeddings
- Whisper audio transcription
- Content moderation
- Token counting
- Retry logic with exponential backoff
- Cost / usage tracking
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import tiktoken
from openai import AsyncOpenAI, RateLimitError, APIStatusError, APIConnectionError
from openai.types.chat import ChatCompletionMessageParam

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost table (USD per 1 000 tokens) – update as OpenAI pricing changes
# ---------------------------------------------------------------------------

_COST_TABLE: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
    "whisper-1": {"input": 0.006, "output": 0.0},   # per minute, approximated per 1k tok
}

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

_MAX_RETRIES: int = 5
_INITIAL_BACKOFF: float = 1.0   # seconds
_BACKOFF_MULTIPLIER: float = 2.0
_MAX_BACKOFF: float = 60.0


async def _with_retry(coro_factory: Any, max_retries: int = _MAX_RETRIES) -> Any:
    """
    Execute an async coroutine factory with exponential backoff on rate limit
    errors.  ``coro_factory`` must be a *callable* that returns a fresh
    coroutine each time (not the coroutine object itself, which can only be
    awaited once).
    """
    backoff = _INITIAL_BACKOFF
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except RateLimitError as exc:
            if attempt == max_retries:
                logger.error("OpenAI rate limit exceeded after %d retries.", max_retries)
                raise
            wait = min(backoff, _MAX_BACKOFF)
            logger.warning(
                "OpenAI rate limit hit (attempt %d/%d). Retrying in %.1fs. %s",
                attempt + 1,
                max_retries,
                wait,
                exc,
            )
            await asyncio.sleep(wait)
            backoff *= _BACKOFF_MULTIPLIER
        except APIConnectionError as exc:
            if attempt == max_retries:
                raise
            wait = min(backoff, _MAX_BACKOFF)
            logger.warning(
                "OpenAI connection error (attempt %d/%d). Retrying in %.1fs. %s",
                attempt + 1,
                max_retries,
                wait,
                exc,
            )
            await asyncio.sleep(wait)
            backoff *= _BACKOFF_MULTIPLIER
        except APIStatusError as exc:
            # 5xx errors are retryable; 4xx (except 429) are not
            if exc.status_code >= 500:
                if attempt == max_retries:
                    raise
                wait = min(backoff, _MAX_BACKOFF)
                logger.warning(
                    "OpenAI server error %d (attempt %d/%d). Retrying in %.1fs.",
                    exc.status_code,
                    attempt + 1,
                    max_retries,
                    wait,
                )
                await asyncio.sleep(wait)
                backoff *= _BACKOFF_MULTIPLIER
            else:
                raise


# ---------------------------------------------------------------------------
# OpenAIClient
# ---------------------------------------------------------------------------


class OpenAIClient:
    """
    Async wrapper around the OpenAI Python SDK.

    Exposes high-level methods used throughout JARVIS with built-in retry,
    token counting, and cost tracking.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key: str = settings.OPENAI_API_KEY
        self._default_model: str = settings.OPENAI_MODEL
        self._default_max_tokens: int = settings.OPENAI_MAX_TOKENS

        self._client: AsyncOpenAI = AsyncOpenAI(
            api_key=self._api_key,
            timeout=60.0,
            max_retries=0,  # We handle retries ourselves
        )

        # Cumulative usage tracking for the lifetime of this client instance
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_cost_usd: float = 0.0

        logger.info("OpenAIClient initialised with model=%s", self._default_model)

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        response_format: dict[str, Any] | None = None,
    ) -> str | AsyncGenerator[str, None]:
        """
        Call the OpenAI Chat Completions endpoint.

        Args:
            messages: OpenAI-formatted message list.
            tools: Optional list of tool definitions (function calling).
            stream: If True, returns an AsyncGenerator yielding text chunks.
            model: Override the default model.
            temperature: Sampling temperature.
            max_tokens: Maximum completion tokens.
            tool_choice: Tool choice mode ("auto", "none", or a specific tool).
            response_format: e.g. {"type": "json_object"} for structured output.

        Returns:
            Full response string if ``stream=False``, otherwise an
            AsyncGenerator that yields partial text strings.
        """
        _model = model or self._default_model
        _max_tokens = max_tokens or self._default_max_tokens

        kwargs: dict[str, Any] = {
            "model": _model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": _max_tokens,
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format

        if stream:
            return self._stream_chat(kwargs, _model)

        # Non-streaming
        async def _call() -> Any:
            return await self._client.chat.completions.create(**kwargs)

        response = await _with_retry(_call)
        self.track_usage(response.usage, _model)

        content = response.choices[0].message.content or ""
        logger.debug(
            "chat_completion finished: model=%s prompt_tokens=%d completion_tokens=%d",
            _model,
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
        )
        return content

    async def chat_completion_with_tools(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Any:
        """
        Return the raw ChatCompletion response object so the caller can inspect
        tool calls (finish_reason == "tool_calls").
        """
        _model = model or self._default_model
        _max_tokens = max_tokens or self._default_max_tokens

        async def _call() -> Any:
            return await self._client.chat.completions.create(
                model=_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=_max_tokens,
            )

        response = await _with_retry(_call)
        self.track_usage(response.usage, _model)
        return response

    async def _stream_chat(
        self, kwargs: dict[str, Any], model: str
    ) -> AsyncGenerator[str, None]:
        """Internal: yield partial text chunks from a streaming chat call."""
        async def _call() -> Any:
            return await self._client.chat.completions.create(**kwargs)

        stream = await _with_retry(_call)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def embedding(
        self,
        text: str | list[str],
        model: str = "text-embedding-3-small",
    ) -> list[list[float]]:
        """
        Generate embeddings for one or more text strings.

        Args:
            text: A single string or a list of strings.
            model: Embedding model name.

        Returns:
            List of embedding vectors (one per input string).
        """
        if isinstance(text, str):
            inputs: list[str] = [text]
        else:
            inputs = text

        async def _call() -> Any:
            return await self._client.embeddings.create(
                model=model,
                input=inputs,
                encoding_format="float",
            )

        response = await _with_retry(_call)

        # Track approximate cost
        if response.usage:
            cost_per_1k = _COST_TABLE.get(model, {}).get("input", 0.0)
            cost = (response.usage.total_tokens / 1000.0) * cost_per_1k
            self._total_cost_usd += cost
            self._total_prompt_tokens += response.usage.total_tokens

        vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        logger.debug(
            "embedding: model=%s inputs=%d dim=%d",
            model,
            len(inputs),
            len(vectors[0]) if vectors else 0,
        )
        return vectors

    # ------------------------------------------------------------------
    # Whisper audio transcription
    # ------------------------------------------------------------------

    async def transcribe_audio(
        self,
        audio_file_path: str,
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """
        Transcribe an audio file using Whisper.

        Args:
            audio_file_path: Path to the audio file (mp3, mp4, wav, etc.).
            language: Optional ISO-639-1 language hint (e.g. "en").
            prompt: Optional context prompt to guide transcription.

        Returns:
            Transcribed text.
        """
        path = Path(audio_file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        async def _call() -> Any:
            with open(path, "rb") as audio_file:
                kwargs: dict[str, Any] = {
                    "model": "whisper-1",
                    "file": audio_file,
                    "response_format": "text",
                }
                if language:
                    kwargs["language"] = language
                if prompt:
                    kwargs["prompt"] = prompt
                return await self._client.audio.transcriptions.create(**kwargs)

        result = await _with_retry(_call)
        transcript = result if isinstance(result, str) else getattr(result, "text", str(result))
        logger.debug("transcribe_audio: file=%s chars=%d", audio_file_path, len(transcript))
        return transcript

    # ------------------------------------------------------------------
    # Content moderation
    # ------------------------------------------------------------------

    async def moderate_content(self, text: str) -> dict[str, Any]:
        """
        Run text through the OpenAI moderation endpoint.

        Returns:
            dict with keys:
            - flagged (bool): whether the content was flagged
            - categories (dict): per-category boolean flags
            - category_scores (dict): per-category float scores
        """
        async def _call() -> Any:
            return await self._client.moderations.create(
                model="omni-moderation-latest",
                input=text,
            )

        response = await _with_retry(_call)
        result = response.results[0]

        return {
            "flagged": result.flagged,
            "categories": {
                k: v
                for k, v in result.categories.model_dump().items()
            },
            "category_scores": {
                k: v
                for k, v in result.category_scores.model_dump().items()
            },
        }

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def count_tokens(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> int:
        """
        Estimate the number of tokens consumed by a list of chat messages.

        Uses tiktoken for accurate per-model counting.  Falls back to a
        character-based heuristic if the model is not supported.

        Args:
            messages: OpenAI chat message dicts (role + content).
            model: Model name used to select the appropriate tokeniser.

        Returns:
            Estimated token count.
        """
        _model = model or self._default_model

        try:
            enc = tiktoken.encoding_for_model(_model)
        except KeyError:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                # Ultimate fallback: ~4 chars per token
                total_chars = sum(
                    len(str(m.get("content", ""))) for m in messages
                )
                return total_chars // 4

        # OpenAI chat format overhead constants
        # See: https://platform.openai.com/docs/guides/chat/managing-tokens
        tokens_per_message = 3
        tokens_per_name = 1
        total = 0

        for msg in messages:
            total += tokens_per_message
            for key, value in msg.items():
                if isinstance(value, str):
                    total += len(enc.encode(value))
                elif isinstance(value, list):
                    # content can be a list of content parts
                    for part in value:
                        if isinstance(part, dict):
                            text = part.get("text", "")
                            if text:
                                total += len(enc.encode(text))
                if key == "name":
                    total += tokens_per_name

        total += 3  # every reply is primed with <|start|>assistant<|message|>
        return total

    def count_text_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens in a plain text string."""
        _model = model or self._default_model
        try:
            enc = tiktoken.encoding_for_model(_model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    # ------------------------------------------------------------------
    # Usage / cost tracking
    # ------------------------------------------------------------------

    def track_usage(self, usage: Any | None, model: str) -> None:
        """
        Record token usage from an API response and accumulate cost.

        Args:
            usage: The ``usage`` object from an OpenAI response.
            model: Model name for cost look-up.
        """
        if usage is None:
            return

        prompt_tokens: int = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens: int = getattr(usage, "completion_tokens", 0) or 0

        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        costs = _COST_TABLE.get(model, _COST_TABLE.get(self._default_model, {}))
        input_cost = (prompt_tokens / 1000.0) * costs.get("input", 0.0)
        output_cost = (completion_tokens / 1000.0) * costs.get("output", 0.0)
        self._total_cost_usd += input_cost + output_cost

        logger.debug(
            "track_usage: model=%s prompt=%d completion=%d cost=$%.6f "
            "cumulative_cost=$%.4f",
            model,
            prompt_tokens,
            completion_tokens,
            input_cost + output_cost,
            self._total_cost_usd,
        )

    def get_usage_stats(self) -> dict[str, Any]:
        """Return cumulative usage statistics for this client instance."""
        return {
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "estimated_cost_usd": round(self._total_cost_usd, 6),
        }

    def reset_usage_stats(self) -> None:
        """Reset cumulative usage counters."""
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost_usd = 0.0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_openai_client_instance: OpenAIClient | None = None
_instance_lock: asyncio.Lock | None = None


def get_openai_client() -> OpenAIClient:
    """
    Return the process-level OpenAIClient singleton.

    Thread-safe at the module level; for async safety within a single event
    loop, this relies on the GIL for the initial assignment.
    """
    global _openai_client_instance
    if _openai_client_instance is None:
        _openai_client_instance = OpenAIClient()
    return _openai_client_instance
