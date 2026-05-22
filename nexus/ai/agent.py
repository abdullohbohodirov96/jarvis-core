import json
from typing import Any

from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError

from ai.prompts import SYSTEM_PROMPT, TASK_EXTRACTION_PROMPT
from core.config import settings
from utils.logger import logger


class NexusBrain:
    """
    Async AI brain for the NEXUS assistant.

    Wraps the OpenAI AsyncOpenAI client and exposes two high-level
    operations:

    - ``generate_response``  — conversational reply using SYSTEM_PROMPT
    - ``extract_tasks``      — structured task extraction using
                               TASK_EXTRACTION_PROMPT with JSON-mode output

    A single client instance is created at construction time and reused
    for all requests; the underlying httpx transport maintains connection
    pooling automatically.
    """

    _MODEL_CHAT: str = "gpt-4-turbo-preview"
    _MODEL_JSON: str = "gpt-4-turbo-preview"
    _MAX_RETRIES: int = 3
    _TEMPERATURE_CHAT: float = 0.7
    _TEMPERATURE_JSON: float = 0.0

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            max_retries=self._MAX_RETRIES,
            timeout=30.0,
        )
        logger.info("NexusBrain initialised (model={})", self._MODEL_CHAT)

    async def generate_response(self, user_input: str) -> str:
        """
        Generate a conversational reply for ``user_input``.

        Parameters
        ----------
        user_input:
            Raw text from the end user (Telegram message, API request, …).

        Returns
        -------
        str
            NEXUS's reply, or a graceful error message if the API call
            fails after all retries.
        """
        logger.debug("generate_response | input_length={}", len(user_input))
        try:
            completion = await self._client.chat.completions.create(
                model=self._MODEL_CHAT,
                temperature=self._TEMPERATURE_CHAT,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ],
            )
            reply: str = completion.choices[0].message.content or ""
            logger.debug(
                "generate_response | tokens_used={} reply_length={}",
                completion.usage.total_tokens if completion.usage else "?",
                len(reply),
            )
            return reply.strip()

        except RateLimitError:
            logger.warning("OpenAI rate-limit hit — backing off.")
            return (
                "I'm momentarily overloaded, Abdulloh. "
                "Give me a few seconds and try again."
            )
        except APIConnectionError as exc:
            logger.error("OpenAI connection error: {}", exc)
            return "I lost my connection to the AI engine. Please retry."
        except APIError as exc:
            logger.error("OpenAI API error (status={}): {}", exc.status_code, exc)
            return f"An API error occurred (HTTP {exc.status_code}). I'll be back shortly."
        except Exception as exc:
            logger.exception("Unexpected error in generate_response: {}", exc)
            return "Something unexpected happened. Logging it now."

    async def extract_tasks(self, text: str) -> dict[str, Any]:
        """
        Extract a structured task from free-form text using JSON mode.

        Parameters
        ----------
        text:
            The user's raw message or any natural-language snippet.

        Returns
        -------
        dict
            Always returns a dict.  On success it matches the schema::

                {
                    "has_task": bool,
                    "title":    str | None,
                    "description": str | None
                }

            On failure it returns ``{"has_task": False, "error": "<reason>"}``.
        """
        logger.debug("extract_tasks | text_length={}", len(text))
        try:
            completion = await self._client.chat.completions.create(
                model=self._MODEL_JSON,
                temperature=self._TEMPERATURE_JSON,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": TASK_EXTRACTION_PROMPT},
                    {"role": "user", "content": text},
                ],
            )
            raw: str = completion.choices[0].message.content or "{}"
            result: dict[str, Any] = json.loads(raw)

            if not isinstance(result, dict):
                raise ValueError(f"Expected dict, got {type(result).__name__}")

            result.setdefault("has_task", False)
            result.setdefault("title", None)
            result.setdefault("description", None)

            logger.debug(
                "extract_tasks | has_task={} title={!r}",
                result["has_task"],
                result.get("title"),
            )
            return result

        except json.JSONDecodeError as exc:
            logger.error("Task extraction returned invalid JSON: {}", exc)
            return {"has_task": False, "error": "invalid_json"}
        except RateLimitError:
            logger.warning("OpenAI rate-limit hit during task extraction.")
            return {"has_task": False, "error": "rate_limited"}
        except APIError as exc:
            logger.error("OpenAI API error during task extraction: {}", exc)
            return {"has_task": False, "error": f"api_error_{exc.status_code}"}
        except Exception as exc:
            logger.exception("Unexpected error in extract_tasks: {}", exc)
            return {"has_task": False, "error": "unexpected"}
