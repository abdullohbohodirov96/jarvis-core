import json
import logging
from typing import Any, Optional
from openai import AsyncOpenAI

from core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Вы — элитный персональный AI-секретарь NEXUS. Ваша цель — анализировать переписку между пользователем (владельцем аккаунта) и его собеседником, выявлять неявные и явные обязательства, va'dalar (обещания) и so'rovlar (запросы на действие), и извлекать их в виде структурированных задач.

Вам предоставляется последнее сообщение, а также история последних нескольких сообщений для понимания контекста разговора.

Проанализируйте ПОСЛЕДНЕЕ сообщение с учетом истории диалога и определите, содержит ли оно:
1. Обещание пользователя что-то сделать для собеседника (va'da / commitment) в ответ на его слова или просьбу.
   Пример контекста: 
     Собеседник: "vazifa berilsa tashab berin" (когда дадут задание, скиньте его)
     Пользователь (последнее): "ertaga hop tashab beraman" (ладно, завтра скину)
   В этом случае ПОСЛЕДНЕЕ сообщение пользователя является обещанием скинуть задание (vazifani yuborish).
   Экстракт: Title = "Va'da: Vazifani yuborish", Description = "Foydalanuvchi va'dasi: 'ertaga hop tashab beraman' (Suhbatdosh so'roviga javoban: 'vazifa berilsa tashab berin')"
2. Запрос/поручение собеседника к пользователю сделать что-то (so'rov / request).
   Пример: "iltimos, maktubni yuboring", "mojesh proverit kod?".
   Экстракт: Title = "So'rov: Maktub yuborish", Description = "Foydalanuvchi so'rovi: '...'"

Вы должны вернуть строго структурированный JSON ответ.
Формат ответа:
{
  "has_task": true/false (содержит ли ПОСЛЕДНЕЕ сообщение задачу/обещание/запрос),
  "title": "Краткое название задачи на языке оригинала сообщения (Uzbek или Russian)",
  "description": "Подробное описание задачи, включая контекст обещания или запроса, имя собеседника и цитаты из диалога",
  "priority": "low" / "medium" / "high",
  "due_date": "ISO8601 дата, если в тексте указано время (например, 'ertaga 12:00' -> '2026-05-24T12:00:00'), иначе null"
}

Правила:
- Будьте точными. Игнорируйте обычную беседу ("salom", "qanday", "yaxshi").
- Задачи должны быть четкими и выполнимыми.
- В поле description подробно распишите диалог из истории, чтобы владелец понимал, в связи с чем возникло это обязательство.
"""

class NexusAnalyzer:
    """Uses OpenAI API to analyze conversation and extract commitments and tasks."""
    
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = "gpt-4o-mini"
        logger.info(f"NexusAnalyzer initialized with model: {self._model}")

    async def analyze_message(
        self,
        sender_name: str,
        message_text: str,
        chat_title: str,
        is_outgoing: bool,
        history_text: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """
        Analyze a single message and return a dict representing the task if found, else None.
        """
        if not message_text.strip():
            return None

        role_label = "Вы (владелец)" if is_outgoing else f"Собеседник '{sender_name}'"
        user_content = (
            f"Контекст чата: {chat_title}\n"
            f"Последнее сообщение: \"{message_text}\" (Отправитель: {role_label})\n"
        )
        if history_text:
            user_content += f"\nИстория последних сообщений в этом чате (для контекста):\n{history_text}"


        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=500
            )

            result_json = response.choices[0].message.content
            if not result_json:
                return None

            data = json.loads(result_json)
            if data.get("has_task"):
                # Append sender and chat context to description for clarity
                extra_desc = f"\n\n[Автоматически извлечено из чата: '{chat_title}' | Отправитель: {sender_name}]"
                data["description"] = (data.get("description") or "") + extra_desc
                return data
            return None

        except Exception as exc:
            logger.error(f"Error during message analysis in NexusAnalyzer: {exc}")
            return None

    async def parse_direct_task(self, text: str) -> Optional[dict[str, Any]]:
        """
        Parse natural language tasks sent directly by the user to the bot.
        E.g. 'loyihani tugatishim kerak ertagacha' -> Task
        """
        prompt = """
Вы — персональный секретарь. Извлеките задачу из прямого указания пользователя.
Верните строго JSON:
{
  "has_task": true,
  "title": "Краткое название задачи",
  "description": "Описание или пусто",
  "priority": "low" / "medium" / "high",
  "due_date": "ISO8601 дата или null"
}
"""
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Указание пользователя: \"{text}\""}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            result_json = response.choices[0].message.content
            if not result_json:
                return None
            return json.loads(result_json)
        except Exception as e:
            logger.error(f"Error parsing direct task in NexusAnalyzer: {e}")
            return None
