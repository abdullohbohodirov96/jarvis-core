#!/usr/bin/env python3
"""
JARVIS Telegram Bot — O'zbek tilida ovozli AI assistant.

Telegram orqali matnli va ovozli xabarlar qabul qiladi,
OpenAI GPT-4o bilan javob beradi, ovozni Whisper API bilan taniydi.

Ishga tushirish:
    TELEGRAM_BOT_TOKEN=... OPENAI_API_KEY=... python jarvis_bot.py
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Ensure jarvis/ is importable
_ROOT = Path(__file__).parent
_JARVIS = _ROOT / "jarvis"
for p in [str(_ROOT), str(_JARVIS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import FSInputFile
from openai import AsyncOpenAI

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("jarvis_bot")

# ── Config ──
# Try to load from jarvis/.env first
_env_path = _JARVIS / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)
    log.info("Loaded .env from %s", _env_path)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ADMIN_ID = int(os.getenv("ADMIN_CHAT_ID", "0"))

if not BOT_TOKEN:
    log.error("TELEGRAM_BOT_TOKEN not set! Get it from @BotFather")
    sys.exit(1)
if not OPENAI_API_KEY:
    log.error("OPENAI_API_KEY not set!")
    sys.exit(1)

# ── OpenAI client ──
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=60.0)

# ── Bot & Dispatcher ──
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# ── Per-user conversation history ──
conversations: dict[int, list[dict]] = {}
MAX_HISTORY = 20


# ══════════════════════════════════════════════════════════════
# JARVIS System Prompt (O'zbek tilida)
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Sen JARVIS — foydalanuvchining shaxsiy AI assistantisan.
Sen Tony Stark ning JARVIS assistantigadek gapirasansan: professional, aqlli, biroz hazilkash va samarali.

Hozirgi vaqt: {current_time}

Sen quyidagi ishlarni bajara olasan:
1. Har qanday savolga javob berish
2. Maslahot va tavsiya berish
3. Matn yozish, tarjima qilish
4. Kod yozish va tushuntirish
5. Vazifalar rejalashtirish
6. Internet haqida ma'lumot berish

MUHIM QOIDALAR:
- Har doim O'ZBEK TILIDA javob ber (agar foydalanuvchi boshqa tilda so'ramasa)
- Qisqa va aniq javob ber — 1-3 gap yetarli oddiy savollar uchun
- Murakkab savollar uchun batafsilroq javob ber
- JARVIS shaxsiyatida gapir: "Ha, xo'jayin", "Albatta, xo'jayin" kabi iboralar ishlat
- Hech qachon "Men AI man" dema — sen JARVIS san
"""


# ══════════════════════════════════════════════════════════════
# AI Chat Function
# ══════════════════════════════════════════════════════════════

async def ai_chat(user_id: int, user_text: str) -> str:
    """Send user message to GPT-4o and return response."""
    # Get or create conversation history
    if user_id not in conversations:
        conversations[user_id] = []

    history = conversations[user_id]
    history.append({"role": "user", "content": user_text})

    # Build messages
    ct = datetime.now(timezone.utc).strftime("%A, %B %d %Y, %H:%M UTC")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(current_time=ct)},
        *history[-MAX_HISTORY:],
    ]

    try:
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )
        reply = response.choices[0].message.content or "Kechirasiz, javob berolmadim."

        # Save assistant reply to history
        history.append({"role": "assistant", "content": reply})

        # Trim history
        if len(history) > MAX_HISTORY * 2:
            conversations[user_id] = history[-MAX_HISTORY * 2:]

        return reply

    except Exception as e:
        log.error("OpenAI error: %s", e)
        return f"Xo'jayin, texnik xatolik yuz berdi: {e}"


# ══════════════════════════════════════════════════════════════
# Voice Transcription (Whisper API)
# ══════════════════════════════════════════════════════════════

async def transcribe_voice(voice_file: io.BytesIO, filename: str = "voice.ogg") -> str:
    """Transcribe voice message using OpenAI Whisper API."""
    try:
        voice_file.name = filename  # type: ignore
        transcript = await openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=voice_file,
            response_format="text",
        )
        result = transcript if isinstance(transcript, str) else str(transcript)
        return result.strip()
    except Exception as e:
        log.error("Whisper transcription error: %s", e)
        return ""


# ══════════════════════════════════════════════════════════════
# Handlers
# ══════════════════════════════════════════════════════════════

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    """Handle /start command."""
    user = message.from_user
    name = user.first_name if user else "Xo'jayin"

    welcome = (
        f"Assalomu alaykum, {name}! 🤖\n\n"
        f"Men **JARVIS** — sizning shaxsiy AI assistantingizman.\n\n"
        f"📝 Menga matn yozing — javob beraman\n"
        f"🎤 Ovozli xabar yuboring — eshitaman va javob beraman\n\n"
        f"Sizga qanday yordam bera olaman?"
    )
    await message.answer(welcome, parse_mode=ParseMode.MARKDOWN)
    log.info("User %s (%s) started the bot", name, user.id if user else "?")


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    """Handle /help command."""
    help_text = (
        "🤖 **JARVIS — Buyruqlar**\n\n"
        "📝 **Matn yozing** — har qanday savol, buyruq, maslahot\n"
        "🎤 **Ovozli xabar** — gapiring, men eshitaman\n"
        "🔄 /reset — suhbatni tozalash\n"
        "❓ /help — shu yordam\n\n"
        "**Misol buyruqlar:**\n"
        "• Bugun ob-havo qanday?\n"
        "• Python da list comprehension tushuntir\n"
        "• Ingliz tilidan tarjima qil: Hello world\n"
        "• Ertaga uchun reja tuz\n"
    )
    await message.answer(help_text, parse_mode=ParseMode.MARKDOWN)


@dp.message(Command("reset"))
async def cmd_reset(message: types.Message):
    """Reset conversation history."""
    user_id = message.from_user.id if message.from_user else 0
    conversations.pop(user_id, None)
    await message.answer("🔄 Suhbat tarixi tozalandi. Yangi suhbat boshlang!")
    log.info("User %s reset conversation", user_id)


@dp.message(F.voice)
async def handle_voice(message: types.Message):
    """Handle voice messages — transcribe with Whisper, then process with GPT."""
    user_id = message.from_user.id if message.from_user else 0
    user_name = message.from_user.first_name if message.from_user else "?"

    # Show typing indicator
    await bot.send_chat_action(message.chat.id, "typing")

    # Download voice file
    voice = message.voice
    if not voice:
        await message.answer("Ovozli xabar topilmadi.")
        return

    try:
        file = await bot.get_file(voice.file_id)
        file_path = file.file_path
        if not file_path:
            await message.answer("Ovoz faylini yuklab bo'lmadi.")
            return

        # Download to BytesIO
        voice_bytes = io.BytesIO()
        await bot.download_file(file_path, voice_bytes)
        voice_bytes.seek(0)

        # Transcribe
        transcript = await transcribe_voice(voice_bytes, "voice.ogg")

        if not transcript:
            await message.answer("Kechirasiz, ovozingizni aniqlay olmadim. Qayta urinib ko'ring.")
            return

        log.info("Voice from %s (%s): %s", user_name, user_id, transcript[:100])

        # Show what was heard
        await message.answer(f"🎤 *Eshitdim:* {transcript}", parse_mode=ParseMode.MARKDOWN)

        # Process with AI
        await bot.send_chat_action(message.chat.id, "typing")
        reply = await ai_chat(user_id, transcript)

        await message.answer(reply, parse_mode=ParseMode.MARKDOWN)
        log.info("JARVIS replied to voice from %s", user_name)

    except Exception as e:
        log.error("Voice handling error: %s", e)
        await message.answer(f"Ovozli xabar qayta ishlashda xatolik: {e}")


@dp.message(F.text)
async def handle_text(message: types.Message):
    """Handle text messages — process with GPT-4o."""
    user_id = message.from_user.id if message.from_user else 0
    user_name = message.from_user.first_name if message.from_user else "?"
    text = message.text or ""

    if not text.strip():
        return

    log.info("Text from %s (%s): %s", user_name, user_id, text[:100])

    # Show typing indicator
    await bot.send_chat_action(message.chat.id, "typing")

    # Process with AI
    reply = await ai_chat(user_id, text)

    # Send reply (split if too long for Telegram)
    if len(reply) <= 4096:
        try:
            await message.answer(reply, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            # Markdown parsing failed — send as plain text
            await message.answer(reply)
    else:
        # Split long messages
        for i in range(0, len(reply), 4096):
            chunk = reply[i:i + 4096]
            try:
                await message.answer(chunk, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await message.answer(chunk)

    log.info("JARVIS replied to %s", user_name)


# ══════════════════════════════════════════════════════════════
# Startup
# ══════════════════════════════════════════════════════════════

async def on_startup():
    """Bot startup tasks."""
    me = await bot.get_me()
    log.info("=" * 50)
    log.info("JARVIS Telegram Bot started!")
    log.info("Bot: @%s (%s)", me.username, me.full_name)
    log.info("Model: %s", OPENAI_MODEL)
    log.info("=" * 50)

    # Notify admin
    if ADMIN_ID:
        try:
            await bot.send_message(
                ADMIN_ID,
                "🟢 JARVIS bot ishga tushdi!\n"
                f"Model: {OPENAI_MODEL}\n"
                f"Vaqt: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
        except Exception:
            pass


async def main():
    """Main entry point."""
    log.info("Starting JARVIS Telegram Bot...")
    dp.startup.register(on_startup)

    # Delete webhook and start polling
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
