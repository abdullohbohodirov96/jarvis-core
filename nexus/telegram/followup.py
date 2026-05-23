"""Follow-up scheduler: checks for pending follow-ups every 60 seconds and sends reminders."""
import asyncio
import logging
from datetime import timezone, datetime

from database.connection import AsyncSessionLocal
import database.operations as db_ops

logger = logging.getLogger(__name__)


async def run_followup_scheduler(bot_instance) -> None:
    """Infinite loop that checks for pending follow-ups every 60 seconds."""
    logger.info("Follow-up scheduler started.")
    while True:
        try:
            await asyncio.sleep(60)
            await process_pending_followups(bot_instance)
        except asyncio.CancelledError:
            logger.info("Follow-up scheduler cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in follow-up scheduler: {e}")


async def process_pending_followups(bot_instance) -> None:
    """Check and send follow-up messages for pending todos."""
    async with AsyncSessionLocal() as db:
        try:
            pending = await db_ops.get_todos_pending_followup(db)
            for todo in pending:
                try:
                    chat_context = ""
                    if todo.from_chat_name:
                        chat_context = f" (<b>{todo.from_chat_name}</b> bilan suhbatdan)"

                    source_text = ""
                    if todo.source == "promise":
                        source_text = "🤝 <b>Sizning va'dangiz</b>"
                    elif todo.source == "request":
                        source_text = "📨 <b>Sizga yo'naltirilgan so'rov</b>"
                    else:
                        source_text = "📋 <b>Vazifa eslatmasi</b>"

                    follow_up_text = (
                        f"⏰ <b>JARVIS Eslatmasi</b>\n\n"
                        f"{source_text}{chat_context}\n\n"
                        f"📝 <b>{todo.title}</b>\n"
                    )
                    if todo.description:
                        follow_up_text += f"<i>{todo.description[:200]}</i>\n"

                    follow_up_text += (
                        f"\n<b>Nima bo'ldi? Bajarildi yoki hali davom etmoqdami?</b>\n\n"
                        f"<i>Ushbu eslatma JARVIS tomonidan avtomatik yuborildi.</i>"
                    )

                    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
                    keyboard = InlineKeyboardMarkup(inline_keyboard=[
                        [
                            InlineKeyboardButton(text="✅ Bajarildi!", callback_data=f"fu_done:{todo.id}"),
                            InlineKeyboardButton(text="⏳ Hali davom etmoqda", callback_data=f"fu_pending:{todo.id}"),
                        ],
                        [
                            InlineKeyboardButton(text="❌ Bekor qilindi", callback_data=f"fu_cancel:{todo.id}"),
                        ]
                    ])

                    await bot_instance.send_message(
                        chat_id=todo.tg_id,
                        text=follow_up_text,
                        parse_mode="HTML",
                        reply_markup=keyboard
                    )

                    await db_ops.mark_followup_sent(db, todo.id)
                    await db.commit()
                    logger.info(f"Follow-up sent for todo #{todo.id}: '{todo.title}'")

                except Exception as e:
                    logger.error(f"Error sending follow-up for todo #{todo.id}: {e}")
        except Exception as e:
            logger.error(f"Error processing follow-ups: {e}")
