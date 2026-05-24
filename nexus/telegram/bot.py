import asyncio
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional
from aiogram import Bot, Dispatcher, Router, F, BaseMiddleware
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    Message,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    WebAppInfo,
    TelegramObject,
    CallbackQuery
)
from aiogram.client.default import DefaultBotProperties
from aiogram.utils.keyboard import InlineKeyboardBuilder

from core.config import settings
from database.connection import AsyncSessionLocal
import database.operations as db_ops
from ai.analyzer import NexusAnalyzer

logger = logging.getLogger(__name__)

bot: Optional[Bot] = None
dp: Optional[Dispatcher] = None
router = Router(name="nexus_bot_router")

# ── 1. Access Control Middleware ──

class AccessMiddleware(BaseMiddleware):
    """
    Middleware that checks user registration and access control status.
    Automatically registers new users in pending status.
    Supports both Message and CallbackQuery events.
    """
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        tg_user = None
        if isinstance(event, Message):
            tg_user = event.from_user
        elif isinstance(event, CallbackQuery):
            tg_user = event.from_user

        if not tg_user:
            return await handler(event, data)

        telegram_id = tg_user.id
        username = tg_user.username
        
        async with AsyncSessionLocal() as db:
            try:
                # Get or create user (owner is auto-approved)
                is_owner = (telegram_id == settings.OWNER_ID)
                user = await db_ops.get_or_create_user(
                    db=db,
                    tg_id=telegram_id,
                    username=username,
                    is_owner=is_owner
                )
                await db.commit()

                # If user is blocked / pending
                if not user.is_allowed:
                    if isinstance(event, Message):
                        if event.text and event.text.startswith("/start"):
                            await event.answer(
                                "⛔️ <b>NEXUS-ga ruxsat cheklangan.</b>\n\n"
                                "Sizning ro'yxatdan o'tish so'rovingiz ma'murlar tomonidan ko'rib chiqilmoqda.\n"
                                "Ruxsat berilishini kuting.",
                                parse_mode=ParseMode.HTML
                            )
                        else:
                            await event.answer("🔒 Ruxsat etilmagan. Ma'mur tomonidan tasdiqlanishini kuting.")
                    elif isinstance(event, CallbackQuery):
                        await event.answer("🔒 Ruxsat etilmagan.", show_alert=True)
                    return  # Terminate pipeline

                # Inject user and db session into the handler context
                data["db_session"] = db
                data["db_user"] = user
                return await handler(event, data)

            except Exception as e:
                logger.error(f"Error in AccessMiddleware for {telegram_id}: {e}")
                await db.rollback()
                if isinstance(event, Message):
                    await event.answer("⚠️ Tizimda xatolik yuz berdi. Keyinroq qayta urinib ko'ring.")
                elif isinstance(event, CallbackQuery):
                    await event.answer("⚠️ Tizimda xatolik yuz berdi.", show_alert=True)
                return

# ── Helper for Task Keyboard ──

def get_tasks_keyboard(todos: list) -> InlineKeyboardMarkup:
    """Generate interactive inline keyboard for tasks."""
    builder = InlineKeyboardBuilder()
    
    # List only incomplete tasks
    active_todos = [t for t in todos if not t.is_done][:5] # Limit to 5 for clean buttons
    
    # Create complete and delete buttons row for each task
    for idx, todo in enumerate(active_todos, 1):
        builder.row(
            InlineKeyboardButton(text=f"✅ #{idx}", callback_data=f"todo_complete:{todo.id}"),
            InlineKeyboardButton(text=f"🔥 P:{todo.priority[0].upper()}", callback_data=f"todo_pri_menu:{todo.id}"),
            InlineKeyboardButton(text=f"❌ O'chirish", callback_data=f"todo_delete:{todo.id}")
        )
        
    # Bottom control buttons
    builder.row(
        InlineKeyboardButton(text="🔄 Yangilash", callback_data="todo_refresh"),
        InlineKeyboardButton(text="🚀 Nexus App", web_app=WebAppInfo(url=settings.MINI_APP_URL))
    )
    return builder.as_markup()

async def render_tasks_message(db_session, tg_id: int) -> str:
    """Render list of active tasks."""
    todos = await db_ops.get_todos(db_session, tg_id)
    active_todos = [t for t in todos if not t.is_done]
    
    if not active_todos:
        return "🎉 <b>Hamma ishlar bajarildi! Hozircha faol vazifalar yo'q.</b>\n\nChiroyli dam oling!"
        
    text = f"📋 <b>Sizning faol vazifalaringiz ({len(active_todos)} ta):</b>\n\n"
    for idx, t in enumerate(active_todos[:5], 1):
        priority_label = "🔴 Yuqori" if t.priority == "high" else "🟡 O'rta" if t.priority == "medium" else "🔵 Past"
        due_str = ""
        if t.due_date:
            due_str = f" (Muddati: {t.due_date.strftime('%d.%m %H:%M')})"
            
        text += f"<b>#{idx}. {t.title}</b>\n"
        if t.description:
            text += f"   <i>{t.description[:80]}...</i>\n"
        text += f"   Pr: {priority_label}{due_str}\n\n"
        
    if len(active_todos) > 5:
        text += f"<i>... va yana {len(active_todos) - 5} ta vazifa Mini App ichida.</i>\n"
        
    text += "\n<i>Tugmalar yordamida vazifalarni yakunlang (✅), ustuvorlikni o'zgartiring (🔥) yoki o'chiring (❌).</i>"
    return text

# ── 2. Commands & Handlers ──

@router.message(Command("start"))
async def cmd_start(message: Message, db_user):
    name = message.from_user.first_name
    admin_badge = " [Ma'mur]" if db_user.is_admin else ""
    
    # Safe check of MINI_APP_URL
    app_url = settings.MINI_APP_URL.strip() if settings.MINI_APP_URL else ""
    reply_markup = None
    warning_text = ""
    
    if app_url.startswith("http://") or app_url.startswith("https://"):
        # Valid URL format
        reply_markup = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🚀 NEXUS App-ni ochish",
                    web_app=WebAppInfo(url=app_url)
                )
            ]
        ])
    else:
        # Invalid / Empty URL
        warning_text = (
            "\n\n⚠️ <b>Dasturchi uchun ogohlantirish:</b>\n"
            "<code>.env</code> faylidagi <code>MINI_APP_URL</code> o'zgaruvchisi o'rnatilmagan yoki noto'ri'g'ri formatda (u https:// bilan boshlanishi kerak).\n"
            "Xatoliklarning oldini olish uchun WebApp tugmasi yashirildi."
        )
    
    welcome_text = (
        f"👋 Assalomu alaykum, <b>{name}</b>!{admin_badge}\n\n"
        f"<b>NEXUS</b> — integratsiyalar va vazifalarni boshqarish bo'yicha shaxsiy yordamchingizga xush kelibsiz!\n\n"
        f"Barcha amallar Telegram Mini App interfeysi orqali amalga oshiriladi.{warning_text}"
    )
    
    if reply_markup:
        await message.answer(welcome_text, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
    else:
        await message.answer(welcome_text, parse_mode=ParseMode.HTML)

def render_board_message(todos: list) -> str:
    """Render the 4-section TODO board as a text message."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    sections = {
        "vazifalar": [],
        "kutilmoqda": [],
        "keraklilar": [],
        "bajarildi": []
    }

    for t in todos:
        section = getattr(t, 'section', 'vazifalar')
        if section in sections:
            sections[section].append(t)

    lines = ["📋 <b>JARVIS — Vazifalar Taxtasi</b>\n"]
    lines.append(f"🕐 Yangilandi: {now.strftime('%d.%m %H:%M')} UTC\n")

    emojis = {"vazifalar": "📌", "kutilmoqda": "⏳", "keraklilar": "🛒", "bajarildi": "✅"}
    names = {"vazifalar": "VAZIFALAR", "kutilmoqda": "KUTILMOQDA", "keraklilar": "KERAKLILARI", "bajarildi": "BAJARILDI"}

    for section_key in ["vazifalar", "kutilmoqda", "keraklilar", "bajarildi"]:
        items = sections[section_key]
        lines.append(f"\n{emojis[section_key]} <b>{names[section_key]}</b> ({len(items)})")
        if items:
            for t in items[:5]:
                priority_mark = "🔴" if t.priority == "high" else "🟡" if t.priority == "medium" else "🔵"
                lines.append(f"  {priority_mark} {t.title}")
            if len(items) > 5:
                lines.append(f"  <i>... va yana {len(items)-5} ta</i>")
        else:
            lines.append("  <i>Bo'sh</i>")

    total_active = len(sections["vazifalar"]) + len(sections["kutilmoqda"]) + len(sections["keraklilar"])
    lines.append(f"\n<i>Jami faol: {total_active} | Bajarildi: {len(sections['bajarildi'])}</i>")

    return "\n".join(lines)


def get_board_keyboard(todos: list) -> InlineKeyboardMarkup:
    """Generate keyboard for the board message."""
    builder = InlineKeyboardBuilder()

    active_todos = [t for t in todos if not t.is_done][:3]
    for todo in active_todos:
        builder.row(
            InlineKeyboardButton(text=f"✅ {todo.title[:20]}", callback_data=f"todo_complete:{todo.id}"),
        )

    builder.row(
        InlineKeyboardButton(text="🔄 Yangilash", callback_data="todo_refresh"),
        InlineKeyboardButton(text="📊 To'liq hisobot", callback_data="full_report"),
    )

    app_url = settings.MINI_APP_URL.strip() if settings.MINI_APP_URL else ""
    if app_url.startswith("http"):
        builder.row(
            InlineKeyboardButton(text="🚀 Mini App", web_app=WebAppInfo(url=app_url))
        )

    return builder.as_markup()


async def update_pinned_board(bot_instance, tg_id: int, db_session) -> None:
    """Update or create the pinned TODO board message in the bot chat."""
    try:
        todos = await db_ops.get_todos(db_session, tg_id)
        user = await db_ops.get_user(db_session, tg_id)

        text = render_board_message(todos)
        keyboard = get_board_keyboard(todos)

        if user and user.pinned_msg_id:
            try:
                await bot_instance.edit_message_text(
                    chat_id=tg_id,
                    message_id=user.pinned_msg_id,
                    text=text,
                    parse_mode="HTML",
                    reply_markup=keyboard
                )
                return
            except Exception:
                pass

        # Send new pinned message
        msg = await bot_instance.send_message(
            chat_id=tg_id,
            text=text,
            parse_mode="HTML",
            reply_markup=keyboard
        )
        await bot_instance.pin_chat_message(chat_id=tg_id, message_id=msg.message_id, disable_notification=True)
        await db_ops.update_pinned_msg_id(db_session, tg_id, msg.message_id)
        await db_session.commit()
    except Exception as e:
        logger.error(f"Error updating pinned board: {e}")


@router.message(Command("tasks"))
async def cmd_tasks(message: Message, db_session):
    todos = await db_ops.get_todos(db_session, message.from_user.id)
    text = render_board_message(todos)
    reply_markup = get_board_keyboard(todos)
    await message.answer(text, parse_mode=ParseMode.HTML, reply_markup=reply_markup)

@router.message(Command("allow"))
async def cmd_allow(message: Message, command: CommandObject, db_session, db_user):
    if not db_user.is_admin:
        await message.answer("❌ Bu buyruq faqat ma'murlar uchun ruxsat etilgan.")
        return
        
    username = command.args
    if not username:
        await message.answer("⚠️ Foydalanuvchi nomini (@username) ko'rsating. Misol: <code>/allow @username</code>", parse_mode=ParseMode.HTML)
        return
        
    user = await db_ops.allow_user(db_session, username)
    if not user:
        await message.answer(f"❌ Foydalanuvchi {username} ma'lumotlar bazasida topilmadi (u kamida bir marta botni ishga tushirgan bo'lishi kerak).")
        return
        
    await db_session.commit()
    await message.answer(f"✅ Foydalanuvchi <b>{username}</b> uchun ruxsat muvaffaqiyatli tasdiqlandi! 🚀", parse_mode=ParseMode.HTML)
    
    # Notify whitelisted user if possible
    try:
        await bot.send_message(
            chat_id=user.tg_id,
            text="🎉 <b>Tabriklaymiz!</b> NEXUS platformasidan foydalanish so'rovingiz ma'mur tomonidan tasdiqlandi.\n"
                 "Ilovaga kirish uchun /start buyrug'ini bosing!",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logger.warning(f"Could not notify user {user.tg_id} about approval: {e}")

@router.message(Command("revoke"))
async def cmd_revoke(message: Message, command: CommandObject, db_session, db_user):
    if not db_user.is_admin:
        await message.answer("❌ Bu buyruq faqat ma'murlar uchun ruxsat etilgan.")
        return
        
    username = command.args
    if not username:
        await message.answer("⚠️ Foydalanuvchi nomini (@username) ko'rsating. Misol: <code>/revoke @username</code>", parse_mode=ParseMode.HTML)
        return
        
    user = await db_ops.revoke_user(db_session, username)
    if not user:
        await message.answer(f"❌ Foydalanuvchi {username} topilmadi.")
        return
        
    await db_session.commit()
    await message.answer(f"❌ Foydalanuvchi <b>{username}</b> uchun ruxsat bekor qilindi.", parse_mode=ParseMode.HTML)
    
    # Notify user if possible
    try:
        await bot.send_message(
            chat_id=user.tg_id,
            text="⚠️ Ma'mur sizning NEXUS platformasiga kirish huquqingizni bekor qildi.",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logger.warning(f"Could not notify user {user.tg_id} about revocation: {e}")

@router.message(Command("users"))
async def cmd_users(message: Message, db_session, db_user):
    if not db_user.is_admin:
        await message.answer("❌ Bu buyruq faqat ma'murlar uchun ruxsat etilgan.")
        return
        
    pending = await db_ops.get_pending_users(db_session)
    if not pending:
        await message.answer("👥 Tasdiqlashni kutayotgan foydalanuvchilar yo'q.")
        return
        
    text = "⏳ <b>Tasdiqlashni kutayotgan foydalanuvchilar:</b>\n\n"
    for idx, u in enumerate(pending, 1):
        username_str = f"@{u.username}" if u.username else f"ID: {u.tg_id}"
        text += f"{idx}. {username_str} (ariza sanasi: {u.created_at.strftime('%d.%m.%Y %H:%M')})\n"
        
    text += "\nRuxsat berish uchun <code>/allow @username</code> buyrug'idan foydalaning."
    await message.answer(text, parse_mode=ParseMode.HTML)

# ── 3. Direct Message / Chat-to-Task Parser ──

@router.message(F.text & ~F.text.startswith("/"))
async def on_direct_chat_message(message: Message, db_session):
    """Directly parse natural language text into structured task saved in DB."""
    text = message.text.strip()
    if not text:
        return
        
    logger.info(f"Parsing direct chat task for {message.from_user.id}: '{text}'")
    analyzer = NexusAnalyzer()
    
    # Send typing status
    await bot.send_chat_action(chat_id=message.chat_id, action="typing")
    
    task_data = await analyzer.parse_direct_task(text)
    if task_data and task_data.get("has_task"):
        # Save to database
        due_date = None
        if task_data.get("due_date"):
            try:
                due_date = datetime.fromisoformat(task_data["due_date"].replace("Z", "+00:00"))
            except Exception:
                pass
                
        todo = await db_ops.add_todo(
            db=db_session,
            tg_id=message.from_user.id,
            title=task_data["title"],
            description=task_data.get("description") or "Kiritilgan matn orqali AI tomonidan yaratilgan vazifa.",
            priority=task_data.get("priority", "medium"),
            due_date=due_date,
            section="vazifalar",
            source="manual",
        )
        await db_session.commit()

        priority_emoji = "🔴" if todo.priority == "high" else "🟡" if todo.priority == "medium" else "🔵"
        await message.answer(
            f"✅ <b>Vazifa muvaffaqiyatli saqlandi!</b>\n\n"
            f"📝 <b>{todo.title}</b>\n"
            f"🔥 Ustuvorlik: {priority_emoji} {todo.priority.upper()}\n"
            f"⏰ Muddati: {todo.due_date.strftime('%d.%m %H:%M') if todo.due_date else 'Belgilanmagan'}\n\n"
            f"<i>Barcha vazifalar ro'yxatini ko'rish uchun /tasks yozing.</i>",
            parse_mode=ParseMode.HTML
        )
        await update_pinned_board(bot, message.from_user.id, db_session)
    else:
        # Just answer with a friendly message explaining what the bot can do
        help_text = (
            "🤖 <b>Men sizning NEXUS shaxsiy yordamchingizman!</b>\n\n"
            "Menga shunchaki topshiriq yozsangiz, uni avtomatik tarzda vazifalar ro'yxatiga kiritaman.\n"
            "<i>Masalan: 'ertaga soat 10 da hisobot tayyorlashim kerak, yuqori ustuvorlik'</i>\n\n"
            "<b>Buyruqlar:</b>\n"
            "/tasks — faol vazifalar ro'yxatini va tugmalarni chiqarish\n"
            "/start — Mini App-ni ochish"
        )
        await message.answer(help_text, parse_mode=ParseMode.HTML)

# ── 4. Callback Query Handlers (Interactive buttons) ──

@router.callback_query(F.data.startswith("todo_complete:"))
async def on_callback_todo_complete(callback: CallbackQuery, db_session):
    todo_id = int(callback.data.split(":")[1])
    todo = await db_ops.complete_todo(db_session, callback.from_user.id, todo_id, is_done=True)

    if todo:
        todo_title = todo.title
        await db_session.commit()
        await callback.answer(f"✅ Vazifa bajarildi: '{todo_title}'")

        todos = await db_ops.get_todos(db_session, callback.from_user.id)
        text = render_board_message(todos)
        reply_markup = get_board_keyboard(todos)
        try:
            await callback.message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
        except Exception:
            pass
        await update_pinned_board(bot, callback.from_user.id, db_session)
    else:
        await callback.answer("❌ Vazifa topilmadi.", show_alert=True)

@router.callback_query(F.data.startswith("todo_delete:"))
async def on_callback_todo_delete(callback: CallbackQuery, db_session):
    todo_id = int(callback.data.split(":")[1])
    success = await db_ops.delete_todo(db_session, callback.from_user.id, todo_id)

    if success:
        await db_session.commit()
        await callback.answer("🗑 Vazifa muvaffaqiyatli o'chirildi.")

        todos = await db_ops.get_todos(db_session, callback.from_user.id)
        text = render_board_message(todos)
        reply_markup = get_board_keyboard(todos)
        try:
            await callback.message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
        except Exception:
            pass
        await update_pinned_board(bot, callback.from_user.id, db_session)
    else:
        await callback.answer("❌ Vazifa topilmadi.", show_alert=True)

@router.callback_query(F.data.startswith("todo_pri_menu:"))
async def on_callback_todo_pri_menu(callback: CallbackQuery, db_session):
    todo_id = int(callback.data.split(":")[1])
    
    # Priority sub-menu
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="🔴 Yuqori", callback_data=f"todo_set_pri:{todo_id}:high"),
        InlineKeyboardButton(text="🟡 O'rta", callback_data=f"todo_set_pri:{todo_id}:medium"),
        InlineKeyboardButton(text="🔵 Past", callback_data=f"todo_set_pri:{todo_id}:low")
    )
    builder.row(InlineKeyboardButton(text="⬅️ Orqaga", callback_data="todo_refresh"))
    
    await callback.message.edit_reply_markup(reply_markup=builder.as_markup())
    await callback.answer("Ustuvorlik darajasini tanlang.")

@router.callback_query(F.data.startswith("todo_set_pri:"))
async def on_callback_todo_set_pri(callback: CallbackQuery, db_session):
    parts = callback.data.split(":")
    todo_id = int(parts[1])
    priority = parts[2]

    todo = await db_ops.update_todo(db_session, callback.from_user.id, todo_id, {"priority": priority})
    if todo:
        await db_session.commit()
        await callback.answer(f"🔥 Ustuvorlik {priority.upper()} qilib o'zgartirildi!")

        todos = await db_ops.get_todos(db_session, callback.from_user.id)
        text = render_board_message(todos)
        reply_markup = get_board_keyboard(todos)
        try:
            await callback.message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
        except Exception:
            pass
    else:
        await callback.answer("❌ Xatolik yuz berdi.", show_alert=True)

@router.callback_query(F.data == "todo_refresh")
async def on_callback_todo_refresh(callback: CallbackQuery, db_session):
    await callback.answer("🔄 Ro'yxat yangilandi.")
    todos = await db_ops.get_todos(db_session, callback.from_user.id)
    text = render_board_message(todos)
    reply_markup = get_board_keyboard(todos)
    await callback.message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=reply_markup)

@router.callback_query(F.data == "full_report")
async def on_full_report(callback: CallbackQuery, db_session):
    await callback.answer("📊 Hisobot tayyorlanmoqda...")
    todos = await db_ops.get_todos(db_session, callback.from_user.id)

    sections = {"vazifalar": [], "kutilmoqda": [], "keraklilar": [], "bajarildi": []}
    for t in todos:
        section = getattr(t, 'section', 'vazifalar')
        if section in sections:
            sections[section].append(t)

    report = "📊 <b>TO'LIQ HISOBOT</b>\n\n"

    for section_key, name, emoji in [
        ("vazifalar", "VAZIFALAR", "📌"),
        ("kutilmoqda", "KUTILMOQDA", "⏳"),
        ("keraklilar", "KERAKLILARI", "🛒"),
        ("bajarildi", "BAJARILDI", "✅"),
    ]:
        items = sections[section_key]
        report += f"{emoji} <b>{name}</b>:\n"
        if items:
            for t in items:
                priority_mark = "🔴" if t.priority == "high" else "🟡" if t.priority == "medium" else "🔵"
                src = ""
                if getattr(t, 'source', 'manual') == 'promise':
                    src = " [va'da]"
                elif getattr(t, 'source', 'manual') == 'request':
                    src = " [so'rov]"
                report += f"  {priority_mark} {t.title}{src}\n"
                if t.description:
                    report += f"      <i>{t.description[:80]}</i>\n"
        else:
            report += "  <i>Bo'sh</i>\n"
        report += "\n"

    await callback.message.answer(report, parse_mode="HTML")

# ── 5. Follow-up Callback Handlers ──

@router.callback_query(F.data.startswith("fu_done:"))
async def on_followup_done(callback: CallbackQuery, db_session):
    todo_id = int(callback.data.split(":")[1])
    todo = await db_ops.move_todo_section(db_session, callback.from_user.id, todo_id, "bajarildi")
    if todo:
        await db_session.commit()
        await callback.answer("🎉 Ajoyib! Vazifa bajarildi deb belgilandi.")
        await callback.message.edit_text(
            callback.message.text + "\n\n✅ <b>Bajarildi!</b>",
            parse_mode="HTML",
            reply_markup=None
        )
        await update_pinned_board(bot, callback.from_user.id, db_session)
    else:
        await callback.answer("❌ Vazifa topilmadi.", show_alert=True)

@router.callback_query(F.data.startswith("fu_pending:"))
async def on_followup_pending(callback: CallbackQuery, db_session):
    await callback.answer("⏳ Tushundim. Keyinroq qayta eslataman.")
    await callback.message.edit_reply_markup(reply_markup=None)

@router.callback_query(F.data.startswith("fu_cancel:"))
async def on_followup_cancel(callback: CallbackQuery, db_session):
    todo_id = int(callback.data.split(":")[1])
    success = await db_ops.delete_todo(db_session, callback.from_user.id, todo_id)
    if success:
        await db_session.commit()
        await callback.answer("🗑 Vazifa o'chirildi.")
        await callback.message.edit_reply_markup(reply_markup=None)
        await update_pinned_board(bot, callback.from_user.id, db_session)
    else:
        await callback.answer("❌ Xatolik yuz berdi.", show_alert=True)

# ── 6. Lifecycle functions ──

async def start_bot():
    global bot, dp
    logger.info("Initializing NEXUS Telegram Bot (aiogram v3)...")

    bot = Bot(
        token=settings.TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()

    # Register Access Control Middleware
    dp.message.middleware(AccessMiddleware())
    dp.callback_query.middleware(AccessMiddleware())

    # Register Handlers Router
    dp.include_router(router)

    # Start follow-up scheduler as background task
    from telegram.followup import run_followup_scheduler
    asyncio.create_task(run_followup_scheduler(bot), name="followup_scheduler")

    # Start polling
    logger.info("Starting bot polling event loop...")
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    except Exception as e:
        logger.exception("Fatal error in aiogram polling: %s", e)

async def stop_bot():
    global bot, dp
    if bot:
        logger.info("Stopping bot connection...")
        await bot.session.close()
        logger.info("Bot session closed successfully.")
