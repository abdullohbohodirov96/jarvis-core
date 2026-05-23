import logging
from typing import Any, Awaitable, Callable, Optional
from aiogram import Bot, Dispatcher, Router, F, BaseMiddleware
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    Message,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    WebAppInfo,
    TelegramObject
)
from aiogram.client.default import DefaultBotProperties

from core.config import settings
from database.connection import AsyncSessionLocal
import database.operations as db_ops

logger = logging.getLogger(__name__)

bot: Optional[Bot] = None
dp: Optional[Dispatcher] = None
router = Router(name="nexus_bot_router")

# ── 1. Access Control Middleware ──

class AccessMiddleware(BaseMiddleware):
    """
    Middleware that checks user registration and access control status.
    Automatically registers new users in pending status.
    """
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not isinstance(event, Message):
            return await handler(event, data)

        message: Message = event
        tg_user = message.from_user
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
                    if message.text and message.text.startswith("/start"):
                        await message.answer(
                            "⛔️ <b>Доступ в NEXUS ограничен.</b>\n\n"
                            "Ваша заявка на регистрацию отправлена на модерацию администраторам.\n"
                            "Ожидайте одобрения доступа.",
                            parse_mode=ParseMode.HTML
                        )
                    else:
                        await message.answer("🔒 Доступ закрыт. Ожидайте одобрения администратором.")
                    return  # Terminate pipeline

                # Inject user and db session into the handler context
                data["db_session"] = db
                data["db_user"] = user
                return await handler(event, data)

            except Exception as e:
                logger.error(f"Error in AccessMiddleware for {telegram_id}: {e}")
                await db.rollback()
                await message.answer("⚠️ Произошла системная ошибка авторизации. Попробуйте позже.")
                return

# ── 2. Commands & Handlers ──

@router.message(Command("start"))
async def cmd_start(message: Message, db_user):
    name = message.from_user.first_name
    admin_badge = " [Администратор]" if db_user.is_admin else ""
    
    # Inline keyboard with WebAppInfo
    builder = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(
                text="🚀 Открыть NEXUS App",
                web_app=WebAppInfo(url=settings.MINI_APP_URL)
            )
        ]
    ])
    
    welcome_text = (
        f"👋 Привет, <b>{name}</b>!{admin_badge}\n\n"
        f"Добро пожаловать в <b>NEXUS</b> — вашу личную платформу управления интеграциями и задачами.\n\n"
        f"Все управление осуществляется через графический интерфейс в Telegram Mini App.\n"
        f"Нажмите кнопку ниже, чтобы войти в приложение! 👇"
    )
    await message.answer(welcome_text, parse_mode=ParseMode.HTML, reply_markup=builder)

@router.message(Command("allow"))
async def cmd_allow(message: Message, command: CommandObject, db_session, db_user):
    if not db_user.is_admin:
        await message.answer("❌ Эта команда доступна только администраторам.")
        return
        
    username = command.args
    if not username:
        await message.answer("⚠️ Укажите юзернейм пользователя. Пример: <code>/allow @username</code>", parse_mode=ParseMode.HTML)
        return
        
    user = await db_ops.allow_user(db_session, username)
    if not user:
        await message.answer(f"❌ Пользователь с юзернеймом {username} не найден в базе данных (он должен хотя бы раз запустить бота).")
        return
        
    await db_session.commit()
    await message.answer(f"✅ Доступ для пользователя <b>{username}</b> успешно одобрен! 🚀", parse_mode=ParseMode.HTML)
    
    # Notify whitelisted user if possible
    try:
        await bot.send_message(
            chat_id=user.tg_id,
            text="🎉 <b>Поздравляем!</b> Ваш доступ к платформе NEXUS успешно одобрен администратором.\n"
                 "Используйте команду /start для входа в приложение!",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logger.warning(f"Could not notify user {user.tg_id} about approval: {e}")

@router.message(Command("revoke"))
async def cmd_revoke(message: Message, command: CommandObject, db_session, db_user):
    if not db_user.is_admin:
        await message.answer("❌ Эта команда доступна только администраторам.")
        return
        
    username = command.args
    if not username:
        await message.answer("⚠️ Укажите юзернейм пользователя. Пример: <code>/revoke @username</code>", parse_mode=ParseMode.HTML)
        return
        
    user = await db_ops.revoke_user(db_session, username)
    if not user:
        await message.answer(f"❌ Пользователь с юзернеймом {username} не найден.")
        return
        
    await db_session.commit()
    await message.answer(f"❌ Доступ для пользователя <b>{username}</b> аннулирован.", parse_mode=ParseMode.HTML)
    
    # Notify user if possible
    try:
        await bot.send_message(
            chat_id=user.tg_id,
            text="⚠️ Администратор отозвал ваш доступ к платформе NEXUS.",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logger.warning(f"Could not notify user {user.tg_id} about revocation: {e}")

@router.message(Command("users"))
async def cmd_users(message: Message, db_session, db_user):
    if not db_user.is_admin:
        await message.answer("❌ Эта команда доступна только администраторам.")
        return
        
    pending = await db_ops.get_pending_users(db_session)
    if not pending:
        await message.answer("👥 Нет пользователей, ожидающих одобрения доступа.")
        return
        
    text = "⏳ <b>Пользователи, ожидающие одобрения:</b>\n\n"
    for idx, u in enumerate(pending, 1):
        username_str = f"@{u.username}" if u.username else f"ID: {u.tg_id}"
        text += f"{idx}. {username_str} (заявка от {u.created_at.strftime('%d.%m.%Y %H:%M')})\n"
        
    text += "\nИспользуйте команду <code>/allow @username</code> для выдачи доступа."
    await message.answer(text, parse_mode=ParseMode.HTML)

# ── 3. Lifecycle functions ──

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
    
    # Register Handlers Router
    dp.include_router(router)
    
    # Start polling
    logger.info("Starting bot polling event loop...")
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    except Exception as e:
        logger.exception("Fatal error in aiogram polling: {}", e)

async def stop_bot():
    global bot, dp
    if bot:
        logger.info("Stopping bot connection...")
        await bot.session.close()
        logger.info("Bot session closed successfully.")
