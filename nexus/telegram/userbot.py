import logging
import asyncio
import os
import random
from datetime import timedelta, timezone as tz
from typing import Any, Optional, Union
from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError
from telethon.tl.types import User, Chat, Channel

from core.config import settings
from database.connection import AsyncSessionLocal
import database.operations as db_ops
from ai.analyzer import NexusAnalyzer

logger = logging.getLogger(__name__)

# Global userbot instance
userbot_client: Optional[TelegramClient] = None
userbot_task: Optional[asyncio.Task] = None
is_listener_running = False

class NexusUserbot:
    """Manages the MTProto Telethon Userbot for the owner's personal Telegram account."""
    
    def __init__(self) -> None:
        self.session_path = "/tmp/nexus_userbot_session"
        self.api_id = settings.TELEGRAM_API_ID
        self.api_hash = settings.TELEGRAM_API_HASH
        
    def get_client(self) -> TelegramClient:
        global userbot_client
        if userbot_client is None:
            userbot_client = TelegramClient(
                session=self.session_path,
                api_id=self.api_id,
                api_hash=self.api_hash
            )
        return userbot_client

    async def is_connected(self) -> bool:
        client = self.get_client()
        try:
            if not client.is_connected():
                await client.connect()
            return await client.is_user_authorized()
        except Exception as e:
            logger.error(f"Userbot connection check failed: {e}")
            return False

    async def send_code(self, phone: str) -> str:
        """Send login code to the phone number. Returns the phone_code_hash."""
        client = self.get_client()
        if not client.is_connected():
            await client.connect()
        result = await client.send_code_request(phone)
        return result.phone_code_hash

    async def verify_code(
        self,
        phone: str,
        phone_code_hash: str,
        code: str,
        password: Optional[str] = None
    ) -> bool:
        """Verify the code and log in. Handles 2FA if password is provided."""
        client = self.get_client()
        if not client.is_connected():
            await client.connect()
            
        try:
            await client.sign_in(phone=phone, code=code, phone_code_hash=phone_code_hash)
            logger.info("Userbot successfully logged in with SMS code.")
            # Start background listener
            asyncio.create_task(self.start_listening())
            return True
        except SessionPasswordNeededError:
            if password:
                await client.sign_in(password=password)
                logger.info("Userbot successfully logged in with 2FA password.")
                # Start background listener
                asyncio.create_task(self.start_listening())
                return True
            else:
                logger.info("2FA password is required for this account.")
                raise SessionPasswordNeededError("2FA required")
        except Exception as e:
            logger.error(f"Failed to verify code: {e}")
            raise e

    async def disconnect(self) -> None:
        global userbot_client, is_listener_running
        if userbot_client and userbot_client.is_connected():
            await userbot_client.disconnect()
            is_listener_running = False
            logger.info("Userbot disconnected.")

    async def get_me(self) -> Optional[User]:
        client = self.get_client()
        if await self.is_connected():
            return await client.get_me()
        return None

    async def list_chats(self) -> list[dict[str, Any]]:
        """List recent dialogs (chats) to select which ones to analyze."""
        client = self.get_client()
        if not await self.is_connected():
            return []
            
        chats = []
        async for dialog in client.iter_dialogs(limit=30):
            # Skip empty or channel chats if necessary, but keep private/groups
            entity = dialog.entity
            chat_type = "private"
            if isinstance(entity, Chat):
                chat_type = "group"
            elif isinstance(entity, Channel):
                chat_type = "channel"
                
            chats.append({
                "id": dialog.id,
                "name": dialog.name or "Unknown Chat",
                "type": chat_type,
                "unread_count": dialog.unread_count,
                "username": getattr(entity, "username", None)
            })
        return chats

    async def start_listening(self) -> None:
        """Start listening for incoming/outgoing messages in whitelisted chats."""
        global is_listener_running, userbot_client
        if is_listener_running:
            return
            
        client = self.get_client()
        if not await self.is_connected():
            logger.warning("Cannot start listening, Userbot not authorized.")
            return

        is_listener_running = True
        logger.info("🟢 Starting Userbot background conversation monitoring daemon...")
        analyzer = NexusAnalyzer()

        @client.on(events.NewMessage())
        async def on_new_message(event: events.NewMessage.Event):
            try:
                # Get chat id
                chat_id = event.chat_id
                
                # Fetch whitelisted chats for the owner
                async with AsyncSessionLocal() as db:
                    owner = await db_ops.get_user(db, settings.OWNER_ID)
                    if not owner or not owner.is_allowed:
                        return
                        
                    analyzed_chats = owner.analyzed_chats or []
                    # Check if this chat_id is in analyzed_chats list
                    if chat_id not in analyzed_chats:
                        return
                        
                    # Extract sender name
                    sender = await event.get_sender()
                    sender_name = "Unknown"
                    if isinstance(sender, User):
                        sender_name = sender.first_name + (f" {sender.last_name}" if sender.last_name else "")
                        if sender.username:
                            sender_name += f" (@{sender.username})"
                            
                    # Context / Chat Title
                    chat = await event.get_chat()
                    chat_title = getattr(chat, "title", "Private Chat")
                    
                    # Analyze message using AI with dialogue context history (last 5 messages)
                    is_outgoing = event.out
                    message_text = event.message.message or ""
                    
                    # Fetch dialogue history context
                    history_text = ""
                    try:
                        history_messages = []
                        async for msg in client.iter_messages(chat_id, limit=5):
                            if msg.message:
                                history_messages.append(msg)
                        
                        history_messages.reverse()
                        
                        for msg in history_messages:
                            msg_sender = await msg.get_sender()
                            msg_sender_name = "Unknown"
                            if isinstance(msg_sender, User):
                                msg_sender_name = msg_sender.first_name + (f" {msg_sender.last_name}" if msg_sender.last_name else "")
                                if msg_sender.username:
                                    msg_sender_name += f" (@{msg_sender.username})"
                            elif msg_sender:
                                msg_sender_name = getattr(msg_sender, "title", "Chat")
                                
                            role = "Siz (egasi)" if msg.out else f"Suhbatdosh '{msg_sender_name}'"
                            history_text += f"- [{role}]: \"{msg.message}\"\n"
                    except Exception as e:
                        logger.warning(f"Failed to fetch conversation history context: {e}")

                    logger.info(f"Analyzing message in chat {chat_title} from {sender_name}: '{message_text[:40]}...'")
                    
                    task_data = await analyzer.analyze_message(
                        sender_name=sender_name,
                        message_text=message_text,
                        chat_title=chat_title,
                        is_outgoing=is_outgoing,
                        history_text=history_text
                    )
                    
                    if task_data:
                        # We extracted a task! Save to database!
                        from datetime import datetime
                        section = "kutilmoqda" if is_outgoing else "vazifalar"
                        source = "promise" if is_outgoing else "request"
                        follow_up_hours = random.uniform(3, 5)
                        follow_up_time = datetime.now(tz.utc) + timedelta(hours=follow_up_hours)

                        todo = await db_ops.add_todo(
                            db=db,
                            tg_id=settings.OWNER_ID,
                            title=task_data["title"],
                            description=task_data["description"],
                            priority=task_data["priority"],
                            due_date=None,
                            section=section,
                            source=source,
                            from_chat_id=chat_id,
                            from_chat_name=chat_title,
                            original_message=message_text[:500],
                            follow_up_at=follow_up_time,
                        )
                        await db.commit()
                        logger.info(f"Successfully auto-extracted task from chat: '{todo.title}'")

                        # Notify the owner via our Main Aiogram Bot!
                        try:
                            from telegram.bot import bot, update_pinned_board
                            if bot:
                                badge = "🔴 <b>Sizning va'dangiz aniqlandi:</b>" if is_outgoing else "🔵 <b>Sizga so'rov yuborildi:</b>"
                                notify_text = (
                                    f"{badge}\n\n"
                                    f"📝 <b>{todo.title}</b>\n"
                                    f"📋 {todo.description}\n\n"
                                    f"<i>Men ushbu vazifani NEXUS ro'yxatingizga avtomatik ravishda qo'shib qo'ydim.</i>"
                                )
                                await bot.send_message(
                                    chat_id=settings.OWNER_ID,
                                    text=notify_text,
                                    parse_mode="HTML"
                                )
                                from database.connection import AsyncSessionLocal
                                async with AsyncSessionLocal() as board_db:
                                    await update_pinned_board(bot, settings.OWNER_ID, board_db)
                        except Exception as notify_err:
                            logger.warning(f"Failed to send task notification: {notify_err}")
            except Exception as e:
                logger.error(f"Error in Userbot message listener: {e}")

        logger.info("Userbot listener active and registered.")
