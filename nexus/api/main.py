import asyncio
import logging
import sys
import time
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any, Optional
from datetime import datetime

from fastapi import FastAPI, Header, HTTPException, Query, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from database.connection import get_db, init_db, check_db_connection
import database.operations as db_ops
from telegram.userbot import NexusUserbot
from utils.logger import logger

_start_time: float = time.monotonic()
_db_ready: bool = False
_telegram_ready: bool = False
_telegram_task: Optional[asyncio.Task] = None

# ── Marketplace Catalog Definition ──
MARKETPLACE_CATALOG = [
    {
        "id": "todoist",
        "name": "Todoist",
        "description": "Синхронизируйте ваши личные задачи с популярным трекером Todoist в реальном времени.",
        "icon": "📝",
        "category": "productivity",
        "requires_config": True,
        "schema": [
            {"name": "todoist_token", "label": "API Токен (Todoist Key)", "placeholder": "Вставьте ваш API токен из настроек Todoist...", "secret": True}
        ]
    },
    {
        "id": "telegram_bot",
        "name": "Telegram Bot",
        "description": "Подключите собственного Telegram-бота для трансляции уведомлений и команд управления через NEXUS.",
        "icon": "🤖",
        "category": "automation",
        "requires_config": True,
        "schema": [
            {"name": "bot_token", "label": "Telegram Bot Token", "placeholder": "1234567890:AABBcc...", "secret": True}
        ]
    },
    {
        "id": "instagram",
        "name": "Instagram",
        "description": "Планирование постов, сбор аналитики и автоматический постинг через Graph API.",
        "icon": "📸",
        "category": "social",
        "requires_config": True,
        "schema": [
            {"name": "access_token", "label": "Access Token", "placeholder": "IGQVJx...", "secret": True},
            {"name": "account_id", "label": "Instagram Business Account ID", "placeholder": "17841400...", "secret": False}
        ]
    },
    {
        "id": "google_sheets",
        "name": "Google Sheets",
        "description": "Экспорт задач, сбор отчетов и логирование активности в Google Таблицы.",
        "icon": "📊",
        "category": "data",
        "requires_config": True,
        "schema": [
            {"name": "spreadsheet_id", "label": "Spreadsheet ID", "placeholder": "1z8v...", "secret": False},
            {"name": "sheet_name", "label": "Sheet Name", "placeholder": "Задачи", "secret": False}
        ]
    }
]

# ── API Models ──

class ConnectRequest(BaseModel):
    tg_id: int
    app_id: str
    credentials: dict[str, Any]

class ToggleRequest(BaseModel):
    tg_id: int
    app_id: str
    is_active: bool

class DisconnectRequest(BaseModel):
    tg_id: int
    app_id: str

class TodoAddRequest(BaseModel):
    tg_id: int
    title: str = Field(..., min_length=1)
    description: Optional[str] = None
    priority: str = "medium"
    due_date: Optional[str] = None  # ISO format string
    section: str = "vazifalar"

class MoveSectionRequest(BaseModel):
    tg_id: int
    section: str  # "vazifalar" | "kutilmoqda" | "keraklilar" | "bajarildi"

class TodoUpdateRequest(BaseModel):
    tg_id: int
    title: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[str] = None
    due_date: Optional[str] = None # ISO format string or empty string to unset

# Userbot API Models
class UserbotConnectRequest(BaseModel):
    tg_id: int
    phone: str

class UserbotVerifyRequest(BaseModel):
    tg_id: int
    phone: str
    phone_code_hash: str
    code: str
    password: Optional[str] = None

class ToggleChatRequest(BaseModel):
    tg_id: int
    chat_id: int
    is_active: bool

# ── Lifespan for FastAPI ──

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _db_ready, _telegram_ready, _telegram_task

    logger.info("Starting {} (env={})", settings.PROJECT_NAME, settings.ENVIRONMENT)

    # Initialize Database
    try:
        await init_db()
        _db_ready = True
        logger.success("Database initialized and ready.")
    except Exception as exc:
        logger.error("Database init failed: {}", exc)

    # Initialize Telegram Bot (Aiogram daemon)
    try:
        from telegram.bot import start_bot, stop_bot
        _telegram_task = asyncio.create_task(
            start_bot(),
            name="telegram_bot_daemon",
        )
        def _on_done(fut: asyncio.Future) -> None:
            global _telegram_ready
            if not fut.cancelled() and fut.exception():
                _telegram_ready = False
                logger.error("Telegram bot daemon crashed: {}", fut.exception())
        _telegram_task.add_done_callback(_on_done)
        _telegram_ready = True
        logger.success("Telegram bot daemon started concurrently.")
    except Exception as exc:
        logger.error("Telegram bot init failed: {}", exc)

    # Initialize Telegram Userbot (Telethon listener)
    try:
        userbot = NexusUserbot()
        if await userbot.is_connected():
            asyncio.create_task(userbot.start_listening())
            logger.success("Userbot connected and listener active on startup.")
    except Exception as exc:
        logger.error("Userbot startup failed: {}", exc)

    logger.success("{} startup complete.", settings.PROJECT_NAME)

    yield

    logger.info("Shutting down {}…", settings.PROJECT_NAME)
    from telegram.bot import stop_bot
    await stop_bot()
    
    # Disconnect Userbot
    try:
        await NexusUserbot().disconnect()
    except Exception:
        pass

    if _telegram_task and not _telegram_task.done():
        _telegram_task.cancel()
        try:
            await _telegram_task
        except asyncio.CancelledError:
            pass
    logger.info("Shutdown complete.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="2.0.0",
    description="NEXUS SaaS Platform Backend (FastAPI + Supabase/PostgreSQL + Aiogram v3 Bot)",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to enforce authorization check
async def get_authorized_user(tg_id: int, db: AsyncSession):
    user = await db_ops.get_user(db, tg_id)
    if not user:
        # Create a pending registration request so admin can approve it
        user = await db_ops.get_or_create_user(db, tg_id=tg_id, is_owner=(tg_id == settings.OWNER_ID))
        await db.commit()
    
    if not user.is_allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Доступ ограничен. Обратитесь к администратору для одобрения заявки."
        )
    return user

# ── 1. Authentication Check ──

@app.get("/api/auth/check")
async def auth_check(
    tg_id: int = Query(..., description="Telegram ID of the user"),
    username: Optional[str] = Query(None, description="Telegram username of the user"),
    db: AsyncSession = Depends(get_db)
):
    try:
        user = await db_ops.get_user(db, tg_id)
        if not user:
            # First-time registration request
            user = await db_ops.get_or_create_user(db, tg_id=tg_id, username=username, is_owner=(tg_id == settings.OWNER_ID))
            await db.commit()
            
        if not user.is_allowed:
            return JSONResponse(
                content={
                    "allowed": False,
                    "registered": True,
                    "is_admin": False,
                    "message": "Доступ заблокирован или ожидает одобрения администратора."
                },
                status_code=403
            )
            
        return {
            "allowed": True,
            "registered": True,
            "is_admin": user.is_admin,
            "username": user.username,
            "message": "Доступ разрешен."
        }
    except Exception as exc:
        logger.exception("Auth check endpoint failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))

# ── 2. Marketplace Catalog ──

@app.get("/api/market/integrations")
async def get_integrations(
    tg_id: int = Query(..., description="Telegram ID of the user"),
    db: AsyncSession = Depends(get_db)
):
    await get_authorized_user(tg_id, db)
    try:
        user_integrations = await db_ops.get_user_integrations(db, tg_id)
        connected_map = {ui.app_id: ui for ui in user_integrations}
        
        result = []
        for app in MARKETPLACE_CATALOG:
            app_id = app["id"]
            is_connected = app_id in connected_map
            is_active = connected_map[app_id].is_active if is_connected else False
            
            # Decrypt credentials but mask secret fields
            masked_creds = {}
            if is_connected:
                raw_creds = db_ops.decrypt_credentials(connected_map[app_id].credentials, app["schema"])
                for item in app["schema"]:
                    name = item["name"]
                    val = raw_creds.get(name, "")
                    if item.get("secret", True) and val:
                        masked_creds[name] = "********"
                    else:
                        masked_creds[name] = val
            
            result.append({
                **app,
                "connected": is_connected,
                "active": is_active,
                "credentials": masked_creds
            })
            
        return result
    except Exception as exc:
        logger.exception("Market integrations catalog failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))

# ── 3. Marketplace Connections ──

@app.post("/api/market/connect")
async def connect_integration(req: ConnectRequest, db: AsyncSession = Depends(get_db)):
    await get_authorized_user(req.tg_id, db)
    try:
        # Validate app_id
        valid_apps = {app["id"] for app in MARKETPLACE_CATALOG}
        if req.app_id not in valid_apps:
            raise HTTPException(status_code=400, detail=f"Недопустимый ID приложения: {req.app_id}")
            
        integration = await db_ops.connect_integration(db, req.tg_id, req.app_id, req.credentials)
        await db.commit()
        return {
            "success": True,
            "message": f"Приложение {req.app_id} успешно подключено.",
            "active": integration.is_active
        }
    except Exception as exc:
        logger.exception("Market connect failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@app.patch("/api/market/toggle")
async def toggle_integration(req: ToggleRequest, db: AsyncSession = Depends(get_db)):
    await get_authorized_user(req.tg_id, db)
    try:
        integration = await db_ops.toggle_integration(db, req.tg_id, req.app_id, req.is_active)
        if not integration:
            raise HTTPException(status_code=404, detail="Интеграция не найдена. Сначала подключите её.")
        await db.commit()
        return {
            "success": True,
            "active": integration.is_active,
            "message": f"Состояние интеграции {req.app_id} обновлено."
        }
    except Exception as exc:
        logger.exception("Market toggle failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@app.delete("/api/market/disconnect")
async def disconnect_integration(
    tg_id: int = Query(...),
    app_id: str = Query(...),
    db: AsyncSession = Depends(get_db)
):
    await get_authorized_user(tg_id, db)
    try:
        success = await db_ops.disconnect_integration(db, tg_id, app_id)
        if not success:
            raise HTTPException(status_code=404, detail="Интеграция не найдена.")
        await db.commit()
        return {
            "success": True,
            "message": f"Интеграция {app_id} успешно удалена."
        }
    except Exception as exc:
        logger.exception("Market disconnect failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))

# ── 4. To-Do Endpoints ──

@app.get("/api/todos")
async def get_todos(
    tg_id: int = Query(..., description="Telegram ID of the user"),
    db: AsyncSession = Depends(get_db)
):
    await get_authorized_user(tg_id, db)
    try:
        todos = await db_ops.get_todos(db, tg_id)
        return [
            {
                "id": todo.id,
                "title": todo.title,
                "description": todo.description,
                "is_done": todo.is_done,
                "section": getattr(todo, 'section', 'vazifalar'),
                "source": getattr(todo, 'source', 'manual'),
                "from_chat_name": getattr(todo, 'from_chat_name', None),
                "priority": todo.priority,
                "due_date": todo.due_date.isoformat() if todo.due_date else None,
                "follow_up_at": todo.follow_up_at.isoformat() if getattr(todo, 'follow_up_at', None) else None,
                "created_at": todo.created_at.isoformat()
            }
            for todo in todos
        ]
    except Exception as exc:
        logger.exception("Get todos failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/todos/add")
async def add_todo(req: TodoAddRequest, db: AsyncSession = Depends(get_db)):
    await get_authorized_user(req.tg_id, db)
    try:
        due = None
        if req.due_date:
            due = datetime.fromisoformat(req.due_date.replace("Z", "+00:00"))
            
        todo = await db_ops.add_todo(
            db,
            tg_id=req.tg_id,
            title=req.title,
            description=req.description,
            priority=req.priority,
            due_date=due,
            section=req.section,
        )
        await db.commit()
        return {
            "success": True,
            "todo": {
                "id": todo.id,
                "title": todo.title,
                "description": todo.description,
                "is_done": todo.is_done,
                "section": getattr(todo, 'section', 'vazifalar'),
                "source": getattr(todo, 'source', 'manual'),
                "from_chat_name": getattr(todo, 'from_chat_name', None),
                "priority": todo.priority,
                "due_date": todo.due_date.isoformat() if todo.due_date else None,
                "follow_up_at": todo.follow_up_at.isoformat() if getattr(todo, 'follow_up_at', None) else None,
                "created_at": todo.created_at.isoformat()
            }
        }
    except Exception as exc:
        logger.exception("Add todo failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@app.patch("/api/todos/{id}")
async def update_todo(id: int, req: TodoUpdateRequest, db: AsyncSession = Depends(get_db)):
    await get_authorized_user(req.tg_id, db)
    try:
        updates = {}
        if req.title is not None:
            updates["title"] = req.title
        if req.description is not None:
            updates["description"] = req.description
        if req.priority is not None:
            updates["priority"] = req.priority
        if req.due_date is not None:
            if req.due_date == "":
                updates["due_date"] = None
            else:
                updates["due_date"] = datetime.fromisoformat(req.due_date.replace("Z", "+00:00"))
                
        todo = await db_ops.update_todo(db, tg_id=req.tg_id, todo_id=id, updates=updates)
        if not todo:
            raise HTTPException(status_code=404, detail="Задача не найдена.")
            
        await db.commit()
        return {
            "success": True,
            "todo": {
                "id": todo.id,
                "title": todo.title,
                "description": todo.description,
                "is_done": todo.is_done,
                "priority": todo.priority,
                "due_date": todo.due_date.isoformat() if todo.due_date else None
            }
        }
    except Exception as exc:
        logger.exception("Update todo failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/todos/{id}/complete")
async def complete_todo(
    id: int,
    tg_id: int = Query(..., description="Telegram ID of the user"),
    is_done: bool = Query(True),
    db: AsyncSession = Depends(get_db)
):
    await get_authorized_user(tg_id, db)
    try:
        todo = await db_ops.complete_todo(db, tg_id=tg_id, todo_id=id, is_done=is_done)
        if not todo:
            raise HTTPException(status_code=404, detail="Задача не найдена.")
        await db.commit()
        return {
            "success": True,
            "is_done": todo.is_done
        }
    except Exception as exc:
        logger.exception("Complete todo failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/todos/{id}/move-section")
async def move_todo_to_section(id: int, req: MoveSectionRequest, db: AsyncSession = Depends(get_db)):
    await get_authorized_user(req.tg_id, db)
    valid_sections = {"vazifalar", "kutilmoqda", "keraklilar", "bajarildi"}
    if req.section not in valid_sections:
        raise HTTPException(status_code=400, detail="Invalid section")
    try:
        todo = await db_ops.move_todo_section(db, req.tg_id, id, req.section)
        if not todo:
            raise HTTPException(status_code=404, detail="Task not found")
        await db.commit()
        return {"success": True, "section": todo.section}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.delete("/api/todos/{id}")
async def delete_todo(
    id: int,
    tg_id: int = Query(..., description="Telegram ID of the user"),
    db: AsyncSession = Depends(get_db)
):
    await get_authorized_user(tg_id, db)
    try:
        success = await db_ops.delete_todo(db, tg_id=tg_id, todo_id=id)
        if not success:
            raise HTTPException(status_code=404, detail="Задача не найдена.")
        await db.commit()
        return {
            "success": True,
            "message": "Задача успешно удалена."
        }
    except Exception as exc:
        logger.exception("Delete todo failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))

# ── 5. Userbot Controls API ──

@app.get("/api/telegram/status")
async def get_userbot_status(
    tg_id: int = Query(..., description="Telegram ID of the user"),
    db: AsyncSession = Depends(get_db)
):
    user = await get_authorized_user(tg_id, db)
    userbot = NexusUserbot()
    is_ok = await userbot.is_connected()
    
    me_info = None
    if is_ok:
        me = await userbot.get_me()
        if me:
            me_info = {
                "id": me.id,
                "first_name": me.first_name,
                "last_name": me.last_name,
                "username": me.username,
                "phone": me.phone
            }
            
    return {
        "connected": is_ok,
        "me": me_info,
        "analyzed_chats": user.analyzed_chats or []
    }

@app.post("/api/telegram/connect")
async def connect_userbot(req: UserbotConnectRequest, db: AsyncSession = Depends(get_db)):
    await get_authorized_user(req.tg_id, db)
    try:
        userbot = NexusUserbot()
        phone_code_hash = await userbot.send_code(req.phone)
        return {
            "success": True,
            "phone_code_hash": phone_code_hash,
            "message": "Код подтверждения отправлен в ваш аккаунт Telegram."
        }
    except Exception as exc:
        logger.exception("Userbot send code failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/telegram/verify")
async def verify_userbot(req: UserbotVerifyRequest, db: AsyncSession = Depends(get_db)):
    await get_authorized_user(req.tg_id, db)
    try:
        userbot = NexusUserbot()
        success = await userbot.verify_code(
            phone=req.phone,
            phone_code_hash=req.phone_code_hash,
            code=req.code,
            password=req.password
        )
        return {
            "success": success,
            "message": "Аккаунт Telegram успешно подключен к NEXUS!"
        }
    except Exception as exc:
        logger.exception("Userbot verify code failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@app.delete("/api/telegram/disconnect")
async def disconnect_userbot(
    tg_id: int = Query(...),
    db: AsyncSession = Depends(get_db)
):
    await get_authorized_user(tg_id, db)
    try:
        userbot = NexusUserbot()
        await userbot.disconnect()
        return {
            "success": True,
            "message": "Аккаунт Telegram отключен."
        }
    except Exception as exc:
        logger.exception("Userbot disconnect failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/telegram/chats")
async def get_userbot_chats(
    tg_id: int = Query(..., description="Telegram ID of the user"),
    db: AsyncSession = Depends(get_db)
):
    await get_authorized_user(tg_id, db)
    try:
        userbot = NexusUserbot()
        chats = await userbot.list_chats()
        return chats
    except Exception as exc:
        logger.exception("Get userbot chats failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/telegram/toggle-chat")
async def toggle_chat_analysis(req: ToggleChatRequest, db: AsyncSession = Depends(get_db)):
    await get_authorized_user(req.tg_id, db)
    try:
        updated_chats = await db_ops.toggle_chat_analysis(
            db=db,
            tg_id=req.tg_id,
            chat_id=req.chat_id,
            is_active=req.is_active
        )
        await db.commit()
        return {
            "success": True,
            "analyzed_chats": updated_chats,
            "message": "Статус анализа чата изменен."
        }
    except Exception as exc:
        logger.exception("Toggle chat analysis failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))

# ── 6. System Health Check ──

@app.get("/health", tags=["System"], summary="Health check")
async def health() -> JSONResponse:
    db_ok: bool = False
    try:
        db_ok = await check_db_connection()
    except Exception:
        pass

    uptime = round(time.monotonic() - _start_time, 2)
    overall = "ok" if (db_ok and _telegram_ready) else "degraded"

    return JSONResponse(
        content={
            "status": overall,
            "project": settings.PROJECT_NAME,
            "env": settings.ENVIRONMENT,
            "uptime_s": uptime,
            "database": "ok" if db_ok else "unreachable",
            "telegram": "ok" if _telegram_ready else "unavailable",
        },
        status_code=200 if overall == "ok" else 500,
    )

# ── 7. Static Files Server ──

static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
@app.get("/app")
async def serve_app():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "NEXUS Platform API is running. Mini App static directory not found."}
