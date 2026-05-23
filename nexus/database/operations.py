import logging
from datetime import datetime
from typing import Any
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import User, UserIntegration, Todo
from database.crypto import encrypt, decrypt

logger = logging.getLogger(__name__)

# ── Credentials Encryption helpers ──

def encrypt_credentials(credentials: dict[str, Any]) -> dict[str, str]:
    """Encrypt all secret string credentials in the dictionary."""
    encrypted = {}
    for k, v in credentials.items():
        if isinstance(v, str):
            encrypted[k] = encrypt(v)
        else:
            encrypted[k] = v
    return encrypted

def decrypt_credentials(credentials: dict[str, Any], schema: list[dict] | None = None) -> dict[str, Any]:
    """
    Decrypt values from the credentials dictionary.
    If a schema is provided, only decrypt fields marked as secret.
    """
    decrypted = {}
    for k, v in credentials.items():
        if isinstance(v, str):
            # Check if this field should be secret (masked in API response)
            is_secret = True
            if schema:
                for item in schema:
                    if item.get("name") == k:
                        is_secret = item.get("secret", True)
                        break
            # We decrypt it
            decrypted[k] = decrypt(v)
        else:
            decrypted[k] = v
    return decrypted

# ── User Operations ──

async def get_user(db: AsyncSession, tg_id: int) -> User | None:
    result = await db.execute(select(User).where(User.tg_id == tg_id))
    return result.scalar_one_or_none()

async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
    # Strip @ if present
    username = username.lstrip("@")
    result = await db.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()

async def get_or_create_user(db: AsyncSession, tg_id: int, username: str | None = None, is_owner: bool = False) -> User:
    user = await get_user(db, tg_id)
    if not user:
        user = User(
            tg_id=tg_id,
            username=username.lstrip("@") if username else None,
            is_allowed=is_owner, # Owner allowed immediately
            is_admin=is_owner
        )
        db.add(user)
        await db.flush()
    else:
        # Update username if changed
        if username and user.username != username.lstrip("@"):
            user.username = username.lstrip("@")
            await db.flush()
    return user

async def allow_user(db: AsyncSession, username: str) -> User | None:
    user = await get_user_by_username(db, username)
    if user:
        user.is_allowed = True
        await db.flush()
    return user

async def revoke_user(db: AsyncSession, username: str) -> User | None:
    user = await get_user_by_username(db, username)
    if user:
        user.is_allowed = False
        user.is_admin = False
        await db.flush()
    return user

async def get_pending_users(db: AsyncSession) -> list[User]:
    result = await db.execute(select(User).where(User.is_allowed == False).order_by(User.created_at.desc()))
    return list(result.scalars().all())

# ── Integration Operations ──

async def get_user_integrations(db: AsyncSession, tg_id: int) -> list[UserIntegration]:
    result = await db.execute(select(UserIntegration).where(UserIntegration.tg_id == tg_id))
    return list(result.scalars().all())

async def connect_integration(db: AsyncSession, tg_id: int, app_id: str, credentials: dict) -> UserIntegration:
    # Encrypt creds first
    encrypted_creds = encrypt_credentials(credentials)
    
    # Check if exists
    result = await db.execute(
        select(UserIntegration).where(
            UserIntegration.tg_id == tg_id,
            UserIntegration.app_id == app_id
        )
    )
    integration = result.scalar_one_or_none()
    
    if integration:
        integration.credentials = encrypted_creds
        integration.is_active = True
    else:
        integration = UserIntegration(
            tg_id=tg_id,
            app_id=app_id,
            credentials=encrypted_creds,
            is_active=True
        )
        db.add(integration)
    
    await db.flush()
    return integration

async def toggle_integration(db: AsyncSession, tg_id: int, app_id: str, is_active: bool) -> UserIntegration | None:
    result = await db.execute(
        select(UserIntegration).where(
            UserIntegration.tg_id == tg_id,
            UserIntegration.app_id == app_id
        )
    )
    integration = result.scalar_one_or_none()
    if integration:
        integration.is_active = is_active
        await db.flush()
    return integration

async def disconnect_integration(db: AsyncSession, tg_id: int, app_id: str) -> bool:
    result = await db.execute(
        select(UserIntegration).where(
            UserIntegration.tg_id == tg_id,
            UserIntegration.app_id == app_id
        )
    )
    integration = result.scalar_one_or_none()
    if integration:
        await db.delete(integration)
        await db.flush()
        return True
    return False

# ── Todo Operations ──

async def get_todos(db: AsyncSession, tg_id: int) -> list[Todo]:
    result = await db.execute(select(Todo).where(Todo.tg_id == tg_id).order_by(Todo.created_at.desc()))
    return list(result.scalars().all())

async def get_todo(db: AsyncSession, tg_id: int, todo_id: int) -> Todo | None:
    result = await db.execute(
        select(Todo).where(
            Todo.tg_id == tg_id,
            Todo.id == todo_id
        )
    )
    return result.scalar_one_or_none()

async def add_todo(
    db: AsyncSession,
    tg_id: int,
    title: str,
    description: str | None = None,
    priority: str = "medium",
    due_date: datetime | None = None
) -> Todo:
    todo = Todo(
        tg_id=tg_id,
        title=title,
        description=description,
        priority=priority,
        due_date=due_date
    )
    db.add(todo)
    await db.flush()
    return todo

async def update_todo(db: AsyncSession, tg_id: int, todo_id: int, updates: dict[str, Any]) -> Todo | None:
    todo = await get_todo(db, tg_id, todo_id)
    if todo:
        for k, v in updates.items():
            if hasattr(todo, k):
                setattr(todo, k, v)
        await db.flush()
    return todo

async def complete_todo(db: AsyncSession, tg_id: int, todo_id: int, is_done: bool = True) -> Todo | None:
    todo = await get_todo(db, tg_id, todo_id)
    if todo:
        todo.is_done = is_done
        await db.flush()
    return todo

async def delete_todo(db: AsyncSession, tg_id: int, todo_id: int) -> bool:
    todo = await get_todo(db, tg_id, todo_id)
    if todo:
        await db.delete(todo)
        await db.flush()
        return True
    return False
