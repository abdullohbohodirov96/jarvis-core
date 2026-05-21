"""
Telegram session and authentication management for JARVIS.

TelegramSessionManager handles the full Telethon auth flow:
phone → code request → code verification → optional 2FA password.
It also provides session status inspection and export.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from telethon import TelegramClient, errors
from telethon.sessions import StringSession
from telethon.tl.types import User

from backend.app.core.config import get_settings
from backend.app.core.exceptions import AuthException, TelegramException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session directory
# ---------------------------------------------------------------------------

_SESSION_DIR = Path(__file__).resolve().parents[4] / "data" / "sessions"
_SESSION_DIR.mkdir(parents=True, exist_ok=True)


class TelegramSessionManager:
    """
    Manages Telethon authentication lifecycle:

    1. ``request_code(phone)``      – trigger SMS/Telegram code delivery.
    2. ``verify_code(phone, code, hash)``  – submit the received code.
    3. ``verify_2fa(password)``     – submit 2FA cloud password if required.
    4. ``get_session_status()``     – inspect current auth state.
    5. ``export_session()``         – serialise session to a portable string.
    6. ``revoke_session()``         – log out and delete local session file.

    The manager holds its own TelegramClient instance separate from the main
    TelegramClientManager so that auth can be performed without disrupting
    the live userbot connection.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._api_id: int = settings.TELEGRAM_API_ID
        self._api_hash: str = settings.TELEGRAM_API_HASH
        self._session_name: str = settings.TELEGRAM_SESSION_NAME
        self._phone: str = settings.TELEGRAM_PHONE

        session_path = str(_SESSION_DIR / self._session_name)
        self._client: TelegramClient = TelegramClient(
            session_path,
            self._api_id,
            self._api_hash,
            connection_retries=3,
            retry_delay=1,
            auto_reconnect=False,  # Don't auto-reconnect during auth
        )

        # Transient state used across auth steps within a single flow
        self._phone_code_hash: str | None = None
        self._pending_phone: str | None = None

    # ------------------------------------------------------------------
    # Auth flow
    # ------------------------------------------------------------------

    async def request_code(self, phone: str) -> str:
        """
        Send a login code to the specified phone number via Telegram.

        Args:
            phone: International phone number (e.g. "+79001234567").

        Returns:
            The ``phone_code_hash`` required for the verification step.

        Raises:
            TelegramException: On any Telegram API error.
        """
        if not self._client.is_connected():
            await self._client.connect()

        try:
            result = await self._client.send_code_request(phone)
            self._phone_code_hash = result.phone_code_hash
            self._pending_phone = phone
            logger.info("request_code: code sent to phone=%s", phone)
            return result.phone_code_hash
        except errors.PhoneNumberInvalidError as exc:
            raise AuthException(
                message=f"Invalid phone number: {phone}",
                code="INVALID_PHONE",
                details={"phone": phone},
            ) from exc
        except errors.PhoneNumberBannedError as exc:
            raise AuthException(
                message="This phone number has been banned from Telegram.",
                code="PHONE_BANNED",
            ) from exc
        except errors.RPCError as exc:
            raise TelegramException(
                message=f"Failed to request code: {exc}",
                code="REQUEST_CODE_ERROR",
                details={"rpc_error": str(exc)},
            ) from exc

    async def verify_code(
        self,
        phone: str,
        code: str,
        phone_code_hash: str,
    ) -> bool:
        """
        Submit the received Telegram login code to complete sign-in.

        Args:
            phone:           The phone number used in ``request_code``.
            code:            The code received via SMS or Telegram message.
            phone_code_hash: The hash returned by ``request_code``.

        Returns:
            True on successful authentication.

        Raises:
            AuthException:    On wrong or expired code.
            TelegramException: On other Telegram errors.
        """
        if not self._client.is_connected():
            await self._client.connect()

        try:
            await self._client.sign_in(
                phone=phone,
                code=code,
                phone_code_hash=phone_code_hash,
            )
            self._phone_code_hash = None
            self._pending_phone = None
            logger.info("verify_code: sign-in successful for phone=%s", phone)
            return True

        except errors.SessionPasswordNeededError:
            # 2FA is enabled — caller must follow up with verify_2fa()
            logger.info("verify_code: 2FA required for phone=%s", phone)
            raise AuthException(
                message="Two-factor authentication is required. Call verify_2fa().",
                code="2FA_REQUIRED",
            )
        except errors.PhoneCodeInvalidError as exc:
            raise AuthException(
                message="The code you entered is incorrect.",
                code="INVALID_CODE",
            ) from exc
        except errors.PhoneCodeExpiredError as exc:
            raise AuthException(
                message="The code has expired. Request a new one.",
                code="CODE_EXPIRED",
            ) from exc
        except errors.PhoneNumberUnoccupiedError:
            # Phone not registered — attempt to register (sign up)
            logger.warning("Phone %s not registered; attempting sign-up.", phone)
            try:
                await self._client.sign_up(
                    code=code,
                    first_name="JARVIS",
                    last_name="User",
                    phone=phone,
                    phone_code_hash=phone_code_hash,
                )
                logger.info("verify_code: sign-up completed for phone=%s", phone)
                return True
            except errors.RPCError as signup_exc:
                raise TelegramException(
                    message=f"Sign-up failed: {signup_exc}",
                    code="SIGNUP_ERROR",
                ) from signup_exc
        except errors.RPCError as exc:
            raise TelegramException(
                message=f"Code verification failed: {exc}",
                code="VERIFY_CODE_ERROR",
                details={"rpc_error": str(exc)},
            ) from exc

    async def verify_2fa(self, password: str) -> bool:
        """
        Submit a Two-Factor Authentication cloud password.

        Must be called after ``verify_code`` raises ``AuthException``
        with code ``"2FA_REQUIRED"``.

        Args:
            password: The Telegram cloud password.

        Returns:
            True on success.

        Raises:
            AuthException:    On wrong password or too many attempts.
            TelegramException: On other Telegram errors.
        """
        if not self._client.is_connected():
            await self._client.connect()

        try:
            await self._client.sign_in(password=password)
            logger.info("verify_2fa: 2FA sign-in successful.")
            return True
        except errors.PasswordHashInvalidError as exc:
            raise AuthException(
                message="Incorrect 2FA password.",
                code="WRONG_2FA_PASSWORD",
            ) from exc
        except errors.RPCError as exc:
            raise TelegramException(
                message=f"2FA verification failed: {exc}",
                code="VERIFY_2FA_ERROR",
                details={"rpc_error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Status / export / revoke
    # ------------------------------------------------------------------

    async def get_session_status(self) -> dict[str, Any]:
        """
        Return a snapshot of the current session state.

        Returns:
            Dict with keys: ``connected``, ``authorized``, ``user``,
            ``session_name``, ``checked_at``.
        """
        if not self._client.is_connected():
            try:
                await self._client.connect()
            except Exception:  # noqa: BLE001
                return {
                    "connected": False,
                    "authorized": False,
                    "user": None,
                    "session_name": self._session_name,
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                }

        authorized = await self._client.is_user_authorized()
        user_info: dict[str, Any] | None = None

        if authorized:
            try:
                me: User = await self._client.get_me()  # type: ignore[assignment]
                user_info = {
                    "id": me.id,
                    "first_name": me.first_name or "",
                    "last_name": me.last_name or "",
                    "username": me.username or "",
                    "phone": me.phone or "",
                    "is_bot": me.bot,
                    "is_verified": me.verified,
                }
            except Exception:  # noqa: BLE001
                pass

        return {
            "connected": self._client.is_connected(),
            "authorized": authorized,
            "user": user_info,
            "session_name": self._session_name,
            "session_file": str(_SESSION_DIR / (self._session_name + ".session")),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    async def export_session(self) -> str:
        """
        Export the current session as a portable Telethon StringSession.

        Useful for transferring the session to another process or environment
        without copying the SQLite file.

        Returns:
            Session string (base64-encoded Telethon StringSession).

        Raises:
            AuthException: If not currently authorized.
        """
        if not self._client.is_connected():
            await self._client.connect()

        if not await self._client.is_user_authorized():
            raise AuthException(
                message="Not authenticated. Complete sign-in first.",
                code="NOT_AUTHORIZED",
            )

        session_string: str = StringSession.save(self._client.session)
        logger.info("export_session: session exported.")
        return session_string

    async def revoke_session(self) -> None:
        """
        Log out from Telegram and delete the local session file.

        After calling this method the user must re-authenticate.
        """
        if not self._client.is_connected():
            try:
                await self._client.connect()
            except Exception:  # noqa: BLE001
                pass

        try:
            await self._client.log_out()
            logger.info("revoke_session: logged out from Telegram.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("revoke_session: log_out error (continuing): %s", exc)
        finally:
            await self._client.disconnect()

        # Remove the session SQLite file
        session_file = _SESSION_DIR / (self._session_name + ".session")
        if session_file.exists():
            session_file.unlink()
            logger.info("revoke_session: session file deleted: %s", session_file)

        # Also invalidate the main client singleton
        try:
            from backend.app.telegram.client import get_telegram_client

            import asyncio
            main_client = await asyncio.wait_for(get_telegram_client(), timeout=5.0)
            await main_client.disconnect()
        except Exception:  # noqa: BLE001
            pass

        # Reset singleton so next call re-creates it
        import backend.app.telegram.client as _client_module
        _client_module._telegram_client_instance = None
