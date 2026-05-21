"""
JWT authentication and password hashing utilities for JARVIS.

Key components:
- Password hashing with bcrypt (via passlib)
- JWT access and refresh token creation/verification
- FastAPI dependency ``get_current_user`` for protected routes
- Pydantic models ``Token`` and ``TokenData``
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import ExpiredSignatureError, JWTError, jwt  # type: ignore
from passlib.context import CryptContext  # type: ignore
from pydantic import BaseModel, Field

from backend.app.core.config import get_settings
from backend.app.core.exceptions import AuthException
from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    """Return the bcrypt hash of *password*.

    Args:
        password: Plain-text password to hash.

    Returns:
        A bcrypt hash string suitable for storage.
    """
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify *plain_password* against a stored *hashed_password*.

    Args:
        plain_password:  The raw password submitted by the user.
        hashed_password: The bcrypt hash stored in the database.

    Returns:
        ``True`` if the passwords match, ``False`` otherwise.
    """
    return _pwd_context.verify(plain_password, hashed_password)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Token(BaseModel):
    """Response body returned by the login endpoint."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="OAuth2 token type")


class TokenData(BaseModel):
    """Decoded claims extracted from a JWT access token."""

    sub: str = Field(..., description="Subject – typically the user ID or email")
    email: Optional[str] = Field(default=None, description="User email")
    scopes: list[str] = Field(default_factory=list, description="Permission scopes")
    exp: Optional[datetime] = Field(default=None, description="Expiry timestamp")
    token_type: str = Field(default="access", description="'access' or 'refresh'")


# ---------------------------------------------------------------------------
# OAuth2 bearer scheme
# ---------------------------------------------------------------------------

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token",
    scheme_name="JWT",
    auto_error=True,
)

# Scheme that makes the token optional (for endpoints that work both
# authenticated and unauthenticated).
oauth2_scheme_optional = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token",
    scheme_name="JWT",
    auto_error=False,
)

# ---------------------------------------------------------------------------
# Token creation
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT access token.

    Args:
        data:          Payload to encode.  Should include ``"sub"`` (subject).
        expires_delta: Custom TTL.  Falls back to
                       ``ACCESS_TOKEN_EXPIRE_MINUTES`` from settings.

    Returns:
        Encoded JWT string.
    """
    to_encode = data.copy()
    expire = _utcnow() + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "iat": _utcnow(), "token_type": "access"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT refresh token with a longer TTL.

    Args:
        data:          Payload to encode.  Should include ``"sub"`` (subject).
        expires_delta: Custom TTL.  Falls back to
                       ``REFRESH_TOKEN_EXPIRE_DAYS`` from settings.

    Returns:
        Encoded JWT string.
    """
    to_encode = data.copy()
    expire = _utcnow() + (
        expires_delta
        if expires_delta is not None
        else timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    )
    to_encode.update({"exp": expire, "iat": _utcnow(), "token_type": "refresh"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


# ---------------------------------------------------------------------------
# Token decoding
# ---------------------------------------------------------------------------


def decode_token(token: str) -> TokenData:
    """Decode and validate a JWT, returning structured ``TokenData``.

    Args:
        token: Raw JWT string (without the ``Bearer `` prefix).

    Returns:
        ``TokenData`` with the decoded claims.

    Raises:
        AuthException: If the token is expired, malformed, or missing required
                       claims.
    """
    try:
        payload: dict = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
    except ExpiredSignatureError:
        raise AuthException(
            message="Token has expired.",
            code="TOKEN_EXPIRED",
        )
    except JWTError as exc:
        raise AuthException(
            message="Could not validate credentials.",
            code="INVALID_TOKEN",
            details={"jwt_error": str(exc)},
        )

    sub: Optional[str] = payload.get("sub")
    if not sub:
        raise AuthException(
            message="Token missing 'sub' claim.",
            code="INVALID_TOKEN",
        )

    return TokenData(
        sub=sub,
        email=payload.get("email"),
        scopes=payload.get("scopes", []),
        exp=payload.get("exp"),
        token_type=payload.get("token_type", "access"),
    )


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def get_current_user(
    token: str = Depends(oauth2_scheme),
) -> TokenData:
    """FastAPI dependency that extracts and validates the current user.

    Injects ``TokenData`` into route handlers.  Raises HTTP 401 when the
    token is absent, expired, or invalid.

    Example::

        @router.get("/me")
        async def me(user: TokenData = Depends(get_current_user)):
            return {"sub": user.sub}

    Args:
        token: Bearer token from the ``Authorization`` header, injected by
               FastAPI's OAuth2 scheme.

    Returns:
        Validated ``TokenData``.

    Raises:
        HTTPException 401: If authentication fails for any reason.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token_data = decode_token(token)
    except AuthException as exc:
        logger.warning("auth_failure", reason=exc.code, detail=exc.message)
        raise credentials_exception from exc

    if token_data.token_type != "access":
        raise credentials_exception

    return token_data


async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme_optional),
) -> Optional[TokenData]:
    """Same as ``get_current_user`` but returns ``None`` instead of raising
    for unauthenticated requests.  Useful for endpoints that are public but
    behave differently for logged-in users.
    """
    if token is None:
        return None
    try:
        return decode_token(token)
    except AuthException:
        return None
