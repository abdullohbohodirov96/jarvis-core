import logging
from functools import lru_cache
from cryptography.fernet import Fernet

from core.config import settings

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    """Return cached Fernet instance. Key loaded once from settings."""
    try:
        return Fernet(settings.ENCRYPTION_KEY.encode())
    except Exception as exc:
        logger.error(f"Failed to initialize Fernet with ENCRYPTION_KEY: {exc}")
        # Generate a fallback key for development if key is missing/invalid
        logger.warning("Generating transient Fernet key for safety. DO NOT use this in production!")
        return Fernet(Fernet.generate_key())

def encrypt(value: str) -> str:
    """Encrypt a plaintext string -> URL-safe base64 ciphertext string."""
    if not value:
        return ""
    return _fernet().encrypt(value.encode()).decode()

def decrypt(token: str) -> str:
    """Decrypt a ciphertext string -> plaintext string. Returns empty string on failure."""
    if not token:
        return ""
    try:
        return _fernet().decrypt(token.encode()).decode()
    except Exception as exc:
        logger.error(f"Decryption failed: {exc}")
        return ""
