"""
Redis client and cache wrapper for JARVIS.

Provides:
  - ``get_redis_client()``   – returns a shared async Redis connection
  - ``RedisCache``           – high-level key-value cache with JSON serde,
                               hash operations, and list operations

All operations handle serialisation / deserialisation transparently.
Keys are automatically namespaced with the application prefix to avoid
collisions when sharing a Redis instance with other services.

Usage example::

    from app.core.redis_client import RedisCache

    cache = RedisCache()

    # Store a dict with a 5-minute TTL
    await cache.set("user:42:profile", {"name": "Alice"}, ttl=300)

    # Retrieve it
    profile = await cache.get("user:42:profile")  # -> {"name": "Alice"}

    # Delete it
    await cache.delete("user:42:profile")
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import redis.asyncio as aioredis
from redis.asyncio import Redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Namespace prefix applied to every key stored by this module
# ---------------------------------------------------------------------------
_KEY_PREFIX: str = "jarvis:"

# ---------------------------------------------------------------------------
# Module-level singleton connection pool
# ---------------------------------------------------------------------------
_redis_client: Optional[Redis] = None


async def get_redis_client() -> Redis:
    """Return a module-level async Redis client, creating it on first call.

    The client uses a connection pool under the hood so it is safe to call
    this from multiple concurrent coroutines.  The same instance is reused
    for the process lifetime.

    Returns:
        An authenticated, connected ``redis.asyncio.Redis`` instance.

    Raises:
        redis.exceptions.ConnectionError: if the server is unreachable.
    """
    global _redis_client

    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        # Eagerly verify the connection is up
        await _redis_client.ping()
        logger.info("Redis client initialised: %s", settings.REDIS_URL)

    return _redis_client


async def close_redis_client() -> None:
    """Close the shared Redis connection pool.

    Call this during application shutdown to release resources cleanly.
    """
    global _redis_client

    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis client closed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prefixed(key: str) -> str:
    """Prepend the application namespace to *key*."""
    return f"{_KEY_PREFIX}{key}"


def _serialize(value: Any) -> str:
    """Serialise *value* to a JSON string."""
    return json.dumps(value, ensure_ascii=False, default=str)


def _deserialize(raw: Optional[str]) -> Any:
    """Deserialise a JSON string back to a Python object.

    Returns ``None`` if *raw* is ``None`` (key not found in Redis).
    """
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Value was not JSON-encoded (e.g. a plain string counter)
        return raw


# ---------------------------------------------------------------------------
# High-level cache class
# ---------------------------------------------------------------------------

class RedisCache:
    """High-level async Redis cache with JSON serialisation.

    All methods are coroutines.  JSON encoding/decoding happens transparently
    so callers work with native Python objects (dicts, lists, str, int, …).

    Keys are automatically namespaced — callers never need to add a prefix.

    Instantiate once per module or inject as a dependency; the underlying
    Redis connection is shared via ``get_redis_client()``.
    """

    # ── Core key-value operations ─────────────────────────────────────────────

    async def get(self, key: str) -> Any:
        """Retrieve and deserialise the value stored at *key*.

        Returns:
            The Python value, or ``None`` if the key does not exist.
        """
        client = await get_redis_client()
        try:
            raw = await client.get(_prefixed(key))
            return _deserialize(raw)
        except Exception as exc:
            logger.warning("RedisCache.get failed for key=%s: %s", key, exc)
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Serialise *value* and store it at *key*.

        Args:
            key:   Cache key (no namespace prefix needed).
            value: Python object to store (must be JSON-serialisable).
            ttl:   Time-to-live in seconds.  ``None`` means no expiry.

        Returns:
            ``True`` on success, ``False`` on error.
        """
        client = await get_redis_client()
        try:
            serialised = _serialize(value)
            if ttl is not None and ttl > 0:
                await client.setex(_prefixed(key), ttl, serialised)
            else:
                await client.set(_prefixed(key), serialised)
            return True
        except Exception as exc:
            logger.warning("RedisCache.set failed for key=%s: %s", key, exc)
            return False

    async def delete(self, key: str) -> int:
        """Delete *key* from the cache.

        Returns:
            The number of keys deleted (0 or 1).
        """
        client = await get_redis_client()
        try:
            return await client.delete(_prefixed(key))
        except Exception as exc:
            logger.warning("RedisCache.delete failed for key=%s: %s", key, exc)
            return 0

    async def exists(self, key: str) -> bool:
        """Check whether *key* exists in the cache.

        Returns:
            ``True`` if the key exists, ``False`` otherwise.
        """
        client = await get_redis_client()
        try:
            result = await client.exists(_prefixed(key))
            return bool(result)
        except Exception as exc:
            logger.warning("RedisCache.exists failed for key=%s: %s", key, exc)
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Set (or update) the TTL on an existing key.

        Args:
            key: Cache key.
            ttl: New time-to-live in seconds.

        Returns:
            ``True`` if the TTL was set, ``False`` if the key does not exist.
        """
        client = await get_redis_client()
        try:
            return bool(await client.expire(_prefixed(key), ttl))
        except Exception as exc:
            logger.warning("RedisCache.expire failed for key=%s: %s", key, exc)
            return False

    async def ttl(self, key: str) -> int:
        """Return the remaining TTL (in seconds) for *key*.

        Returns:
            Remaining TTL in seconds, -1 if no TTL is set, or -2 if the key
            does not exist.
        """
        client = await get_redis_client()
        try:
            return await client.ttl(_prefixed(key))
        except Exception as exc:
            logger.warning("RedisCache.ttl failed for key=%s: %s", key, exc)
            return -2

    # ── Atomic counter ────────────────────────────────────────────────────────

    async def increment(self, key: str, amount: int = 1) -> int:
        """Atomically increment the integer stored at *key* by *amount*.

        If the key does not exist it is initialised to 0 before incrementing.

        Args:
            key:    Cache key.
            amount: Value to add (default 1).

        Returns:
            The new integer value after incrementing.
        """
        client = await get_redis_client()
        try:
            return await client.incrby(_prefixed(key), amount)
        except Exception as exc:
            logger.warning("RedisCache.increment failed for key=%s: %s", key, exc)
            return 0

    async def decrement(self, key: str, amount: int = 1) -> int:
        """Atomically decrement the integer stored at *key* by *amount*.

        Returns:
            The new integer value after decrementing.
        """
        client = await get_redis_client()
        try:
            return await client.decrby(_prefixed(key), amount)
        except Exception as exc:
            logger.warning("RedisCache.decrement failed for key=%s: %s", key, exc)
            return 0

    # ── List operations ───────────────────────────────────────────────────────

    async def get_list(self, key: str, start: int = 0, end: int = -1) -> list[Any]:
        """Retrieve all (or a slice of) items in a Redis list.

        Items are JSON-deserialised before returning.

        Args:
            key:   Cache key.
            start: Start index (inclusive, 0-based).
            end:   End index (inclusive, -1 = last element).

        Returns:
            A Python list of deserialised values.
        """
        client = await get_redis_client()
        try:
            raw_items = await client.lrange(_prefixed(key), start, end)
            return [_deserialize(item) for item in raw_items]
        except Exception as exc:
            logger.warning("RedisCache.get_list failed for key=%s: %s", key, exc)
            return []

    async def push_list(self, key: str, *values: Any, ttl: Optional[int] = None) -> int:
        """Append one or more *values* to the right end of the Redis list at *key*.

        Args:
            key:    Cache key.
            values: One or more Python objects to serialise and append.
            ttl:    Optional TTL to set on the list key (seconds).

        Returns:
            The length of the list after the push.
        """
        client = await get_redis_client()
        try:
            serialised = [_serialize(v) for v in values]
            length = await client.rpush(_prefixed(key), *serialised)
            if ttl is not None and ttl > 0:
                await client.expire(_prefixed(key), ttl)
            return length
        except Exception as exc:
            logger.warning("RedisCache.push_list failed for key=%s: %s", key, exc)
            return 0

    async def lpush_list(self, key: str, *values: Any, ttl: Optional[int] = None) -> int:
        """Prepend one or more *values* to the left end of the list at *key*.

        Useful for maintaining a most-recent-first queue.

        Returns:
            The length of the list after the push.
        """
        client = await get_redis_client()
        try:
            serialised = [_serialize(v) for v in values]
            length = await client.lpush(_prefixed(key), *serialised)
            if ttl is not None and ttl > 0:
                await client.expire(_prefixed(key), ttl)
            return length
        except Exception as exc:
            logger.warning("RedisCache.lpush_list failed for key=%s: %s", key, exc)
            return 0

    async def list_length(self, key: str) -> int:
        """Return the number of elements in the list at *key*."""
        client = await get_redis_client()
        try:
            return await client.llen(_prefixed(key))
        except Exception as exc:
            logger.warning("RedisCache.list_length failed for key=%s: %s", key, exc)
            return 0

    async def trim_list(self, key: str, start: int, end: int) -> bool:
        """Trim the list at *key* so it contains only the elements in [start, end].

        Useful for keeping a bounded sliding window of recent items.

        Returns:
            ``True`` on success.
        """
        client = await get_redis_client()
        try:
            await client.ltrim(_prefixed(key), start, end)
            return True
        except Exception as exc:
            logger.warning("RedisCache.trim_list failed for key=%s: %s", key, exc)
            return False

    # ── Hash operations ───────────────────────────────────────────────────────

    async def set_hash(self, key: str, mapping: dict[str, Any]) -> bool:
        """Store a flat dict as a Redis hash.

        Each field value is JSON-serialised individually so complex types are
        supported.

        Args:
            key:     Cache key for the hash.
            mapping: Dict of field → value pairs to store.

        Returns:
            ``True`` on success.
        """
        client = await get_redis_client()
        try:
            serialised_mapping = {field: _serialize(value) for field, value in mapping.items()}
            await client.hset(_prefixed(key), mapping=serialised_mapping)
            return True
        except Exception as exc:
            logger.warning("RedisCache.set_hash failed for key=%s: %s", key, exc)
            return False

    async def get_hash(self, key: str) -> dict[str, Any]:
        """Retrieve all fields from the Redis hash at *key*.

        Returns:
            A dict of field → deserialised value, or an empty dict if the key
            does not exist.
        """
        client = await get_redis_client()
        try:
            raw = await client.hgetall(_prefixed(key))
            return {field: _deserialize(raw_value) for field, raw_value in raw.items()}
        except Exception as exc:
            logger.warning("RedisCache.get_hash failed for key=%s: %s", key, exc)
            return {}

    async def get_hash_field(self, key: str, field: str) -> Any:
        """Retrieve a single *field* from the hash at *key*.

        Returns:
            The deserialised value, or ``None`` if not found.
        """
        client = await get_redis_client()
        try:
            raw = await client.hget(_prefixed(key), field)
            return _deserialize(raw)
        except Exception as exc:
            logger.warning(
                "RedisCache.get_hash_field failed for key=%s field=%s: %s", key, field, exc
            )
            return None

    async def set_hash_field(self, key: str, field: str, value: Any) -> bool:
        """Set a single *field* in the hash at *key*.

        Returns:
            ``True`` on success.
        """
        client = await get_redis_client()
        try:
            await client.hset(_prefixed(key), field, _serialize(value))
            return True
        except Exception as exc:
            logger.warning(
                "RedisCache.set_hash_field failed for key=%s field=%s: %s", key, field, exc
            )
            return False

    async def delete_hash_field(self, key: str, *fields: str) -> int:
        """Delete one or more *fields* from the hash at *key*.

        Returns:
            Number of fields that were deleted.
        """
        client = await get_redis_client()
        try:
            return await client.hdel(_prefixed(key), *fields)
        except Exception as exc:
            logger.warning(
                "RedisCache.delete_hash_field failed for key=%s: %s", key, exc
            )
            return 0

    # ── Set operations ────────────────────────────────────────────────────────

    async def add_to_set(self, key: str, *values: Any) -> int:
        """Add one or more *values* to the Redis set at *key*.

        Returns:
            Number of new members added (existing members are not counted).
        """
        client = await get_redis_client()
        try:
            serialised = [_serialize(v) for v in values]
            return await client.sadd(_prefixed(key), *serialised)
        except Exception as exc:
            logger.warning("RedisCache.add_to_set failed for key=%s: %s", key, exc)
            return 0

    async def get_set(self, key: str) -> set[Any]:
        """Retrieve all members of the Redis set at *key*.

        Returns:
            A Python ``set`` of deserialised values.
        """
        client = await get_redis_client()
        try:
            raw_members = await client.smembers(_prefixed(key))
            return {_deserialize(m) for m in raw_members}
        except Exception as exc:
            logger.warning("RedisCache.get_set failed for key=%s: %s", key, exc)
            return set()

    async def is_set_member(self, key: str, value: Any) -> bool:
        """Check whether *value* is a member of the set at *key*."""
        client = await get_redis_client()
        try:
            return bool(await client.sismember(_prefixed(key), _serialize(value)))
        except Exception as exc:
            logger.warning("RedisCache.is_set_member failed for key=%s: %s", key, exc)
            return False

    # ── Pub/Sub helpers ───────────────────────────────────────────────────────

    async def publish(self, channel: str, message: Any) -> int:
        """Publish *message* to a Redis pub/sub *channel*.

        Returns:
            The number of clients that received the message.
        """
        client = await get_redis_client()
        try:
            return await client.publish(_prefixed(channel), _serialize(message))
        except Exception as exc:
            logger.warning("RedisCache.publish failed for channel=%s: %s", channel, exc)
            return 0

    # ── Pattern matching / key introspection ──────────────────────────────────

    async def keys(self, pattern: str) -> list[str]:
        """Return all keys matching *pattern* (glob-style, no prefix needed).

        The namespace prefix is added automatically to the pattern.

        Warning:
            ``KEYS`` scans the entire keyspace — avoid in production loops.
            Use ``SCAN`` via ``scan_keys()`` for large keyspaces.

        Returns:
            List of matching keys *without* the namespace prefix.
        """
        client = await get_redis_client()
        try:
            raw_keys = await client.keys(f"{_KEY_PREFIX}{pattern}")
            # Strip the prefix so callers get back their original key names
            return [k.removeprefix(_KEY_PREFIX) for k in raw_keys]
        except Exception as exc:
            logger.warning("RedisCache.keys failed for pattern=%s: %s", pattern, exc)
            return []

    async def scan_keys(self, pattern: str, count: int = 100) -> list[str]:
        """Non-blocking key scan using Redis SCAN.

        Preferred over ``keys()`` for large keyspaces.

        Returns:
            All matching keys without the namespace prefix.
        """
        client = await get_redis_client()
        results: list[str] = []
        try:
            async for key in client.scan_iter(
                match=f"{_KEY_PREFIX}{pattern}", count=count
            ):
                results.append(key.removeprefix(_KEY_PREFIX))
        except Exception as exc:
            logger.warning("RedisCache.scan_keys failed for pattern=%s: %s", pattern, exc)
        return results

    # ── Multi-key delete ──────────────────────────────────────────────────────

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching *pattern* using SCAN + DELETE pipeline.

        Returns:
            Total number of keys deleted.
        """
        client = await get_redis_client()
        total_deleted = 0
        try:
            pipeline = client.pipeline(transaction=False)
            async for key in client.scan_iter(match=f"{_KEY_PREFIX}{pattern}", count=100):
                pipeline.delete(key)
            results = await pipeline.execute()
            total_deleted = sum(r for r in results if isinstance(r, int))
        except Exception as exc:
            logger.warning("RedisCache.delete_pattern failed for pattern=%s: %s", pattern, exc)
        return total_deleted


# ---------------------------------------------------------------------------
# Module-level singleton instance (convenience import)
# ---------------------------------------------------------------------------

#: Default module-level cache instance.  Import and use directly:
#:
#:   from app.core.redis_client import cache
#:   await cache.set("foo", {"bar": 1}, ttl=60)
cache: RedisCache = RedisCache()
