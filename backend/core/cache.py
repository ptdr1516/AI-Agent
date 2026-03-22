"""
Redis-backed async cache with in-memory fallback.

Design decisions:
  - RedisCache is a singleton, instantiated once at module level.
  - If Redis is not configured or unavailable, it silently falls back to an
    in-memory TTL dict — so the app still works in local dev without Redis.
  - All operations are async-safe: no blocking calls, no global locks.
  - TTL is per-key to support different expiry policies for different callers.
  - Keys are namespaced with a prefix to avoid collisions with other apps
    sharing the same Redis instance.
"""

import asyncio
import json
import time
from typing import Any, Optional

from core.config import settings
from core.logger import log


# ── In-memory fallback cache ──────────────────────────────────────────────────
class _MemoryCache:
    """Thread-safe TTL cache used when Redis is unavailable."""

    def __init__(self):
        self._store: dict[str, tuple[Any, float]] = {}  # key → (value, expires_at)
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    async def set(self, key: str, value: str, ttl: int) -> None:
        async with self._lock:
            self._store[key] = (value, time.monotonic() + ttl)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def close(self) -> None:
        pass  # Nothing to close for memory cache


# ── Redis cache ───────────────────────────────────────────────────────────────
class _RedisCache:
    """Async Redis cache using redis-py's AsyncRedis client."""

    def __init__(self, url: str, prefix: str):
        self._url = url
        self._prefix = prefix
        self._client = None

    def _key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    async def _ensure_client(self):
        if self._client is None:
            try:
                from redis.asyncio import from_url
                self._client = await from_url(
                    self._url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                log.info(f"Redis cache connected: {self._url}")
            except Exception as exc:
                log.warning(f"Redis connection failed: {exc}")
                raise

    async def get(self, key: str) -> Optional[str]:
        await self._ensure_client()
        return await self._client.get(self._key(key))

    async def set(self, key: str, value: str, ttl: int) -> None:
        await self._ensure_client()
        await self._client.setex(self._key(key), ttl, value)

    async def delete(self, key: str) -> None:
        await self._ensure_client()
        await self._client.delete(self._key(key))

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


# ── Public Cache class with fallback ─────────────────────────────────────────
class Cache:
    """
    Async cache with automatic Redis → in-memory fallback.

    Usage:
        value = await cache.get("my_key")
        if value is None:
            value = await expensive_call()
            await cache.set("my_key", value, ttl=300)

    Can also store/retrieve Python dicts via get_json / set_json helpers.
    """

    def __init__(self):
        self._redis: Optional[_RedisCache] = None
        self._memory = _MemoryCache()
        self._using_redis = False

        if settings.REDIS_URL:
            self._redis = _RedisCache(
                url=settings.REDIS_URL,
                prefix=settings.REDIS_KEY_PREFIX,
            )
            log.info(f"Cache configured — Redis URL: {settings.REDIS_URL}")
        else:
            log.info("REDIS_URL not set — cache will use in-memory fallback.")

    async def _backend(self):
        """Return the active backend, falling back gracefully if Redis is down."""
        if self._redis is not None:
            try:
                await self._redis._ensure_client()
                return self._redis
            except Exception:
                log.warning("Redis unavailable — falling back to in-memory cache.")
                self._redis = None  # stop retrying after first failure
        return self._memory

    async def get(self, key: str) -> Optional[str]:
        backend = await self._backend()
        value = await backend.get(key)
        if value is not None:
            # Notify the current request's metrics accumulator (no-op if none bound)
            try:
                from core.metrics_collector import get_current_metrics
                m = get_current_metrics()
                if m is not None:
                    m.add_cache_hit()
            except Exception:
                pass
        return value

    async def set(self, key: str, value: str, ttl: int) -> None:
        backend = await self._backend()
        await backend.set(key, value, ttl)

    async def delete(self, key: str) -> None:
        backend = await self._backend()
        await backend.delete(key)

    async def get_json(self, key: str) -> Optional[Any]:
        raw = await self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set_json(self, key: str, value: Any, ttl: int) -> None:
        await self.set(key, json.dumps(value), ttl)

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
        await self._memory.close()


# Global singleton — imported by tools and routes
cache = Cache()
