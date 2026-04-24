from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class CacheBackend(Protocol):
    def get(self, key: str) -> str | None:
        ...

    def set(self, key: str, value: str) -> None:
        ...


@dataclass
class NullCacheBackend:
    def get(self, key: str) -> str | None:
        return None

    def set(self, key: str, value: str) -> None:
        return None


@dataclass
class InMemoryCacheBackend:
    store: dict[str, str] = field(default_factory=dict)

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def set(self, key: str, value: str) -> None:
        self.store[key] = value


class RedisCacheBackend:
    def __init__(self, redis_client, ttl_seconds: int | None = None):
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds

    def get(self, key: str) -> str | None:
        return self._redis.get(key)

    def set(self, key: str, value: str) -> None:
        if self._ttl_seconds is not None and self._ttl_seconds > 0:
            self._redis.set(key, value, ex=self._ttl_seconds)
            return
        self._redis.set(key, value)


def build_cache_backend(settings) -> CacheBackend:
    if not getattr(settings, "cache_enabled", False):
        return NullCacheBackend()

    redis_url = getattr(settings, "cache_redis_url", None)
    if not redis_url:
        raise ValueError("CACHE_ENABLED requires CACHE_REDIS_URL to be configured")

    try:
        import redis
    except ImportError as exc:  # pragma: no cover - exercised only when Redis is enabled
        raise RuntimeError("CACHE_ENABLED requires the 'redis' package") from exc

    client = redis.Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=getattr(settings, "cache_socket_timeout_seconds", 2),
        socket_timeout=getattr(settings, "cache_socket_timeout_seconds", 2),
    )
    return RedisCacheBackend(client, ttl_seconds=getattr(settings, "cache_ttl_seconds", None))