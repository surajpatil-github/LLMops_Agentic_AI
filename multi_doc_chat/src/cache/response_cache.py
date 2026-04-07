"""
Response Cache — exact-match cache for identical queries.

Cheaper and faster than SemanticCache (no embedding computation).
Use both: ResponseCache first (O(1)), then SemanticCache (~O(n)).

Key  : sha256(session_id + query)
Value: answer string
TTL  : configurable (default 5 min)
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

from multi_doc_chat.logger import GLOBAL_LOGGER as log

try:
    import redis as redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class ResponseCache:
    """Simple SHA-256 keyed Redis cache for RAG answers."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 300,
    ):
        self.ttl = ttl_seconds
        self._client: Optional[object] = None

        if _REDIS_AVAILABLE:
            try:
                self._client = redis_lib.from_url(redis_url, decode_responses=True)
                self._client.ping()
                log.info("ResponseCache connected to Redis", url=redis_url)
            except Exception as exc:
                log.warning("ResponseCache: Redis unavailable", error=str(exc))

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def get(self, session_id: str, query: str) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            cached = self._client.get(self._key(session_id, query))
            if cached:
                log.info("Response cache HIT", session_id=session_id)
                return json.loads(cached)
            return None
        except Exception as exc:
            log.error("ResponseCache get failed", error=str(exc))
            return None

    def set(self, session_id: str, query: str, answer: str) -> None:
        if not self.enabled:
            return
        try:
            self._client.setex(
                self._key(session_id, query),
                self.ttl,
                json.dumps(answer),
            )
        except Exception as exc:
            log.error("ResponseCache set failed", error=str(exc))

    def invalidate_session(self, session_id: str) -> None:
        """Best-effort: scan and delete all keys for a session."""
        if not self.enabled:
            return
        try:
            pattern = f"resp_cache:{session_id}:*"
            keys = list(self._client.scan_iter(pattern, count=200))
            if keys:
                self._client.delete(*keys)
            log.info("ResponseCache session invalidated", session_id=session_id, keys=len(keys))
        except Exception as exc:
            log.error("ResponseCache invalidate failed", error=str(exc))

    @staticmethod
    def _key(session_id: str, query: str) -> str:
        digest = hashlib.sha256(f"{session_id}:{query}".encode()).hexdigest()
        return f"resp_cache:{session_id}:{digest}"
