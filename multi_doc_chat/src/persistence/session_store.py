"""
Session Store — persistent, Redis-backed chat history.

Problem solved:
  The original in-memory SESSIONS dict is reset on every server restart,
  breaking ongoing conversations. In Kubernetes with multiple replicas,
  each pod has its own dict, so load-balanced requests see different histories.

Solution:
  Store chat history in Redis with TTL. Falls back to an in-memory dict
  gracefully when Redis is unavailable (single-replica / development).

Schema (per session):
  Key   : session:<session_id>
  Value : JSON-encoded List[{"role": "user"|"assistant", "content": str}]
  TTL   : configurable (default 24 h)
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from multi_doc_chat.logger import GLOBAL_LOGGER as log

try:
    import redis as redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


_HistoryEntry = Dict[str, str]   # {"role": "user", "content": "..."}


class SessionStore:
    """
    Read/write chat history with Redis-first, in-memory fallback.
    Also tracks active session count (for Prometheus gauge).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 86400,
        backend: str = "redis",
    ):
        self.ttl = ttl_seconds
        self._memory: Dict[str, List[_HistoryEntry]] = {}
        self._redis: Optional[object] = None

        if backend == "redis" and _REDIS_AVAILABLE:
            try:
                self._redis = redis_lib.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                log.info("SessionStore using Redis backend", url=redis_url)
            except Exception as exc:
                log.warning("SessionStore: Redis unavailable — using in-memory", error=str(exc))
                self._redis = None

        if self._redis is None:
            log.info("SessionStore using in-memory backend")

    # ── Public API ─────────────────────────────────────────────────────────

    def get_history(self, session_id: str) -> List[_HistoryEntry]:
        """Return the chat history for a session (empty list if not found)."""
        if self._redis:
            try:
                raw = self._redis.get(self._key(session_id))
                if raw:
                    return json.loads(raw)
                return []
            except Exception as exc:
                log.error("SessionStore get failed", error=str(exc))
                return self._memory.get(session_id, [])
        return self._memory.get(session_id, [])

    def append_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Append a user/assistant turn to the session history."""
        history = self.get_history(session_id)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_message})
        self._save(session_id, history)

    def create_session(self, session_id: str) -> None:
        """Initialise an empty session."""
        self._save(session_id, [])
        log.info("Session created", session_id=session_id)

    def delete_session(self, session_id: str) -> None:
        """Remove session data (e.g. after user logout or TTL management)."""
        if self._redis:
            try:
                self._redis.delete(self._key(session_id))
            except Exception as exc:
                log.error("SessionStore delete failed", error=str(exc))
        self._memory.pop(session_id, None)
        log.info("Session deleted", session_id=session_id)

    def session_exists(self, session_id: str) -> bool:
        if self._redis:
            try:
                return bool(self._redis.exists(self._key(session_id)))
            except Exception:
                pass
        return session_id in self._memory

    def active_count(self) -> int:
        """Approximate count of active sessions (Prometheus gauge)."""
        if self._redis:
            try:
                keys = list(self._redis.scan_iter("session:*", count=1000))
                return len(keys)
            except Exception:
                pass
        return len(self._memory)

    # ── Internals ──────────────────────────────────────────────────────────

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}"

    def _save(self, session_id: str, history: List[_HistoryEntry]) -> None:
        serialized = json.dumps(history, ensure_ascii=False)
        if self._redis:
            try:
                self._redis.setex(self._key(session_id), self.ttl, serialized)
                return
            except Exception as exc:
                log.error("SessionStore save to Redis failed", error=str(exc))
        # Fallback to memory
        self._memory[session_id] = history
