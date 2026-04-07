"""
Semantic Cache — saves LLM calls by returning cached answers for
*semantically similar* (not just identical) queries.

How it works:
  1. Embed the incoming query.
  2. Scan recent cache entries in Redis for cosine-similarity > threshold.
  3. Cache hit  → return stored answer instantly (0 tokens spent).
  4. Cache miss → compute answer, store (embedding + answer) in Redis with TTL.

Cost impact: ~$0 per cached hit vs. ~$0.001–0.01 per LLM call.
Latency    : <5 ms for cache hit vs. 500–3000 ms for LLM call.

Trade-off: stale answers if documents change. TTL (default 1 h) limits this.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import List, Optional, Tuple

import numpy as np

from multi_doc_chat.logger import GLOBAL_LOGGER as log

try:
    import redis as redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class SemanticCache:
    """
    Redis-backed semantic cache.

    Stores: { embedding: List[float], answer: str, query: str, ts: float }
    Key pattern: sem_cache:<session_id>:<index>   (sorted set per session)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
        max_entries: int = 500,
    ):
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._client: Optional[object] = None

        if _REDIS_AVAILABLE:
            try:
                self._client = redis_lib.from_url(redis_url, decode_responses=False)
                self._client.ping()
                log.info("SemanticCache connected to Redis", url=redis_url)
            except Exception as exc:
                log.warning("SemanticCache: Redis unavailable — cache disabled", error=str(exc))
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    # ── Public API ─────────────────────────────────────────────────────────

    def lookup(
        self,
        query_embedding: List[float],
        session_id: str,
    ) -> Optional[str]:
        """Return cached answer if a similar query exists, else None."""
        if not self.enabled:
            return None
        try:
            entries = self._load_entries(session_id)
            best_score, best_answer = self._find_best(query_embedding, entries)
            if best_score >= self.threshold:
                log.info(
                    "Semantic cache HIT",
                    session_id=session_id,
                    similarity=round(best_score, 4),
                )
                return best_answer
            return None
        except Exception as exc:
            log.error("SemanticCache lookup failed", error=str(exc))
            return None

    def store(
        self,
        query: str,
        query_embedding: List[float],
        answer: str,
        session_id: str,
    ) -> None:
        """Store a query/answer pair in the cache."""
        if not self.enabled:
            return
        try:
            key = self._namespace(session_id)
            entry = json.dumps({
                "query": query,
                "embedding": query_embedding,
                "answer": answer,
                "ts": time.time(),
            }).encode()

            # Use a Redis list; trim to max_entries to bound memory
            pipe = self._client.pipeline()
            pipe.lpush(key, entry)
            pipe.ltrim(key, 0, self.max_entries - 1)
            pipe.expire(key, self.ttl)
            pipe.execute()
            log.debug("SemanticCache stored entry", session_id=session_id)
        except Exception as exc:
            log.error("SemanticCache store failed", error=str(exc))

    def invalidate(self, session_id: str) -> None:
        """Clear all cache entries for a session (e.g. after new doc upload)."""
        if not self.enabled:
            return
        try:
            self._client.delete(self._namespace(session_id))
            log.info("SemanticCache invalidated", session_id=session_id)
        except Exception as exc:
            log.error("SemanticCache invalidate failed", error=str(exc))

    # ── Internals ──────────────────────────────────────────────────────────

    @staticmethod
    def _namespace(session_id: str) -> str:
        return f"sem_cache:{session_id}"

    def _load_entries(self, session_id: str) -> List[dict]:
        raw = self._client.lrange(self._namespace(session_id), 0, self.max_entries - 1)
        entries = []
        for r in raw:
            try:
                entries.append(json.loads(r.decode()))
            except Exception:
                pass
        return entries

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def _find_best(
        self, query_emb: List[float], entries: List[dict]
    ) -> Tuple[float, Optional[str]]:
        best_score = -1.0
        best_answer: Optional[str] = None
        for entry in entries:
            try:
                score = self._cosine(query_emb, entry["embedding"])
                if score > best_score:
                    best_score = score
                    best_answer = entry["answer"]
            except Exception:
                pass
        return best_score, best_answer
