"""
Prometheus metrics for the RAG pipeline.

Exposed at GET /metrics (via prometheus-client + starlette-prometheus).

Metrics collected:
  rag_requests_total            — counter  (labels: endpoint, status)
  rag_request_duration_seconds  — histogram (labels: endpoint)
  rag_tokens_input_total        — counter  (labels: model)
  rag_tokens_output_total       — counter  (labels: model)
  rag_estimated_cost_usd_total  — counter  (labels: model)
  rag_cache_hits_total          — counter  (labels: cache_type)
  rag_cache_misses_total        — counter  (labels: cache_type)
  rag_docs_retrieved            — histogram (labels: stage)
  rag_rerank_top_score          — histogram
  rag_upload_docs_total         — counter
  rag_active_sessions           — gauge
"""
from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram, Gauge
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False

_BUCKETS_LATENCY = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
_BUCKETS_DOCS = (1, 2, 3, 5, 8, 10, 15, 20)
_BUCKETS_SCORE = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)


class RAGMetrics:
    """Singleton wrapper around Prometheus metric objects."""

    _instance: "RAGMetrics | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_metrics()
        return cls._instance

    def _init_metrics(self) -> None:
        if not _PROM_AVAILABLE:
            return
        # ── Requests ────────────────────────────────────────────────────────
        self.requests_total = Counter(
            "rag_requests_total",
            "Total RAG requests",
            ["endpoint", "status"],
        )
        self.request_duration = Histogram(
            "rag_request_duration_seconds",
            "End-to-end request latency",
            ["endpoint"],
            buckets=_BUCKETS_LATENCY,
        )

        # ── Tokens & Cost ────────────────────────────────────────────────────
        self.tokens_input = Counter(
            "rag_tokens_input_total",
            "Input tokens consumed",
            ["model"],
        )
        self.tokens_output = Counter(
            "rag_tokens_output_total",
            "Output tokens generated",
            ["model"],
        )
        self.estimated_cost_usd = Counter(
            "rag_estimated_cost_usd_total",
            "Estimated USD cost of LLM calls",
            ["model"],
        )

        # ── Cache ────────────────────────────────────────────────────────────
        self.cache_hits = Counter(
            "rag_cache_hits_total",
            "Cache hits by type",
            ["cache_type"],          # "response" | "semantic"
        )
        self.cache_misses = Counter(
            "rag_cache_misses_total",
            "Cache misses by type",
            ["cache_type"],
        )

        # ── Retrieval ────────────────────────────────────────────────────────
        self.docs_retrieved = Histogram(
            "rag_docs_retrieved",
            "Number of docs retrieved",
            ["stage"],               # "hybrid" | "reranked"
            buckets=_BUCKETS_DOCS,
        )
        self.rerank_top_score = Histogram(
            "rag_rerank_top_score",
            "Top cross-encoder score after reranking",
            buckets=_BUCKETS_SCORE,
        )

        # ── Sessions & Uploads ───────────────────────────────────────────────
        self.upload_docs = Counter(
            "rag_upload_docs_total",
            "Total documents indexed",
        )
        self.active_sessions = Gauge(
            "rag_active_sessions",
            "Currently active chat sessions",
        )

    # ── Convenience helpers ────────────────────────────────────────────────

    def record_cache_hit(self, cache_type: str) -> None:
        if hasattr(self, "cache_hits"):
            self.cache_hits.labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str) -> None:
        if hasattr(self, "cache_misses"):
            self.cache_misses.labels(cache_type=cache_type).inc()

    def record_tokens(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        if hasattr(self, "tokens_input"):
            self.tokens_input.labels(model=model).inc(input_tokens)
            self.tokens_output.labels(model=model).inc(output_tokens)
            self.estimated_cost_usd.labels(model=model).inc(cost_usd)

    def record_retrieval(self, stage: str, count: int) -> None:
        if hasattr(self, "docs_retrieved"):
            self.docs_retrieved.labels(stage=stage).observe(count)


# Module-level singleton
_metrics: RAGMetrics | None = None


def get_metrics() -> RAGMetrics:
    global _metrics
    if _metrics is None:
        _metrics = RAGMetrics()
    return _metrics
