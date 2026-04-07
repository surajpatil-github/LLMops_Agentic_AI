"""
OpenTelemetry distributed tracing setup.

When enabled (monitoring.tracing.enabled=true in config), every request
gets a trace that shows:
  - HTTP request span (FastAPI auto-instrumentation)
  - Embedding span
  - Hybrid retrieval span (BM25 + FAISS)
  - Reranking span
  - LLM call span
  - Cache lookup/store spans

Traces are exported to an OTLP-compatible collector (Jaeger, Tempo, etc.)
via gRPC. Disable gracefully when no collector is running.
"""
from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

from multi_doc_chat.logger import GLOBAL_LOGGER as log

_OTEL_AVAILABLE = False
_tracer = None

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    _OTEL_AVAILABLE = True
except ImportError:
    log.warning("opentelemetry-sdk not installed — tracing disabled")


def setup_tracing(
    service_name: str = "multidocchat",
    otlp_endpoint: str = "http://otel-collector:4317",
    sample_rate: float = 1.0,
    enabled: bool = False,
) -> None:
    """
    Initialize OpenTelemetry with OTLP gRPC exporter.
    Call once at application startup.
    """
    global _tracer

    if not enabled or not _OTEL_AVAILABLE:
        log.info("Tracing disabled or unavailable")
        return

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        resource = Resource.create({"service.name": service_name})
        sampler = TraceIdRatioBased(sample_rate)
        provider = TracerProvider(resource=resource, sampler=sampler)

        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_name)

        log.info(
            "OpenTelemetry tracing initialized",
            service=service_name,
            endpoint=otlp_endpoint,
            sample_rate=sample_rate,
        )
    except Exception as exc:
        log.error("Failed to initialize tracing", error=str(exc))


def get_tracer():
    """Return the global tracer (or None if tracing is disabled)."""
    return _tracer


@contextmanager
def span(name: str, attributes: Optional[dict] = None) -> Generator:
    """
    Context manager that creates a trace span when tracing is available,
    otherwise is a no-op (zero overhead).

    Usage:
        with span("rag.retrieval", {"k": 5}):
            docs = retriever.invoke(query)
    """
    if _tracer is None:
        yield
        return

    with _tracer.start_as_current_span(name) as s:
        if attributes:
            for k, v in attributes.items():
                s.set_attribute(k, str(v))
        try:
            yield s
        except Exception as exc:
            s.record_exception(exc)
            s.set_status(
                __import__("opentelemetry.trace", fromlist=["StatusCode"]).StatusCode.ERROR
            )
            raise


def timed_span(name: str):
    """
    Decorator that wraps a function in a trace span AND logs its wall-clock time.

    Usage:
        @timed_span("rag.reranking")
        def rerank(query, docs):
            ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            with span(name):
                result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            log.debug(f"{name} completed", elapsed_ms=round(elapsed * 1000, 1))
            return result
        return wrapper
    return decorator
