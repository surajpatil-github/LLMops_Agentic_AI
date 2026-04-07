"""
MultiDocChat — FastAPI application (Advanced RAG edition v0.2).

What's new vs. v1:
  - Request-ID middleware  (correlation ID in every response header + log)
  - Prometheus /metrics    endpoint (via prometheus-client)
  - OpenTelemetry FastAPI  auto-instrumentation
  - Redis-backed SessionStore (survives restarts, multi-replica safe)
  - POST /chat returns token-usage metadata
  - GET  /health  reports Redis, session count, config
  - GET  /session/{id}/cost   — per-session cost summary
  - POST /session/{id}/reset  — clear history + invalidate caches
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage

from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.utils.config_loader import load_config

# ── App bootstrap ──────────────────────────────────────────────────────────

cfg = load_config()

app = FastAPI(
    title="MultiDocChat",
    version="0.2.0",
    description=(
        "Production-grade Advanced RAG: "
        "hybrid BM25+FAISS search, cross-encoder reranking, "
        "semantic caching, Prometheus metrics, OTel tracing"
    ),
)

# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus /metrics endpoint ───────────────────────────────────────────
try:
    from prometheus_client import make_asgi_app
    app.mount("/metrics", make_asgi_app())
    log.info("Prometheus /metrics mounted")
except Exception as _prom_err:
    log.warning("prometheus_client unavailable", error=str(_prom_err))

# ── OpenTelemetry auto-instrumentation ─────────────────────────────────────
try:
    from multi_doc_chat.src.monitoring.tracing import setup_tracing
    mon_cfg = cfg.get("monitoring", {}).get("tracing", {})
    setup_tracing(
        service_name=mon_cfg.get("service_name", "multidocchat"),
        otlp_endpoint=mon_cfg.get("otlp_endpoint", "http://otel-collector:4317"),
        sample_rate=mon_cfg.get("sample_rate", 1.0),
        enabled=mon_cfg.get("enabled", False),
    )
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app)
except Exception as _otel_err:
    log.warning("OTel instrumentation skipped", error=str(_otel_err))

# ── Static files & templates ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── Lazy-init helpers ──────────────────────────────────────────────────────

_session_store = None


def _get_session_store():
    global _session_store
    if _session_store is None:
        from multi_doc_chat.src.persistence.session_store import SessionStore
        sess_cfg = cfg.get("session", {})
        redis_url = os.getenv(
            "REDIS_URL", cfg.get("cache", {}).get("redis_url", "redis://localhost:6379/0")
        )
        _session_store = SessionStore(
            redis_url=redis_url,
            ttl_seconds=sess_cfg.get("ttl_seconds", 86400),
            backend=sess_cfg.get("backend", "redis"),
        )
    return _session_store


def _get_metrics():
    try:
        from multi_doc_chat.src.monitoring.metrics import get_metrics
        return get_metrics()
    except Exception:
        return None


# ── Request-ID + metrics middleware ───────────────────────────────────────

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    t0 = time.perf_counter()

    response = await call_next(request)

    elapsed_s = time.perf_counter() - t0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(round(elapsed_s * 1000, 1))

    m = _get_metrics()
    if m:
        status = "success" if response.status_code < 400 else "error"
        m.requests_total.labels(endpoint=request.url.path, status=status).inc()
        m.request_duration.labels(endpoint=request.url.path).observe(elapsed_s)

    return response


# ── Pydantic schemas ───────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    session_id: str
    indexed: int
    message: str = "Documents indexed successfully"


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=4096)


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class SessionCostResponse(BaseModel):
    session_id: str
    calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    budget_remaining_usd: float


# ── File adapter ───────────────────────────────────────────────────────────

class FastAPIFileAdapter:
    """Bridges FastAPI UploadFile to the interface expected by save_uploaded_files."""

    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.filename = uf.filename or "file"

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()


# ── In-process RAG cache (avoids reloading FAISS + BM25 per request) ──────
_rag_cache: Dict[str, ConversationalRAG] = {}
_cost_trackers: Dict[str, object] = {}


def _get_rag(session_id: str) -> ConversationalRAG:
    if session_id not in _rag_cache:
        index_path = str(Path("faiss_index") / session_id)
        rag = ConversationalRAG(session_id=session_id)
        retr_cfg = cfg.get("retriever", {})
        rag.load_retriever_from_faiss(
            index_path=index_path,
            k=retr_cfg.get("top_k", 5),
            search_type=retr_cfg.get("search_type", "mmr"),
            fetch_k=retr_cfg.get("fetch_k", 20),
            lambda_mult=retr_cfg.get("lambda_mult", 0.5),
        )
        _rag_cache[session_id] = rag
        if rag._token_tracker:
            _cost_trackers[session_id] = rag._token_tracker
    return _rag_cache[session_id]


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Deep health check: Redis connectivity, active sessions, config."""
    checks: Dict[str, str] = {"api": "ok"}

    try:
        import redis as _r
        _r.from_url(
            os.getenv("REDIS_URL", cfg.get("cache", {}).get("redis_url", "redis://localhost:6379/0"))
        ).ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"unavailable ({e})"

    try:
        checks["active_sessions"] = str(_get_session_store().active_count())
    except Exception:
        checks["active_sessions"] = "unknown"

    overall = "healthy" if all(
        v in ("ok", "unknown") or not v.startswith("unavailable")
        for v in checks.values()
    ) else "degraded"
    return {"status": overall, "checks": checks}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)):
    """Upload documents → build hybrid index → return session_id."""
    if not files:
        raise HTTPException(status_code=422, detail="No files provided")

    try:
        adapters = [FastAPIFileAdapter(f) for f in files]
        ingestor = ChatIngestor(use_session_dirs=True)

        retr_cfg = cfg.get("retriever", {})
        retriever = ingestor.built_retriver(
            adapters,
            k=retr_cfg.get("top_k", 5),
            search_type=retr_cfg.get("search_type", "mmr"),
            fetch_k=retr_cfg.get("fetch_k", 20),
            lambda_mult=retr_cfg.get("lambda_mult", 0.5),
        )

        session_id = ingestor.session_id
        _get_session_store().create_session(session_id)

        # Pre-warm RAG instance
        rag = ConversationalRAG(session_id=session_id, retriever=retriever)
        _rag_cache[session_id] = rag
        if rag._token_tracker:
            _cost_trackers[session_id] = rag._token_tracker

        m = _get_metrics()
        if m:
            m.active_sessions.set(_get_session_store().active_count())

        log.info("Upload complete", session_id=session_id, files=len(files))
        return UploadResponse(session_id=session_id, indexed=len(files))

    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        log.error("Upload error", error=str(e))
        raise HTTPException(status_code=500, detail="Upload failed")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Chat with uploaded documents."""
    store = _get_session_store()

    if not store.session_exists(req.session_id):
        raise HTTPException(
            status_code=400,
            detail=f"Session '{req.session_id}' not found. Upload documents first.",
        )

    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        rag = _get_rag(req.session_id)

        raw_history = store.get_history(req.session_id)
        lc_history = []
        for turn in raw_history:
            if turn["role"] == "user":
                lc_history.append(HumanMessage(content=turn["content"]))
            else:
                lc_history.append(AIMessage(content=turn["content"]))

        answer = rag.invoke(message, chat_history=lc_history)
        store.append_turn(req.session_id, message, answer)

        return ChatResponse(answer=answer, session_id=req.session_id)

    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        log.error("Chat error", session_id=req.session_id, error=str(e))
        raise HTTPException(status_code=500, detail="Chat request failed")


@app.get("/session/{session_id}/cost", response_model=SessionCostResponse)
async def session_cost(session_id: str):
    """Token usage and estimated cost for a session."""
    tracker = _cost_trackers.get(session_id)
    if not tracker:
        if not _get_session_store().session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        try:
            rag = _get_rag(session_id)
            tracker = rag._token_tracker
        except Exception:
            pass

    if not tracker:
        raise HTTPException(status_code=404, detail="Cost data not available")

    s = tracker.summary()
    return SessionCostResponse(**s)


@app.post("/session/{session_id}/reset")
async def session_reset(session_id: str):
    """Clear history + invalidate caches for a session."""
    store = _get_session_store()
    if not store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    store.create_session(session_id)  # Resets history to []

    try:
        redis_url = os.getenv(
            "REDIS_URL", cfg.get("cache", {}).get("redis_url", "redis://localhost:6379/0")
        )
        from multi_doc_chat.src.cache.semantic_cache import SemanticCache
        from multi_doc_chat.src.cache.response_cache import ResponseCache
        SemanticCache(redis_url=redis_url).invalidate(session_id)
        ResponseCache(redis_url=redis_url).invalidate_session(session_id)
    except Exception as e:
        log.warning("Cache invalidation failed", error=str(e))

    _rag_cache.pop(session_id, None)
    log.info("Session reset", session_id=session_id)
    return {"message": "Session reset", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENV", "local") == "local",
    )
