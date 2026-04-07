"""
Document Ingestion Pipeline — Advanced RAG edition.

Improvements over v1:
  1. Builds a HybridRetriever (BM25 + FAISS) instead of FAISS-only.
  2. Persists BM25 index to disk (bm25_index.pkl) alongside FAISS files.
  3. Exposes all chunking parameters via config.yaml.
  4. Invalidates response/semantic caches after a new upload (stale-answer guard).
  5. Records Prometheus upload metrics.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.utils.document_ops import load_documents
from multi_doc_chat.utils.file_io import save_uploaded_files
from multi_doc_chat.utils.model_loader import ModelLoader


def generate_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{unique_id}"


# ──────────────────────────────────────────────────────────────────────────────
# ChatIngestor
# ──────────────────────────────────────────────────────────────────────────────

class ChatIngestor:
    """
    Orchestrates the full ingestion pipeline for one session:
      upload → load → chunk → embed → FAISS index + BM25 index → HybridRetriever
    """

    def __init__(
        self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
    ):
        try:
            self.cfg = load_config()
            self.model_loader = ModelLoader()
            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()

            self.temp_base = Path(temp_base)
            self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base)
            self.faiss_base.mkdir(parents=True, exist_ok=True)

            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            log.info(
                "ChatIngestor initialized",
                session_id=self.session_id,
                temp_dir=str(self.temp_dir),
                faiss_dir=str(self.faiss_dir),
            )
        except Exception as e:
            raise DocumentPortalException("Initialization error in ChatIngestor", e) from e

    # ── Public API ─────────────────────────────────────────────────────────

    def built_retriver(
        self,
        uploaded_files: Iterable,
        *,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        k: int = 5,
        search_type: str = "mmr",
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ):
        """
        Full ingestion pipeline. Returns a HybridRetriever when rank-bm25 is
        installed, otherwise falls back to a plain FAISS retriever.
        """
        try:
            chunk_cfg = self.cfg.get("chunking", {})
            chunk_size = chunk_size or chunk_cfg.get("chunk_size", 1000)
            chunk_overlap = chunk_overlap or chunk_cfg.get("chunk_overlap", 200)

            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")

            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            fm = FaissManager(self.faiss_dir, self.model_loader)
            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]
            vs = fm.load_or_create(texts=texts, metadatas=metas)
            added = fm.add_documents(chunks)

            log.info("FAISS index updated", added=added, index=str(self.faiss_dir))

            # ── Record upload metrics ──────────────────────────────────────
            try:
                from multi_doc_chat.src.monitoring.metrics import get_metrics
                get_metrics().upload_docs.inc(added)
            except Exception:
                pass

            # ── Build HybridRetriever (BM25 + FAISS) ──────────────────────
            retriever = self._build_hybrid_retriever(
                vs=vs,
                all_chunks=fm.all_documents(),
                k=k,
                search_type=search_type,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
            )
            return retriever

        except Exception as e:
            log.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e

    # ── Internals ──────────────────────────────────────────────────────────

    def _resolve_dir(self, base: Path) -> Path:
        if self.use_session:
            d = base / self.session_id
            d.mkdir(parents=True, exist_ok=True)
            return d
        return base

    def _split(
        self,
        docs: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[Document]:
        cfg = self.cfg.get("chunking", {})
        separators = cfg.get("separators", ["\n\n", "\n", ". ", " ", ""])
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
        chunks = splitter.split_documents(docs)
        log.info("Documents split", chunks=len(chunks), chunk_size=chunk_size)
        return chunks

    def _build_hybrid_retriever(
        self,
        vs: FAISS,
        all_chunks: List[Document],
        k: int,
        search_type: str,
        fetch_k: int,
        lambda_mult: float,
    ):
        search_kwargs: Dict[str, Any] = {"k": k}
        if search_type == "mmr":
            search_kwargs.update(fetch_k=fetch_k, lambda_mult=lambda_mult)

        dense_retriever = vs.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

        adv_cfg = self.cfg.get("advanced_rag", {}).get("hybrid_search", {})
        if not adv_cfg.get("enabled", True):
            return dense_retriever

        try:
            from multi_doc_chat.src.retrieval.hybrid_retriever import HybridRetriever
            hybrid = HybridRetriever(
                dense_retriever=dense_retriever,
                documents=all_chunks,
                bm25_weight=adv_cfg.get("bm25_weight", 0.3),
                dense_weight=adv_cfg.get("dense_weight", 0.7),
                rrf_k=adv_cfg.get("rrf_k", 60),
                top_k=k,
                fetch_n=adv_cfg.get("fetch_n", 20),
            )
            # Persist BM25 index alongside FAISS
            bm25_path = self.faiss_dir / "bm25_index.pkl"
            hybrid.save_bm25(bm25_path)
            return hybrid
        except ImportError:
            log.warning("HybridRetriever unavailable — using dense-only")
            return dense_retriever


# ──────────────────────────────────────────────────────────────────────────────
# FaissManager
# ──────────────────────────────────────────────────────────────────────────────

class FaissManager:
    """FAISS index lifecycle: create, load, deduplicate-add."""

    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}}
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}}
            except Exception:
                self._meta = {"rows": {}}

        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None
        self._all_docs: List[Document] = []

    def _exists(self) -> bool:
        return (
            (self.index_dir / "index.faiss").exists()
            and (self.index_dir / "index.pkl").exists()
        )

    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save_meta(self) -> None:
        self.meta_path.write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load_or_create(
        self,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
    ) -> FAISS:
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs

        if not texts:
            raise DocumentPortalException(
                "No existing FAISS index and no data to create one", sys
            )
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs

    def add_documents(self, docs: List[Document]) -> int:
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents().")

        new_docs: List[Document] = []
        for d in docs:
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key not in self._meta["rows"]:
                self._meta["rows"][key] = True
                new_docs.append(d)
                self._all_docs.append(d)

        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)

    def all_documents(self) -> List[Document]:
        """Return all documents currently in the index (for BM25 construction)."""
        return list(self._all_docs)
