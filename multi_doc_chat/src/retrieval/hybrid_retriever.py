"""
Hybrid Retriever: BM25 (sparse) + FAISS (dense) fused via Reciprocal Rank Fusion (RRF).

Why hybrid?
  - Dense (FAISS) excels at semantic / paraphrase matching.
  - Sparse (BM25) excels at exact keyword / entity matching.
  - RRF combines both rankings without needing score normalization.

RRF formula:  score(d) = Σ_r  weight_r / (k + rank_r(d))
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from pydantic import PrivateAttr

from multi_doc_chat.logger import GLOBAL_LOGGER as log

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    log.warning("rank-bm25 not installed — falling back to dense-only retrieval")


class HybridRetriever(BaseRetriever):
    """
    Retriever that fuses BM25 keyword search and FAISS semantic search
    using Reciprocal Rank Fusion.
    """

    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    rrf_k: int = 60
    top_k: int = 5
    fetch_n: int = 20           # Candidates fetched from each retriever before fusion

    # Private attrs (not serialized by Pydantic)
    _dense_retriever: object = PrivateAttr(default=None)
    _documents: List[Document] = PrivateAttr(default_factory=list)
    _bm25: Optional[object] = PrivateAttr(default=None)

    def __init__(
        self,
        dense_retriever,
        documents: List[Document],
        *,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        rrf_k: int = 60,
        top_k: int = 5,
        fetch_n: int = 20,
    ):
        super().__init__(
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            rrf_k=rrf_k,
            top_k=top_k,
            fetch_n=fetch_n,
        )
        self._dense_retriever = dense_retriever
        self._documents = documents

        if _BM25_AVAILABLE and documents:
            corpus = [d.page_content.lower().split() for d in documents]
            self._bm25 = BM25Okapi(corpus)
            log.info(
                "HybridRetriever ready",
                num_docs=len(documents),
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
                rrf_k=rrf_k,
            )
        else:
            log.warning("HybridRetriever running dense-only (BM25 unavailable or no docs)")

    # ── LangChain interface ─────────────────────────────────────────────────

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self._hybrid_search(query)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # For true async, wrap in executor; good enough for I/O-bound RAG
        return self._hybrid_search(query)

    # ── Core fusion logic ────────────────────────────────────────────────────

    def _hybrid_search(self, query: str) -> List[Document]:
        dense_docs = self._dense_search(query)

        if self._bm25 is None or not self._documents:
            return dense_docs[: self.top_k]

        bm25_docs = self._bm25_search(query)
        fused = self._rrf_fuse(dense_docs, bm25_docs)

        log.info(
            "Hybrid search",
            query_preview=query[:60],
            dense_hits=len(dense_docs),
            bm25_hits=len(bm25_docs),
            fused_top_k=len(fused),
        )
        return fused

    def _dense_search(self, query: str) -> List[Document]:
        try:
            return self._dense_retriever.invoke(query)
        except Exception as exc:
            log.error("Dense retrieval failed", error=str(exc))
            return []

    def _bm25_search(self, query: str) -> List[Document]:
        try:
            tokens = query.lower().split()
            scores = self._bm25.get_scores(tokens)
            top_idx = np.argsort(scores)[::-1][: self.fetch_n]
            return [self._documents[i] for i in top_idx if scores[i] > 0]
        except Exception as exc:
            log.error("BM25 retrieval failed", error=str(exc))
            return []

    def _rrf_fuse(
        self,
        dense_docs: List[Document],
        bm25_docs: List[Document],
    ) -> List[Document]:
        scores: Dict[str, float] = {}
        doc_store: Dict[str, Document] = {}

        def _key(doc: Document) -> str:
            return doc.page_content[:256]

        for rank, doc in enumerate(dense_docs):
            k = _key(doc)
            scores[k] = scores.get(k, 0.0) + self.dense_weight / (self.rrf_k + rank + 1)
            doc_store[k] = doc

        for rank, doc in enumerate(bm25_docs):
            k = _key(doc)
            scores[k] = scores.get(k, 0.0) + self.bm25_weight / (self.rrf_k + rank + 1)
            doc_store[k] = doc

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [doc_store[k] for k in sorted_keys[: self.top_k]]

    # ── Persistence helpers ──────────────────────────────────────────────────

    def save_bm25(self, path: Path) -> None:
        """Persist the BM25 index alongside the FAISS index."""
        if self._bm25 is not None:
            with open(path, "wb") as fh:
                pickle.dump(
                    {"bm25": self._bm25, "doc_texts": [d.page_content for d in self._documents]},
                    fh,
                )
            log.info("BM25 index saved", path=str(path))

    @classmethod
    def load_bm25(cls, path: Path):
        """Return (bm25_index, doc_texts) from a saved pickle."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        return data["bm25"], data["doc_texts"]
