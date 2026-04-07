"""
Cross-Encoder Reranker.

Why rerank?
  The first-stage retriever (FAISS/BM25) optimises for recall —
  it returns candidates that *might* be relevant.
  The cross-encoder scores each (query, document) pair jointly,
  giving much higher precision at the cost of more compute.

  Typical flow:
    Retrieve top-20  →  rerank  →  keep top-5  →  LLM

Model default: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22 M params, ~5 ms/pair on CPU — very lightweight.
  - Drop-in upgrade path: swap for cross-encoder/ms-marco-electra-base
    or Cohere Rerank API (set use_api=True).
"""
from __future__ import annotations

from typing import List, Tuple

from langchain.schema import Document
from multi_doc_chat.logger import GLOBAL_LOGGER as log

try:
    from sentence_transformers import CrossEncoder
    _CE_AVAILABLE = True
except ImportError:
    _CE_AVAILABLE = False
    log.warning("sentence-transformers not installed — reranking disabled")


class CrossEncoderReranker:
    """
    Re-scores a list of candidate documents given a query using a cross-encoder model.
    Returns the top-n documents sorted by relevance score.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.top_n = top_n
        self.batch_size = batch_size
        self._model = None

        if _CE_AVAILABLE:
            try:
                self._model = CrossEncoder(model_name)
                log.info("CrossEncoderReranker loaded", model=model_name)
            except Exception as exc:
                log.error("Failed to load CrossEncoder", model=model_name, error=str(exc))

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents and return top_n most relevant.
        Falls back to original order if model unavailable.
        """
        if not documents:
            return documents

        if self._model is None:
            log.warning("Reranker unavailable — returning original order")
            return documents[: self.top_n]

        try:
            pairs: List[Tuple[str, str]] = [
                (query, doc.page_content) for doc in documents
            ]
            scores: List[float] = self._model.predict(
                pairs, batch_size=self.batch_size, show_progress_bar=False
            )

            scored = sorted(
                zip(scores, documents),
                key=lambda x: x[0],
                reverse=True,
            )

            top_docs = [doc for _, doc in scored[: self.top_n]]

            log.info(
                "Reranking complete",
                query_preview=query[:60],
                candidates=len(documents),
                returned=len(top_docs),
                top_score=round(float(scored[0][0]), 4) if scored else None,
            )
            return top_docs

        except Exception as exc:
            log.error("Reranking failed", error=str(exc))
            return documents[: self.top_n]

    def rerank_with_scores(
        self, query: str, documents: List[Document]
    ) -> List[Tuple[float, Document]]:
        """Same as rerank() but returns (score, doc) tuples — useful for debugging."""
        if self._model is None:
            return [(0.0, d) for d in documents[: self.top_n]]

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [(float(s), d) for s, d in scored[: self.top_n]]
