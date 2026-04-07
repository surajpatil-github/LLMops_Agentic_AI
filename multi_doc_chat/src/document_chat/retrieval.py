"""
ConversationalRAG — Advanced edition.

Pipeline per query:
  1. Response cache check  (exact match, ~1 ms)
  2. Embed query
  3. Semantic cache check  (cosine similarity, ~5 ms)
  4. [Optional] HyDE      (hypothetical document, +1 LLM call)
  5. Hybrid retrieval     (BM25 + FAISS via RRF)
  6. Cross-encoder rerank (re-score top-N candidates)
  7. LLM answer generation
  8. Token tracking & cost estimation
  9. Store in caches
 10. Record Prometheus metrics

Fallback: every advanced component degrades gracefully to the v1 behaviour
          if its dependency is unavailable or disabled in config.
"""
from __future__ import annotations

import os
import sys
import time
from operator import itemgetter
from typing import Any, Dict, List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import ValidationError

from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.model.models import ChatAnswer, PromptType
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.utils.model_loader import ModelLoader


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with:
      - Hybrid retrieval (BM25 + FAISS)
      - Cross-encoder reranking
      - HyDE query expansion
      - Semantic + response caching
      - Token tracking & budget enforcement
      - Prometheus metrics + OTel tracing
    """

    def __init__(self, session_id: Optional[str] = None, retriever=None):
        try:
            self.session_id = session_id
            self.cfg = load_config()

            self.llm = ModelLoader().load_llm()
            self.contextualize_prompt = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            # ── Optional components (all fail-safe) ────────────────────────
            self._reranker = self._init_reranker()
            self._sem_cache = self._init_semantic_cache()
            self._resp_cache = self._init_response_cache()
            self._token_tracker = self._init_token_tracker()
            self._metrics = self._load_metrics()
            self._embeddings = None   # lazy-load for semantic cache

            log.info(
                "ConversationalRAG initialized",
                session_id=self.session_id,
                reranker=self._reranker.is_available if self._reranker else False,
                semantic_cache=self._sem_cache.enabled if self._sem_cache else False,
                response_cache=self._resp_cache.enabled if self._resp_cache else False,
            )
        except Exception as e:
            log.error("ConversationalRAG init failed", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", sys)

    # ── Public API ─────────────────────────────────────────────────────────

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "mmr",
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Load FAISS + optional BM25 index from disk and build retriever."""
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index not found: {index_path}")

            embeddings = ModelLoader().load_embeddings()
            vs = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )

            if search_kwargs is None:
                search_kwargs = {"k": k}
                if search_type == "mmr":
                    search_kwargs.update(fetch_k=fetch_k, lambda_mult=lambda_mult)

            dense_retriever = vs.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )

            # Try to load the persisted BM25 index
            self.retriever = self._try_load_hybrid(
                index_path, vs, dense_retriever, k, search_type, search_kwargs
            )
            self._build_lcel_chain()

            log.info(
                "Retriever loaded",
                index_path=index_path,
                search_type=search_type,
                k=k,
                hybrid=self.retriever.__class__.__name__,
            )
            return self.retriever

        except Exception as e:
            log.error("Failed to load retriever", error=str(e))
            raise DocumentPortalException("Loading error in ConversationalRAG", sys)

    def invoke(
        self,
        user_input: str,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> str:
        """
        Full advanced RAG pipeline:
          cache → contextualize → retrieve → rerank → LLM → track cost → cache store
        """
        t_start = time.perf_counter()
        chat_history = chat_history or []

        try:
            if self.retriever is None:
                raise DocumentPortalException(
                    "RAG not initialized. Call load_retriever_from_faiss() first.", sys
                )

            # ── 1. Exact-match response cache ──────────────────────────────
            if self._resp_cache and self._resp_cache.enabled:
                cached = self._resp_cache.get(self.session_id or "", user_input)
                if cached:
                    self._record_cache("response", hit=True)
                    return cached
            self._record_cache("response", hit=False)

            # ── 2. Semantic cache (near-duplicate queries) ─────────────────
            query_emb: Optional[List[float]] = None
            if self._sem_cache and self._sem_cache.enabled:
                query_emb = self._embed_query(user_input)
                if query_emb:
                    cached = self._sem_cache.lookup(query_emb, self.session_id or "")
                    if cached:
                        self._record_cache("semantic", hit=True)
                        return cached
            self._record_cache("semantic", hit=False)

            from multi_doc_chat.src.monitoring.tracing import span

            # ── 3. Contextualize query with chat history ───────────────────
            with span("rag.contextualize"):
                standalone = self._contextualize_query(user_input, chat_history)

            # ── 4. Retrieve documents (hybrid BM25+FAISS, optional HyDE) ──
            with span("rag.retrieval", {"session_id": self.session_id or ""}):
                docs = self._fetch_docs(standalone)

            if self._metrics:
                self._metrics.record_retrieval("hybrid", len(docs))

            # ── 5. Cross-encoder rerank ────────────────────────────────────
            if self._reranker and self._reranker.is_available:
                with span("rag.rerank"):
                    scored = self._reranker.rerank_with_scores(standalone, docs)
                    docs = [d for _, d in scored]
                    if scored and self._metrics:
                        self._metrics.rerank_top_score.observe(scored[0][0])
                        self._metrics.record_retrieval("reranked", len(docs))

            # ── 6. Format context ──────────────────────────────────────────
            context = self._format_docs(docs)

            # ── 7. LLM answer (call qa_prompt + llm directly — no double retrieval) ──
            with span("rag.llm_call"):
                answer = self._generate_answer(user_input, context, chat_history)

            if not answer:
                return "No answer generated."

            # ── 8. Validate ────────────────────────────────────────────────
            try:
                answer = ChatAnswer(answer=str(answer)).answer
            except ValidationError as ve:
                log.error("Invalid answer", error=str(ve))

            # ── 9. Token tracking + cost ───────────────────────────────────
            if self._token_tracker:
                try:
                    model_name = self._get_model_name()
                    prompt_text = f"{user_input} {context}"
                    record = self._token_tracker.count_and_charge(
                        model=model_name,
                        input_text=prompt_text,
                        output_text=answer,
                    )
                    if self._metrics:
                        self._metrics.record_tokens(
                            model=model_name,
                            input_tokens=record.input_tokens,
                            output_tokens=record.output_tokens,
                            cost_usd=record.cost_usd,
                        )
                except Exception as te:
                    log.warning("Token tracking failed", error=str(te))

            # ── 10. Store in caches ────────────────────────────────────────
            if self._resp_cache and self._resp_cache.enabled:
                self._resp_cache.set(self.session_id or "", user_input, answer)
            if self._sem_cache and self._sem_cache.enabled and query_emb:
                self._sem_cache.store(user_input, query_emb, answer, self.session_id or "")

            elapsed = time.perf_counter() - t_start
            log.info(
                "RAG invoke complete",
                session_id=self.session_id,
                latency_ms=round(elapsed * 1000, 1),
                answer_preview=answer[:120],
            )
            return answer

        except DocumentPortalException:
            raise
        except Exception as e:
            log.error("ConversationalRAG invoke failed", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", sys)

    # ── Internals ──────────────────────────────────────────────────────────

    def _build_lcel_chain(self):
        if self.retriever is None:
            raise DocumentPortalException("No retriever set before building chain", sys)

        question_rewriter = (
            {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
            | self.contextualize_prompt
            | self.llm
            | StrOutputParser()
        )

        retrieve_docs = question_rewriter | self.retriever | self._format_docs

        self.chain = (
            {
                "context": retrieve_docs,
                "input": itemgetter("input"),
                "chat_history": itemgetter("chat_history"),
            }
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )
        log.info("LCEL chain built", session_id=self.session_id)

    def _contextualize_query(
        self, user_input: str, chat_history: List[BaseMessage]
    ) -> str:
        """Rewrite query as a standalone question using chat history."""
        if not chat_history:
            return user_input
        try:
            chain = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )
            return chain.invoke({"input": user_input, "chat_history": chat_history})
        except Exception as e:
            log.warning("Query contextualization failed — using raw input", error=str(e))
            return user_input

    def _fetch_docs(self, standalone_query: str) -> List[Document]:
        """Retrieve documents, optionally via HyDE."""
        try:
            adv = self.cfg.get("advanced_rag", {}).get("hyde", {})
            if adv.get("enabled", False):
                try:
                    from multi_doc_chat.src.retrieval.hyde import HyDERetriever
                    hyde = HyDERetriever(
                        llm=self.llm,
                        base_retriever=self.retriever,
                        num_hypothetical=adv.get("num_hypothetical", 1),
                    )
                    return hyde.retrieve(standalone_query)
                except Exception as he:
                    log.warning("HyDE failed, falling back to direct retrieval", error=str(he))
            return self.retriever.invoke(standalone_query)
        except Exception as e:
            log.error("Document retrieval failed", error=str(e))
            return []

    def _generate_answer(
        self, user_input: str, context: str, chat_history: List[BaseMessage]
    ) -> str:
        """Call qa_prompt + LLM directly (no re-retrieval)."""
        qa_chain = self.qa_prompt | self.llm | StrOutputParser()
        return qa_chain.invoke({
            "input": user_input,
            "context": context,
            "chat_history": chat_history,
        })

    def _try_load_hybrid(
        self,
        index_path: str,
        vs: FAISS,
        dense_retriever,
        k: int,
        search_type: str,
        search_kwargs: dict,
    ):
        """Attempt to reconstruct HybridRetriever from saved BM25 pickle."""
        from pathlib import Path
        bm25_path = Path(index_path) / "bm25_index.pkl"

        adv_cfg = self.cfg.get("advanced_rag", {}).get("hybrid_search", {})
        if not adv_cfg.get("enabled", True) or not bm25_path.exists():
            return dense_retriever

        try:
            from multi_doc_chat.src.retrieval.hybrid_retriever import HybridRetriever
            from langchain.schema import Document as LCDoc

            bm25_obj, doc_texts = HybridRetriever.load_bm25(bm25_path)

            # Reconstruct minimal Document list for BM25 scoring
            docs = [LCDoc(page_content=t) for t in doc_texts]

            hybrid = HybridRetriever(
                dense_retriever=dense_retriever,
                documents=docs,
                bm25_weight=adv_cfg.get("bm25_weight", 0.3),
                dense_weight=adv_cfg.get("dense_weight", 0.7),
                rrf_k=adv_cfg.get("rrf_k", 60),
                top_k=k,
                fetch_n=adv_cfg.get("fetch_n", 20),
            )
            # Replace the freshly-built BM25 with the loaded one
            object.__setattr__(hybrid, "_bm25", bm25_obj)
            return hybrid
        except Exception as exc:
            log.warning("Could not load HybridRetriever — using dense-only", error=str(exc))
            return dense_retriever

    def _embed_query(self, text: str) -> Optional[List[float]]:
        try:
            if self._embeddings is None:
                self._embeddings = ModelLoader().load_embeddings()
            return self._embeddings.embed_query(text)
        except Exception as exc:
            log.warning("Embedding failed for semantic cache", error=str(exc))
            return None

    def _get_model_name(self) -> str:
        llm_cfg = self.cfg.get("llm", {})
        provider = os.getenv("LLM_PROVIDER", "groq")
        return llm_cfg.get(provider, {}).get("model_name", "unknown")

    def _build_prompt_text(
        self,
        user_input: str,
        context: str,
        chat_history: List[BaseMessage],
    ) -> str:
        history_text = " ".join(
            m.content for m in chat_history if hasattr(m, "content")
        )
        return f"{history_text} {user_input} {context}"

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _record_cache(self, cache_type: str, hit: bool) -> None:
        if self._metrics:
            if hit:
                self._metrics.record_cache_hit(cache_type)
            else:
                self._metrics.record_cache_miss(cache_type)

    # ── Component initializers (all safe) ──────────────────────────────────

    def _init_reranker(self):
        try:
            adv = self.cfg.get("advanced_rag", {}).get("reranker", {})
            if not adv.get("enabled", True):
                return None
            from multi_doc_chat.src.retrieval.reranker import CrossEncoderReranker
            return CrossEncoderReranker(
                model_name=adv.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                top_n=adv.get("top_n", 5),
                batch_size=adv.get("batch_size", 32),
            )
        except Exception as e:
            log.warning("Reranker init failed", error=str(e))
            return None

    def _init_semantic_cache(self):
        try:
            cache_cfg = self.cfg.get("cache", {})
            sem_cfg = cache_cfg.get("semantic", {})
            if not sem_cfg.get("enabled", True):
                return None
            from multi_doc_chat.src.cache.semantic_cache import SemanticCache
            return SemanticCache(
                redis_url=os.getenv("REDIS_URL", cache_cfg.get("redis_url", "redis://localhost:6379/0")),
                similarity_threshold=sem_cfg.get("similarity_threshold", 0.95),
                ttl_seconds=sem_cfg.get("ttl_seconds", 3600),
            )
        except Exception as e:
            log.warning("SemanticCache init failed", error=str(e))
            return None

    def _init_response_cache(self):
        try:
            cache_cfg = self.cfg.get("cache", {})
            resp_cfg = cache_cfg.get("response", {})
            if not resp_cfg.get("enabled", True):
                return None
            from multi_doc_chat.src.cache.response_cache import ResponseCache
            return ResponseCache(
                redis_url=os.getenv("REDIS_URL", cache_cfg.get("redis_url", "redis://localhost:6379/0")),
                ttl_seconds=resp_cfg.get("ttl_seconds", 300),
            )
        except Exception as e:
            log.warning("ResponseCache init failed", error=str(e))
            return None

    def _init_token_tracker(self):
        try:
            cost_cfg = self.cfg.get("cost", {})
            if not cost_cfg.get("enabled", True):
                return None
            from multi_doc_chat.src.cost.token_tracker import TokenTracker
            return TokenTracker(
                session_id=self.session_id or "unknown",
                budget_usd=cost_cfg.get("budget_per_session_usd", 1.0),
                pricing=cost_cfg.get("pricing", {}),
                warn_at_fraction=cost_cfg.get("warn_at_fraction", 0.8),
            )
        except Exception as e:
            log.warning("TokenTracker init failed", error=str(e))
            return None

    def _load_metrics(self):
        try:
            from multi_doc_chat.src.monitoring.metrics import get_metrics
            return get_metrics()
        except Exception:
            return None
