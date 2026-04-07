"""
HyDE — Hypothetical Document Embeddings  (Gao et al., 2022)

Insight: embedding a *hypothetical answer* to the query often
         aligns better with real answer documents than embedding
         the query itself, because queries and answers live in
         different parts of the embedding space.

Trade-off: +1 LLM call per query (adds ~200–400 ms latency).
           Best enabled for complex, indirect, or technical queries.

Usage:
    hyde = HyDERetriever(llm=llm, base_retriever=retriever)
    docs = hyde.retrieve("What causes inflation?")
"""
from __future__ import annotations

from typing import List

from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from multi_doc_chat.logger import GLOBAL_LOGGER as log


_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Write a short, factual passage (2-3 sentences) "
     "that directly answers the following question. "
     "Write as if you are extracting text from a relevant document. "
     "Do not say 'I' or mention that you are an AI."),
    ("human", "{question}"),
])


class HyDERetriever:
    """
    Wraps any base retriever with HyDE query expansion.

    Instead of embedding the raw query, we:
      1. Ask the LLM to write a short hypothetical answer.
      2. Embed that hypothetical text.
      3. Use it to search the vector store.
    """

    def __init__(self, llm, base_retriever, num_hypothetical: int = 1):
        self.llm = llm
        self.base_retriever = base_retriever
        self.num_hypothetical = num_hypothetical
        self._chain = _HYDE_PROMPT | llm | StrOutputParser()
        log.info("HyDERetriever initialized", num_hypothetical=num_hypothetical)

    def retrieve(self, query: str) -> List[Document]:
        """Generate hypothetical doc(s), retrieve with each, deduplicate."""
        try:
            hypothetical_docs: List[str] = []
            for _ in range(self.num_hypothetical):
                hyp = self._chain.invoke({"question": query})
                hypothetical_docs.append(hyp)
                log.debug("HyDE generated hypothetical doc", preview=hyp[:120])

            seen: set[str] = set()
            results: List[Document] = []
            for hyp_text in hypothetical_docs:
                for doc in self.base_retriever.invoke(hyp_text):
                    key = doc.page_content[:256]
                    if key not in seen:
                        seen.add(key)
                        results.append(doc)

            log.info(
                "HyDE retrieval complete",
                query_preview=query[:60],
                hypotheticals=len(hypothetical_docs),
                unique_docs=len(results),
            )
            return results

        except Exception as exc:
            log.error("HyDE failed — falling back to direct retrieval", error=str(exc))
            return self.base_retriever.invoke(query)
