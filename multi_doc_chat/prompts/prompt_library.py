"""
Prompt library for MultiDocChat.

Prompts registered here:
  contextualize_question  — rewrite user query as standalone using chat history
  context_qa              — answer from retrieved context only
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── Contextualization ──────────────────────────────────────────────────────
contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a conversation history and the user's latest question, "
        "rewrite the question as a self-contained standalone question that "
        "can be understood without the prior conversation. "
        "Do NOT answer it — only reformulate if needed; return it unchanged otherwise."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ── Grounded QA ────────────────────────────────────────────────────────────
context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a precise, factual assistant. Answer the user's question using "
        "ONLY the information in the retrieved context below. "
        "If the answer is not present in the context, say exactly: \"I don't know.\"\n"
        "Rules:\n"
        "  - Cite the source document or section when possible.\n"
        "  - Be concise: 1–4 sentences unless the question requires more detail.\n"
        "  - Never fabricate facts or use prior knowledge outside the context.\n\n"
        "Context:\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ── Registry ───────────────────────────────────────────────────────────────
PROMPT_REGISTRY: dict[str, ChatPromptTemplate] = {
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
}
