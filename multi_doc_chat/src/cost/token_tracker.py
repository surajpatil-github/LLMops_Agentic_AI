"""
Token Tracker — counts tokens and estimates cost per session.

Why this matters:
  - Groq llama-3.3-70b: $0.59/$0.79 per 1M in/out tokens
  - A single long-context RAG call can consume 3–5 k tokens
  - Without tracking, costs are invisible until the bill arrives

Implementation:
  - tiktoken for exact token counting (OpenAI tokenizer, close enough for Groq/Gemini)
  - Configurable per-model pricing from config.yaml
  - Budget limit: raises BudgetExceededError when session exceeds cap
  - Exposes totals for Prometheus + structured logs
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from multi_doc_chat.logger import GLOBAL_LOGGER as log

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
    # cl100k_base works well as a proxy for Groq Llama & Google models
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_AVAILABLE = False
    _ENCODER = None


class BudgetExceededError(Exception):
    """Raised when a session's estimated cost would exceed the configured cap."""


@dataclass
class CallRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


class TokenTracker:
    """
    Per-session token and cost tracker.

    Usage:
        tracker = TokenTracker(session_id="s123", budget_usd=1.0, pricing=cfg)
        tracker.count_and_charge(model="llama-3.3-70b-versatile",
                                  input_text=prompt, output_text=answer)
    """

    def __init__(
        self,
        session_id: str,
        budget_usd: float = 1.0,
        pricing: Optional[Dict[str, Dict[str, float]]] = None,
        warn_at_fraction: float = 0.8,
    ):
        self.session_id = session_id
        self.budget_usd = budget_usd
        self.warn_at_fraction = warn_at_fraction
        self.pricing = pricing or {}

        self._records: List[CallRecord] = []
        self._total_input = 0
        self._total_output = 0
        self._total_cost = 0.0

    # ── Public API ─────────────────────────────────────────────────────────

    def count_and_charge(
        self,
        model: str,
        input_text: str,
        output_text: str,
    ) -> CallRecord:
        """
        Count tokens for one LLM call and update session totals.
        Raises BudgetExceededError if the call would exceed the budget.
        """
        in_tok = self.count_tokens(input_text)
        out_tok = self.count_tokens(output_text)
        cost = self._estimate_cost(model, in_tok, out_tok)

        if self.budget_usd > 0 and (self._total_cost + cost) > self.budget_usd:
            raise BudgetExceededError(
                f"Session {self.session_id}: budget ${self.budget_usd:.4f} exceeded "
                f"(current=${self._total_cost:.4f}, this call=${cost:.4f})"
            )

        record = CallRecord(
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )
        self._records.append(record)
        self._total_input += in_tok
        self._total_output += out_tok
        self._total_cost += cost

        self._maybe_warn()
        log.info(
            "Token usage recorded",
            session_id=self.session_id,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            call_cost_usd=round(cost, 6),
            session_total_usd=round(self._total_cost, 6),
        )
        return record

    def count_tokens(self, text: str) -> int:
        if _TIKTOKEN_AVAILABLE and _ENCODER:
            try:
                return len(_ENCODER.encode(text))
            except Exception:
                pass
        # Rough fallback: ~4 chars per token
        return max(1, len(text) // 4)

    @property
    def total_cost_usd(self) -> float:
        return round(self._total_cost, 6)

    @property
    def total_input_tokens(self) -> int:
        return self._total_input

    @property
    def total_output_tokens(self) -> int:
        return self._total_output

    @property
    def budget_remaining_usd(self) -> float:
        return max(0.0, round(self.budget_usd - self._total_cost, 6))

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "calls": len(self._records),
            "total_input_tokens": self._total_input,
            "total_output_tokens": self._total_output,
            "total_cost_usd": self.total_cost_usd,
            "budget_remaining_usd": self.budget_remaining_usd,
        }

    # ── Internals ──────────────────────────────────────────────────────────

    def _estimate_cost(self, model: str, in_tok: int, out_tok: int) -> float:
        prices = self.pricing.get(model, {})
        in_price = prices.get("input_per_1m", 0.0)
        out_price = prices.get("output_per_1m", 0.0)
        return (in_tok * in_price + out_tok * out_price) / 1_000_000

    def _maybe_warn(self) -> None:
        if self.budget_usd <= 0:
            return
        fraction_used = self._total_cost / self.budget_usd
        if fraction_used >= self.warn_at_fraction:
            log.warning(
                "Session approaching budget limit",
                session_id=self.session_id,
                used_usd=round(self._total_cost, 6),
                budget_usd=self.budget_usd,
                fraction=round(fraction_used, 3),
            )
