"""
state.py
--------
Defines the LangGraph state schema for the GraphRAG pipeline.

All nodes read from and write to this TypedDict.  LangGraph handles
merging between nodes automatically based on the reducer annotations.

Design notes
------------
- `query` is set once at the start and never mutated — always the original.
- `rewritten_query` is updated by Agent B and may be updated again on retry.
- `messages` uses LangChain's add_messages reducer so history accumulates
  rather than being replaced (this is the default in newer LangGraph versions).
- `iterations` acts as a safety valve to break infinite self-correction loops.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class PipelineState(TypedDict):
    # ── Input (set once, never changed) ──────────────────────────────────────
    query: str

    # ── Agent B output (may be updated each iteration) ───────────────────────
    rewritten_query: str
    entity_keywords: list[str]
    query_type: str  # "factual" | "relational" | "comparative" | "procedural"

    # ── Retrieval results ─────────────────────────────────────────────────────
    graph_context: list[str]    # from Neo4j graph retriever
    vector_context: list[str]   # from FAISS fallback (may be empty)

    # ── Generation output ─────────────────────────────────────────────────────
    answer: str
    confidence: float           # 0.0 – 1.0; drives self-correction routing

    # ── Loop control ──────────────────────────────────────────────────────────
    iterations: int             # incremented each self-correction retry
    low_confidence_warning: bool  # set True if we exhaust retries

    # ── Conversation history (accumulated via add_messages reducer) ───────────
    messages: Annotated[list[AnyMessage], add_messages]


def initial_state(query: str) -> PipelineState:
    """Return a fresh state dict for a new query."""
    return PipelineState(
        query=query,
        rewritten_query="",
        entity_keywords=[],
        query_type="factual",
        graph_context=[],
        vector_context=[],
        answer="",
        confidence=0.0,
        iterations=0,
        low_confidence_warning=False,
        messages=[],
    )
