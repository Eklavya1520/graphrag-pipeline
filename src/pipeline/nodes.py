"""
nodes.py
--------
All LangGraph node functions.  Each function takes a PipelineState dict
and returns a partial dict of keys to update.

Node execution order (happy path):
  rewrite_query -> retrieve_from_graph -> retrieve_from_vector (conditional)
      -> generate_answer -> score_confidence

The self-correction loop re-enters at rewrite_query if confidence is low.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage

from agents.query_rewriter import QueryRewriter
from agents.answer_gen import AnswerGenerator
from graph.graph_retriever import GraphRetriever
from graph.neo4j_client import Neo4jClient
from vector.vector_store import VectorStore
from pipeline.state import PipelineState
from config import cfg

logger = logging.getLogger(__name__)

# Module-level singletons (shared across invocations — avoids re-initialising LLMs each call)
_rewriter: QueryRewriter | None = None
_generator: AnswerGenerator | None = None
_vector_store: VectorStore | None = None


def _get_rewriter() -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter


def _get_generator() -> AnswerGenerator:
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        loaded = _vector_store.load()
        if not loaded:
            logger.warning(
                "Vector store not found at %s — vector fallback disabled",
                cfg.vector_store_path,
            )
    return _vector_store


# ── Node: Query Rewriter (Agent B) ────────────────────────────────────────────

def node_rewrite_query(state: PipelineState) -> dict:
    """
    Agent B — rewrites the user query and extracts entity keywords.

    On subsequent iterations (self-correction retries), the rewrite prompt
    implicitly produces a different formulation because the LLM sees the
    original question each time; adding 'retry' suffix nudges variation.
    """
    query = state["query"]
    iteration = state.get("iterations", 0)

    # Nudge Agent B to try a different angle on retries
    if iteration > 0:
        query = f"{query} [retry {iteration}: approach from a different angle]"

    rewriter = _get_rewriter()
    result = rewriter.rewrite(query)

    logger.info(
        "[rewrite_query] iter=%d  kw=%s  type=%s",
        iteration, result.entity_keywords, result.query_type,
    )

    return {
        "rewritten_query": result.rewritten_query,
        "entity_keywords": result.entity_keywords,
        "query_type": result.query_type,
        "messages": [HumanMessage(content=result.rewritten_query)],
    }


# ── Node: Graph Retriever ─────────────────────────────────────────────────────

def node_retrieve_from_graph(state: PipelineState) -> dict:
    """
    Query Neo4j using the entity keywords extracted by Agent B.
    Returns graph_context — a list of human-readable relationship strings.
    """
    keywords = state.get("entity_keywords", [])
    if not keywords:
        # Fall back to nouns from the rewritten query
        keywords = state.get("rewritten_query", state["query"]).split()[:3]

    logger.info("[retrieve_graph] keywords=%s", keywords)

    try:
        with Neo4jClient() as client:
            retriever = GraphRetriever(client)
            context = retriever.retrieve(keywords, top_k=cfg.graph_top_k)
    except ConnectionError as exc:
        logger.error("Neo4j connection failed: %s", exc)
        context = []

    logger.info("[retrieve_graph] got %d context items", len(context))
    return {"graph_context": context}


# ── Node: Vector Fallback ─────────────────────────────────────────────────────

def node_retrieve_from_vector(state: PipelineState) -> dict:
    """
    FAISS/Chroma fallback retriever.
    Only called when graph context is thin (controlled by edge logic).
    """
    query = state.get("rewritten_query") or state["query"]
    logger.info("[retrieve_vector] falling back to vector search for: '%s'", query[:60])

    store = _get_vector_store()
    context = store.search(query, top_k=cfg.vector_top_k)

    logger.info("[retrieve_vector] got %d context items", len(context))
    return {"vector_context": context}


# ── Node: Answer Generator ────────────────────────────────────────────────────

def node_generate_answer(state: PipelineState) -> dict:
    """
    Synthesise a final answer from the combined graph + vector context.
    """
    question       = state.get("rewritten_query") or state["query"]
    graph_context  = state.get("graph_context", [])
    vector_context = state.get("vector_context", [])
    history        = state.get("messages", [])

    generator = _get_generator()
    answer = generator.generate(
        question=question,
        graph_context=graph_context,
        vector_context=vector_context,
        conversation_history=history,
    )

    logger.info("[generate_answer] answer length=%d chars", len(answer))
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ── Node: Confidence Scorer (critic) ──────────────────────────────────────────

def node_score_confidence(state: PipelineState) -> dict:
    """
    Ask the LLM to self-evaluate the generated answer.
    Updates confidence and increments the iteration counter.
    """
    question       = state.get("rewritten_query") or state["query"]
    answer         = state.get("answer", "")
    graph_context  = state.get("graph_context", [])
    vector_context = state.get("vector_context", [])
    iterations     = state.get("iterations", 0)

    generator = _get_generator()
    confidence = generator.score_confidence(
        question=question,
        answer=answer,
        graph_context=graph_context,
        vector_context=vector_context,
    )

    logger.info(
        "[score_confidence] iter=%d  confidence=%.2f  threshold=%.2f",
        iterations, confidence, cfg.confidence_threshold,
    )

    return {
        "confidence": confidence,
        "iterations": iterations + 1,
    }


# ── Node: Final output with low-confidence flag ───────────────────────────────

def node_flag_low_confidence(state: PipelineState) -> dict:
    """
    Terminal node — attaches a warning to the answer when we exhaust retries.
    """
    answer = state.get("answer", "I was unable to find a confident answer.")
    warning = (
        "\n\n---\n⚠️  **Low confidence**: the pipeline could not retrieve sufficient "
        "context after 3 attempts. This answer may be incomplete or imprecise."
    )
    return {
        "answer": answer + warning,
        "low_confidence_warning": True,
    }
