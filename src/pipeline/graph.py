"""
graph.py
--------
Assembles and compiles the LangGraph StateGraph.

Graph topology
--------------

  [START]
     |
     v
  rewrite_query  (Agent B)
     |
     v
  retrieve_from_graph  (Neo4j)
     |
     +--[graph sufficient]--> generate_answer
     |
     +--[graph sparse]------> retrieve_from_vector --> generate_answer
                                                             |
                                                             v
                                                      score_confidence
                                                             |
                                          +--[conf >= 0.6]--+--[END]
                                          |
                                          +--[conf < 0.6, iter < max]--> rewrite_query (retry)
                                          |
                                          +--[iter >= max]-----------> flag_low_confidence --> [END]
"""

import logging

from langgraph.graph import END, START, StateGraph

from pipeline.state import PipelineState
from pipeline.nodes import (
    node_flag_low_confidence,
    node_generate_answer,
    node_retrieve_from_graph,
    node_retrieve_from_vector,
    node_rewrite_query,
    node_score_confidence,
)
from pipeline.edges import (
    route_after_confidence_scoring,
    route_after_graph_retrieval,
)

logger = logging.getLogger(__name__)


def build_graph():
    """
    Construct and compile the LangGraph pipeline.

    Returns a compiled graph that can be invoked with:
        graph = build_graph()
        result = graph.invoke(initial_state("your question"))
    """
    builder = StateGraph(PipelineState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    builder.add_node("rewrite_query",        node_rewrite_query)
    builder.add_node("retrieve_from_graph",  node_retrieve_from_graph)
    builder.add_node("retrieve_from_vector", node_retrieve_from_vector)
    builder.add_node("generate_answer",      node_generate_answer)
    builder.add_node("score_confidence",     node_score_confidence)
    builder.add_node("flag_low_confidence",  node_flag_low_confidence)

    # ── Entry point ────────────────────────────────────────────────────────────
    builder.add_edge(START, "rewrite_query")

    # ── Linear edges ──────────────────────────────────────────────────────────
    builder.add_edge("rewrite_query",        "retrieve_from_graph")
    builder.add_edge("retrieve_from_vector", "generate_answer")
    builder.add_edge("generate_answer",      "score_confidence")
    builder.add_edge("flag_low_confidence",  END)

    # ── Conditional: graph retrieval result ───────────────────────────────────
    builder.add_conditional_edges(
        "retrieve_from_graph",
        route_after_graph_retrieval,
        {
            "generate": "generate_answer",
            "vector":   "retrieve_from_vector",
        },
    )

    # ── Conditional: self-correction routing ──────────────────────────────────
    builder.add_conditional_edges(
        "score_confidence",
        route_after_confidence_scoring,
        {
            "end":      END,
            "retry":    "rewrite_query",
            "flag_low": "flag_low_confidence",
        },
    )

    compiled = builder.compile()
    logger.info("LangGraph pipeline compiled successfully")
    return compiled


# Module-level singleton — import and reuse this
pipeline = build_graph()
