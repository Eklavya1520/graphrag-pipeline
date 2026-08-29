"""
edges.py
--------
Conditional edge functions for the LangGraph pipeline.

These functions inspect the current state and return a string that tells
LangGraph which node to route to next.

Routing logic
-------------
After score_confidence:
  - confidence >= threshold AND iterations == 1  →  "end"         (great answer, done)
  - confidence <  threshold AND iterations <  max →  "retry"       (rewrite and try again)
  - iterations >= max                             →  "flag_low"    (exhausted retries)

After retrieve_from_graph:
  - context is sufficient                         →  "generate"    (skip vector fallback)
  - context is sparse                             →  "vector"      (trigger fallback)
"""

import logging

from pipeline.state import PipelineState
from config import cfg

logger = logging.getLogger(__name__)


def route_after_graph_retrieval(state: PipelineState) -> str:
    """
    Decide whether to call the vector fallback or go straight to answer generation.
    Returns 'generate' or 'vector'.
    """
    graph_context = state.get("graph_context", [])

    if len(graph_context) >= 2:
        logger.debug("[edge] Graph context sufficient (%d items) — skipping vector", len(graph_context))
        return "generate"

    logger.debug("[edge] Graph context sparse (%d items) — triggering vector fallback", len(graph_context))
    return "vector"


def route_after_confidence_scoring(state: PipelineState) -> str:
    """
    Core self-correction routing.

    Returns one of:
      "end"       → accept answer, terminate pipeline
      "retry"     → rewrite query and retry from the top
      "flag_low"  → max iterations hit, flag low confidence and terminate
    """
    confidence  = state.get("confidence", 0.0)
    iterations  = state.get("iterations", 0)
    max_iters   = cfg.max_correction_iterations
    threshold   = cfg.confidence_threshold

    if confidence >= threshold:
        logger.info("[edge] Confidence %.2f >= %.2f — accepting answer", confidence, threshold)
        return "end"

    if iterations >= max_iters:
        logger.info(
            "[edge] Max iterations (%d) reached with confidence %.2f — flagging low confidence",
            max_iters, confidence,
        )
        return "flag_low"

    logger.info(
        "[edge] Confidence %.2f < %.2f — retrying (iteration %d/%d)",
        confidence, threshold, iterations, max_iters,
    )
    return "retry"
