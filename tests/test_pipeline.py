"""
test_pipeline.py
----------------
Integration-level tests for the full LangGraph pipeline.
Neo4j and LLM calls are mocked so these run without external dependencies.

Run with:  pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline.state import initial_state, PipelineState
from pipeline.edges import route_after_confidence_scoring, route_after_graph_retrieval


# ── State helpers ──────────────────────────────────────────────────────────────

class TestInitialState:
    def test_initial_state_has_correct_defaults(self):
        state = initial_state("Test question?")
        assert state["query"] == "Test question?"
        assert state["rewritten_query"] == ""
        assert state["graph_context"] == []
        assert state["vector_context"] == []
        assert state["confidence"] == 0.0
        assert state["iterations"] == 0
        assert state["low_confidence_warning"] is False

    def test_initial_state_query_is_preserved(self):
        q = "What is the transformer architecture?"
        state = initial_state(q)
        assert state["query"] == q


# ── Edge routing ───────────────────────────────────────────────────────────────

class TestEdgeRouting:
    def _make_state(self, **kwargs) -> dict:
        base = initial_state("test")
        base.update(kwargs)
        return base

    # route_after_graph_retrieval
    def test_sufficient_graph_context_routes_to_generate(self):
        state = self._make_state(graph_context=["ctx1", "ctx2", "ctx3"])
        assert route_after_graph_retrieval(state) == "generate"

    def test_sparse_graph_context_routes_to_vector(self):
        state = self._make_state(graph_context=["only one item"])
        assert route_after_graph_retrieval(state) == "vector"

    def test_empty_graph_context_routes_to_vector(self):
        state = self._make_state(graph_context=[])
        assert route_after_graph_retrieval(state) == "vector"

    # route_after_confidence_scoring
    def test_high_confidence_routes_to_end(self):
        state = self._make_state(confidence=0.9, iterations=1)
        assert route_after_confidence_scoring(state) == "end"

    def test_low_confidence_within_limit_routes_to_retry(self):
        # default max_correction_iterations = 3 from config
        state = self._make_state(confidence=0.3, iterations=1)
        result = route_after_confidence_scoring(state)
        assert result == "retry"

    def test_max_iterations_routes_to_flag_low(self):
        # When iterations == max_correction_iterations (3), should flag
        state = self._make_state(confidence=0.2, iterations=3)
        result = route_after_confidence_scoring(state)
        assert result == "flag_low"

    def test_confidence_exactly_at_threshold_routes_to_end(self):
        # confidence == threshold should be accepted
        state = self._make_state(confidence=0.6, iterations=1)
        result = route_after_confidence_scoring(state)
        assert result == "end"

    def test_confidence_just_below_threshold_routes_to_retry(self):
        state = self._make_state(confidence=0.59, iterations=1)
        result = route_after_confidence_scoring(state)
        assert result == "retry"


# ── Node tests (mocked) ────────────────────────────────────────────────────────

class TestNodes:
    @patch("pipeline.nodes._get_rewriter")
    def test_rewrite_query_node_updates_state(self, mock_get_rewriter):
        from agents.query_rewriter import RewriteResult
        from pipeline.nodes import node_rewrite_query

        mock_rewriter = MagicMock()
        mock_rewriter.rewrite.return_value = RewriteResult(
            rewritten_query="What is the Transformer model architecture?",
            entity_keywords=["Transformer", "attention"],
            query_type="factual",
        )
        mock_get_rewriter.return_value = mock_rewriter

        state = initial_state("tell me about transformer")
        result = node_rewrite_query(state)

        assert result["rewritten_query"] == "What is the Transformer model architecture?"
        assert "Transformer" in result["entity_keywords"]
        assert result["query_type"] == "factual"

    @patch("pipeline.nodes._get_generator")
    def test_generate_answer_node_updates_state(self, mock_get_generator):
        from pipeline.nodes import node_generate_answer

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "The transformer uses attention mechanisms."
        mock_get_generator.return_value = mock_generator

        state = initial_state("what is attention?")
        state["rewritten_query"] = "What is the attention mechanism in transformers?"
        state["graph_context"] = ["Transformer --[USES]--> Attention"]
        state["vector_context"] = []

        result = node_generate_answer(state)
        assert result["answer"] == "The transformer uses attention mechanisms."

    @patch("pipeline.nodes._get_generator")
    def test_score_confidence_node_increments_iterations(self, mock_get_generator):
        from pipeline.nodes import node_score_confidence

        mock_generator = MagicMock()
        mock_generator.score_confidence.return_value = 0.85
        mock_get_generator.return_value = mock_generator

        state = initial_state("question")
        state["answer"] = "Some answer"
        state["iterations"] = 0

        result = node_score_confidence(state)
        assert result["confidence"] == 0.85
        assert result["iterations"] == 1  # incremented from 0

    def test_flag_low_confidence_node_adds_warning(self):
        from pipeline.nodes import node_flag_low_confidence

        state = initial_state("question")
        state["answer"] = "Some uncertain answer."

        result = node_flag_low_confidence(state)
        assert "Low confidence" in result["answer"]
        assert result["low_confidence_warning"] is True
