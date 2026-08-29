"""
test_query_rewriter.py
----------------------
Tests for Agent B (QueryRewriter).
All LLM calls are mocked.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.query_rewriter import (
    QueryRewriter,
    RewriteResult,
    _parse_rewrite_output,
    _fallback_keyword_extraction,
)


class TestParseRewriteOutput:
    def test_well_formed_output(self):
        raw = """REWRITTEN: What is the architecture of the Transformer model in NLP?
KEYWORDS: Transformer, attention, encoder, decoder
TYPE: factual"""
        result = _parse_rewrite_output(raw, "tell me about transformer")
        assert "Transformer" in result.rewritten_query
        assert "Transformer" in result.entity_keywords
        assert result.query_type == "factual"

    def test_comparative_type_parsed(self):
        raw = """REWRITTEN: What are the differences between RNN and Transformer architectures?
KEYWORDS: RNN, Transformer
TYPE: comparative"""
        result = _parse_rewrite_output(raw, "rnn vs transformer")
        assert result.query_type == "comparative"

    def test_missing_keywords_uses_fallback(self):
        raw = """REWRITTEN: What is BERT used for in NLP?
TYPE: factual"""
        result = _parse_rewrite_output(raw, "what is bert")
        # Should fall back to extracting capitalised words from rewritten query
        assert len(result.entity_keywords) > 0

    def test_empty_raw_falls_back_gracefully(self):
        result = _parse_rewrite_output("", "original query")
        assert result.rewritten_query == "original query"

    def test_keywords_capped_at_4(self):
        raw = """REWRITTEN: Something
KEYWORDS: A, B, C, D, E, F
TYPE: factual"""
        result = _parse_rewrite_output(raw, "orig")
        assert len(result.entity_keywords) <= 4


class TestFallbackKeywordExtraction:
    def test_extracts_capitalised_words(self):
        text = "What is the BERT architecture and how does Transformer work?"
        keywords = _fallback_keyword_extraction(text)
        assert "BERT" in keywords or "Transformer" in keywords

    def test_filters_stopwords(self):
        text = "What Is The Architecture Of The Model?"
        keywords = _fallback_keyword_extraction(text)
        # Common stopwords like "What", "Is", "The", "Of" should be filtered
        assert "What" not in keywords
        assert "The" not in keywords

    def test_empty_text_returns_empty(self):
        assert _fallback_keyword_extraction("") == []


class TestQueryRewriter:
    @patch("agents.query_rewriter._get_llm")
    def test_rewrite_returns_result(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = """REWRITTEN: What is the attention mechanism in Transformer models?
KEYWORDS: attention, Transformer
TYPE: factual"""
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        rewriter = QueryRewriter()
        result = rewriter.rewrite("tell me about attention")

        assert isinstance(result, RewriteResult)
        assert "attention" in result.rewritten_query.lower() or "Attention" in result.rewritten_query
        assert result.query_type == "factual"

    @patch("agents.query_rewriter._get_llm")
    def test_rewrite_fails_open_on_exception(self, mock_get_llm):
        """If the LLM call throws, the rewriter should return the original query."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM API error")
        mock_get_llm.return_value = mock_llm

        rewriter = QueryRewriter()
        result = rewriter.rewrite("original question here")

        assert result.rewritten_query == "original question here"

    @patch("agents.query_rewriter._get_llm")
    def test_empty_query_returns_immediately(self, mock_get_llm):
        mock_get_llm.return_value = MagicMock()
        rewriter = QueryRewriter()
        result = rewriter.rewrite("")
        assert result.rewritten_query == ""
        assert result.entity_keywords == []
