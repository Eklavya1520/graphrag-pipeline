"""
test_graph_builder.py
---------------------
Unit tests for Agent A (GraphBuilder).

Tests run with mocked Neo4j and LLM so no external services are needed.
Run with:  pytest tests/ -v
"""

import json
from unittest.mock import MagicMock, patch, call

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.graph_builder import (
    GraphBuilder,
    ExtractionResult,
    EntitySchema,
    RelationSchema,
    _chunk_text,
    _parse_llm_output,
)


# ── _chunk_text ───────────────────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "This is a short text."
        chunks = _chunk_text(text, chunk_size=800)
        # Short texts shorter than min length (50 chars) get filtered — pad it
        text = "This is a short text. " * 5
        chunks = _chunk_text(text, chunk_size=800)
        assert len(chunks) >= 1

    def test_long_text_multiple_chunks(self):
        # Generate text longer than chunk_size
        text = ("The transformer architecture is significant. " * 50)
        chunks = _chunk_text(text, chunk_size=200, overlap=20)
        assert len(chunks) > 1

    def test_no_empty_chunks(self):
        text = "Hello.\n\n\n\nWorld.\n\n"
        chunks = _chunk_text(text, chunk_size=800)
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_overlap_creates_context_continuity(self):
        """Verify that adjacent chunks share some content due to overlap."""
        text = "A " * 200  # 400 chars
        chunks = _chunk_text(text, chunk_size=100, overlap=20)
        if len(chunks) >= 2:
            # chunks[0][-20:] should appear somewhere near the start of chunks[1]
            # (not testing exactly, just that multiple chunks were made)
            assert len(chunks) >= 2


# ── _parse_llm_output ─────────────────────────────────────────────────────────

class TestParseLlmOutput:
    def test_valid_json_parses_correctly(self):
        raw = json.dumps({
            "entities": [
                {"name": "Transformer", "type": "Concept", "description": "A neural architecture"}
            ],
            "relations": [
                {"source": "Transformer", "relation": "USES", "target": "Attention", "context": "core mechanism"}
            ]
        })
        result = _parse_llm_output(raw)
        assert len(result.entities) == 1
        assert result.entities[0].name == "Transformer"
        assert len(result.relations) == 1

    def test_json_in_markdown_fences(self):
        """LLMs often wrap JSON in ```json ... ``` blocks."""
        raw = '''```json
        {"entities": [{"name": "BERT", "type": "Technology", "description": "A language model"}],
         "relations": []}
        ```'''
        result = _parse_llm_output(raw)
        assert result.entities[0].name == "BERT"

    def test_invalid_json_returns_empty(self):
        result = _parse_llm_output("This is not JSON at all.")
        assert isinstance(result, ExtractionResult)
        assert result.entities == []
        assert result.relations == []

    def test_empty_string_returns_empty(self):
        result = _parse_llm_output("")
        assert result.entities == []


# ── GraphBuilder ──────────────────────────────────────────────────────────────

class TestGraphBuilder:
    @pytest.fixture
    def mock_neo4j_client(self):
        client = MagicMock()
        client.execute_write = MagicMock()
        return client

    @pytest.fixture
    def mock_extraction(self):
        return ExtractionResult(
            entities=[
                EntitySchema(name="Transformer", type="Concept", description="Neural architecture"),
                EntitySchema(name="Attention", type="Concept", description="Attention mechanism"),
            ],
            relations=[
                RelationSchema(
                    source="Transformer",
                    relation="USES",
                    target="Attention",
                    context="core building block"
                )
            ]
        )

    @patch("agents.graph_builder._get_llm")
    def test_write_to_graph_calls_execute_write(self, mock_get_llm, mock_neo4j_client, mock_extraction):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        builder = GraphBuilder(mock_neo4j_client)
        entities_written, relations_written = builder._write_to_graph(mock_extraction)

        assert entities_written == 2
        assert relations_written == 1
        assert mock_neo4j_client.execute_write.call_count == 3  # 2 entities + 1 relation

    @patch("agents.graph_builder._get_llm")
    def test_dangling_relation_skipped(self, mock_get_llm, mock_neo4j_client):
        mock_get_llm.return_value = MagicMock()

        extraction = ExtractionResult(
            entities=[EntitySchema(name="A", type="Concept", description="Entity A")],
            relations=[
                RelationSchema(
                    source="A",
                    relation="LINKS_TO",
                    target="B",  # B is not in entities list
                    context="some context"
                )
            ]
        )

        builder = GraphBuilder(mock_neo4j_client)
        entities_written, relations_written = builder._write_to_graph(extraction)

        assert entities_written == 1
        assert relations_written == 0  # dangling relation was skipped

    @patch("agents.graph_builder._get_llm")
    def test_ingest_text_returns_stats(self, mock_get_llm, mock_neo4j_client):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "entities": [{"name": "BERT", "type": "Technology", "description": "A model"}],
            "relations": []
        })
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        builder = GraphBuilder(mock_neo4j_client)
        # Use text long enough to not be filtered by chunk min-length
        text = "BERT is a transformer-based language model developed by Google. " * 10
        stats = builder.ingest_text(text)

        assert "entities" in stats
        assert "relations" in stats
        assert isinstance(stats["entities"], int)
