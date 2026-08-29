"""
conftest.py
-----------
Shared pytest fixtures and configuration.
"""

import sys
from pathlib import Path
import pytest

# Ensure src/ is always on the path for all test modules
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """
    Set dummy environment variables so Config() doesn't blow up when
    tests import modules that read env at import time.
    """
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-pytest")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test_password")
    monkeypatch.setenv("NEO4J_DATABASE", "neo4j")
    monkeypatch.setenv("VECTOR_STORE_TYPE", "faiss")
    monkeypatch.setenv("VECTOR_STORE_PATH", "./data/vector_store")
    monkeypatch.setenv("MAX_CORRECTION_ITERATIONS", "3")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.6")
    monkeypatch.setenv("GRAPH_TOP_K", "5")
    monkeypatch.setenv("VECTOR_TOP_K", "5")
    monkeypatch.setenv("RAGAS_LLM", "gpt-4o-mini")
