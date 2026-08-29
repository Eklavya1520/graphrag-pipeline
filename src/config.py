"""
config.py
---------
Central configuration loader. Reads from .env (or environment) and exposes
a single Config dataclass used throughout the project.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass
class Config:
    # LLM
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # Neo4j
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    neo4j_database: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))

    # Vector store
    vector_store_type: str = field(default_factory=lambda: os.getenv("VECTOR_STORE_TYPE", "faiss"))
    vector_store_path: str = field(default_factory=lambda: os.getenv("VECTOR_STORE_PATH", "./data/vector_store"))

    # Pipeline behaviour
    max_correction_iterations: int = field(
        default_factory=lambda: int(os.getenv("MAX_CORRECTION_ITERATIONS", "3"))
    )
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
    )
    graph_top_k: int = field(default_factory=lambda: int(os.getenv("GRAPH_TOP_K", "5")))
    vector_top_k: int = field(default_factory=lambda: int(os.getenv("VECTOR_TOP_K", "5")))

    # Ragas
    ragas_llm: str = field(default_factory=lambda: os.getenv("RAGAS_LLM", "gpt-4o-mini"))

    def validate(self) -> None:
        """Raise early if critical config is missing."""
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in your .env file.")
        if self.llm_provider == "google" and not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set in your .env file.")
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD is not set in your .env file.")


# Module-level singleton — import this everywhere
cfg = Config()
