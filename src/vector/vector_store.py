"""
vector_store.py
---------------
FAISS-backed vector store for fallback retrieval when graph context is thin.
Optionally swappable for Chroma via VECTOR_STORE_TYPE env var.

Usage:
    store = VectorStore()
    store.add_documents(["doc text 1", "doc text 2"])
    store.save()

    # Later:
    store = VectorStore()
    store.load()
    results = store.search("attention mechanism", top_k=5)
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from config import cfg

logger = logging.getLogger(__name__)


def _get_embedder():
    """Return the appropriate LangChain embeddings object based on config."""
    if cfg.llm_provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=cfg.google_api_key,
        )
    else:
        # Default to OpenAI
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=cfg.embedding_model,
            openai_api_key=cfg.openai_api_key,
        )


class VectorStore:
    """
    Thin wrapper that handles both FAISS and Chroma backends.
    Documents are stored as raw text alongside their embeddings so we can
    return the original text in search results.
    """

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._path = Path(store_path or cfg.vector_store_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._embedder = _get_embedder()
        self._store_type = cfg.vector_store_type.lower()
        self._backend = None  # lazy initialised
        self._docs: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def add_documents(self, texts: list[str]) -> None:
        """Embed and index a list of text strings."""
        if not texts:
            return

        logger.info("Embedding %d documents for vector store…", len(texts))

        if self._store_type == "chroma":
            self._add_chroma(texts)
        else:
            self._add_faiss(texts)

        self._docs.extend(texts)
        logger.info("Vector store now has %d documents", len(self._docs))

    def search(self, query: str, top_k: int | None = None) -> list[str]:
        """Return top-k most similar document texts for a query."""
        k = top_k or cfg.vector_top_k
        if self._backend is None:
            logger.warning("Vector store not loaded — call load() first or add documents.")
            return []

        if self._store_type == "chroma":
            return self._search_chroma(query, k)
        else:
            return self._search_faiss(query, k)

    def save(self) -> None:
        if self._store_type == "chroma":
            # Chroma persists automatically
            pass
        else:
            self._save_faiss()

    def load(self) -> bool:
        """Load from disk. Returns True if successful."""
        if self._store_type == "chroma":
            return self._load_chroma()
        else:
            return self._load_faiss()

    # ── FAISS backend ─────────────────────────────────────────────────────────

    def _add_faiss(self, texts: list[str]) -> None:
        import faiss

        embeddings = self._embedder.embed_documents(texts)
        vectors = np.array(embeddings, dtype=np.float32)

        if self._backend is None:
            dim = vectors.shape[1]
            self._backend = faiss.IndexFlatL2(dim)

        self._backend.add(vectors)

    def _search_faiss(self, query: str, top_k: int) -> list[str]:
        q_vec = np.array(
            [self._embedder.embed_query(query)], dtype=np.float32
        )
        distances, indices = self._backend.search(q_vec, top_k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self._docs):
                results.append(self._docs[idx])
        return results

    def _save_faiss(self) -> None:
        import faiss
        if self._backend is not None:
            faiss.write_index(self._backend, str(self._path / "index.faiss"))
            with open(self._path / "docs.pkl", "wb") as f:
                pickle.dump(self._docs, f)
            logger.info("Saved FAISS index to %s", self._path)

    def _load_faiss(self) -> bool:
        import faiss
        index_path = self._path / "index.faiss"
        docs_path  = self._path / "docs.pkl"
        if not index_path.exists():
            return False
        self._backend = faiss.read_index(str(index_path))
        with open(docs_path, "rb") as f:
            self._docs = pickle.load(f)
        logger.info("Loaded FAISS index (%d docs)", len(self._docs))
        return True

    # ── Chroma backend ────────────────────────────────────────────────────────

    def _add_chroma(self, texts: list[str]) -> None:
        import chromadb
        if self._backend is None:
            client = chromadb.PersistentClient(path=str(self._path))
            self._backend = client.get_or_create_collection("graphrag")

        embeddings = self._embedder.embed_documents(texts)
        ids = [f"doc_{len(self._docs) + i}" for i in range(len(texts))]
        self._backend.add(embeddings=embeddings, documents=texts, ids=ids)

    def _search_chroma(self, query: str, top_k: int) -> list[str]:
        q_vec = self._embedder.embed_query(query)
        results = self._backend.query(query_embeddings=[q_vec], n_results=top_k)
        return results.get("documents", [[]])[0]

    def _load_chroma(self) -> bool:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(self._path))
            self._backend = client.get_collection("graphrag")
            return True
        except Exception:
            return False
