"""
graph_retriever.py
------------------
Retrieves context from Neo4j given a list of entity keywords extracted
from the (rewritten) user query.

Returns a list of plain-text context strings that get fed directly
into the LLM prompt.
"""

import logging
from typing import Any

from graph.neo4j_client import Neo4jClient
from graph.cypher_templates import (
    build_neighbourhood_query,
    build_fuzzy_query,
)
from config import cfg

logger = logging.getLogger(__name__)


def _records_to_context(records: list[dict[str, Any]]) -> list[str]:
    """
    Convert raw Cypher result records into human-readable context strings.
    Example output:
      "Transformer [Concept] -- USES --> Attention Mechanism (context: 'core building block')"
    """
    seen: set[str] = set()
    ctx: list[str] = []

    for r in records:
        entity      = r.get("entity", "?")
        etype       = r.get("entity_type", "")
        edesc       = r.get("entity_desc", "")
        relation    = r.get("relation", "")
        rel_context = r.get("rel_context") or r.get("context", "")
        neighbour   = r.get("neighbour", "")

        # Entity description line
        if edesc:
            line = f"{entity} [{etype}]: {edesc}"
            if line not in seen:
                seen.add(line)
                ctx.append(line)

        # Relationship line
        if relation and neighbour:
            rel_line = f"{entity} --[{relation}]--> {neighbour}"
            if rel_context:
                rel_line += f"  (context: \"{rel_context}\")"
            if rel_line not in seen:
                seen.add(rel_line)
                ctx.append(rel_line)

    return ctx


class GraphRetriever:
    """
    Main retrieval class. Tries keyword-based neighbourhood search first;
    falls back to description-level fuzzy scan if nothing is returned.
    """

    def __init__(self, client: Neo4jClient | None = None) -> None:
        # Allow injection for testing
        self._client = client
        self._owns_client = client is None

    def retrieve(self, keywords: list[str], top_k: int | None = None) -> list[str]:
        """
        Parameters
        ----------
        keywords : entity keyword hints extracted by Agent B
        top_k    : max number of context strings to return

        Returns
        -------
        list of context strings (may be empty if graph has nothing relevant)
        """
        k = top_k or cfg.graph_top_k
        context: list[str] = []

        if self._owns_client:
            client = Neo4jClient()
            client.connect()
        else:
            client = self._client  # type: ignore[assignment]

        try:
            # --- Primary: neighbourhood query ---
            cypher, params = build_neighbourhood_query(keywords, limit=k * 2)
            records = client.query(cypher, params)
            logger.debug("Graph neighbourhood query returned %d records", len(records))
            context = _records_to_context(records)

            # --- Fallback: fuzzy description scan ---
            if len(context) < 2 and keywords:
                logger.debug("Neighbourhood sparse — trying fuzzy scan on '%s'", keywords[0])
                cypher2, params2 = build_fuzzy_query(keywords[0], limit=k)
                records2 = client.query(cypher2, params2)
                context.extend(_records_to_context(records2))

            # Deduplicate while preserving order
            seen: set[str] = set()
            unique: list[str] = []
            for item in context:
                if item not in seen:
                    seen.add(item)
                    unique.append(item)

            logger.info("Graph retriever returning %d context items", len(unique[:k]))
            return unique[:k]

        finally:
            if self._owns_client:
                client.close()

    def is_context_sufficient(self, context: list[str]) -> bool:
        """Heuristic: if we have fewer than 2 context items, treat as insufficient."""
        return len(context) >= 2
