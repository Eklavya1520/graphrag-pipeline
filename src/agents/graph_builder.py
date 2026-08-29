"""
graph_builder.py  —  Agent A
-----------------------------
Converts raw text into a structured knowledge graph in Neo4j.

The agent:
  1. Splits input text into manageable chunks
  2. Calls the LLM with a structured-output prompt to extract entities + relations
  3. Writes the extracted triples into Neo4j using MERGE (idempotent)

Structured output schema
------------------------
The LLM is asked to return a JSON object matching ExtractionResult:
{
  "entities": [
    {"name": "Transformer", "type": "Concept", "description": "..."},
    ...
  ],
  "relations": [
    {"source": "Transformer", "relation": "USES", "target": "Attention Mechanism", "context": "..."},
    ...
  ]
}
"""

import json
import logging
import re
import textwrap
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from graph.neo4j_client import Neo4jClient
from graph.cypher_templates import MERGE_ENTITY, MERGE_RELATIONSHIP
from config import cfg

logger = logging.getLogger(__name__)

# ── Pydantic schemas for structured LLM output ───────────────────────────────

class EntitySchema(BaseModel):
    name: str = Field(description="Entity name, title-cased, max 5 words")
    type: str = Field(description="One of: Concept, Person, Technology, Organization, Method, Dataset, Other")
    description: str = Field(description="One-sentence description of this entity")


class RelationSchema(BaseModel):
    source: str = Field(description="Name of the source entity")
    relation: str = Field(description="Relation type in SCREAMING_SNAKE_CASE, e.g. USES, IS_PART_OF, DEVELOPED_BY")
    target: str = Field(description="Name of the target entity")
    context: str = Field(description="Short phrase explaining this specific relationship")


class ExtractionResult(BaseModel):
    entities: list[EntitySchema] = Field(default_factory=list)
    relations: list[RelationSchema] = Field(default_factory=list)


# ── Extraction prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a knowledge graph construction assistant.
    Given a passage of text, extract all named entities and the relationships between them.

    Rules:
    - Only extract entities explicitly mentioned in the text.
    - Entity names must be concise (≤5 words), title-cased.
    - Relation types use SCREAMING_SNAKE_CASE (e.g. DEVELOPED_BY, IS_PART_OF, USES).
    - Each relation must reference entities from the entities list.
    - Return valid JSON matching the schema exactly. No extra commentary.

    Schema:
    {
      "entities": [{"name": str, "type": str, "description": str}],
      "relations": [{"source": str, "relation": str, "target": str, "context": str}]
    }
""").strip()


def _get_llm():
    """Return a LangChain chat model based on config."""
    if cfg.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=cfg.llm_model,
            google_api_key=cfg.google_api_key,
            temperature=0.1,
        )
    elif cfg.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=cfg.llm_model,
            anthropic_api_key=cfg.anthropic_api_key,
            temperature=0.1,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=cfg.llm_model,
            openai_api_key=cfg.openai_api_key,
            temperature=0.1,
        )


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Simple sliding-window chunker.
    chunk_size and overlap are in characters (not tokens).
    A more production-ready approach would use a proper tokenizer.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at sentence boundary
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + overlap:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 50]  # skip tiny trailing chunks


def _parse_llm_output(raw: str) -> ExtractionResult:
    """
    Parse JSON from LLM output.  The model sometimes wraps JSON in markdown
    code fences, so we strip those first.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(cleaned)
        return ExtractionResult(**data)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse LLM extraction output: %s\nRaw: %s", e, raw[:300])
        return ExtractionResult()


# ── Main Agent A class ────────────────────────────────────────────────────────

class GraphBuilder:
    """
    Agent A — builds the knowledge graph from raw text.

    Example
    -------
    with Neo4jClient() as client:
        builder = GraphBuilder(client)
        stats = builder.ingest_text("Transformers were introduced by Vaswani et al. ...")
        print(stats)
    """

    def __init__(self, neo4j_client: Neo4jClient) -> None:
        self._client = neo4j_client
        self._llm = _get_llm()

    def ingest_text(self, text: str, source_label: str = "unknown") -> dict[str, int]:
        """
        Full ingestion pipeline for a single document string.
        Returns counts of entities/relations written.
        """
        chunks = _chunk_text(text)
        logger.info("Ingesting '%s' — %d chunks", source_label, len(chunks))

        total_entities = 0
        total_relations = 0

        for i, chunk in enumerate(chunks):
            logger.debug("Processing chunk %d/%d", i + 1, len(chunks))
            extraction = self._extract(chunk)
            e, r = self._write_to_graph(extraction)
            total_entities += e
            total_relations += r

        logger.info(
            "Finished '%s': %d entities, %d relations written",
            source_label, total_entities, total_relations,
        )
        return {"entities": total_entities, "relations": total_relations}

    def _extract(self, chunk: str) -> ExtractionResult:
        """Call the LLM to extract entities + relations from a text chunk."""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Text:\n\n{chunk}"),
        ]
        response = self._llm.invoke(messages)
        return _parse_llm_output(response.content)

    def _write_to_graph(self, result: ExtractionResult) -> tuple[int, int]:
        """Write extracted entities and relations into Neo4j."""
        entities_written = 0
        relations_written = 0

        # Write entities first
        for entity in result.entities:
            try:
                self._client.execute_write(
                    MERGE_ENTITY,
                    {
                        "name": entity.name,
                        "entity_type": entity.type,
                        "description": entity.description,
                    },
                )
                entities_written += 1
            except Exception as e:
                logger.warning("Failed to write entity '%s': %s", entity.name, e)

        # Write relationships (both entities must already exist)
        known_names = {e.name for e in result.entities}
        for rel in result.relations:
            if rel.source not in known_names or rel.target not in known_names:
                # Skip dangling references
                continue
            try:
                self._client.execute_write(
                    MERGE_RELATIONSHIP,
                    {
                        "source": rel.source,
                        "rel_type": rel.relation,
                        "target": rel.target,
                        "context": rel.context,
                    },
                )
                relations_written += 1
            except Exception as e:
                logger.warning(
                    "Failed to write relation %s -[%s]-> %s: %s",
                    rel.source, rel.relation, rel.target, e,
                )

        return entities_written, relations_written
