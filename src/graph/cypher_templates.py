"""
cypher_templates.py
-------------------
Reusable Cypher query templates for the graph retriever.
Keeping them centralised makes it easier to tune retrieval logic
without touching the retriever itself.
"""

from string import Template


# ── Node upsert (used during ingestion) ──────────────────────────────────────

MERGE_ENTITY = """
MERGE (n:Entity {name: $name})
ON CREATE SET
    n.type        = $entity_type,
    n.description = $description,
    n.created_at  = datetime()
ON MATCH SET
    n.description = CASE
        WHEN n.description IS NULL OR n.description = ''
        THEN $description
        ELSE n.description
    END
"""

MERGE_RELATIONSHIP = """
MATCH (a:Entity {name: $source})
MATCH (b:Entity {name: $target})
MERGE (a)-[r:RELATION {type: $rel_type}]->(b)
ON CREATE SET
    r.context    = $context,
    r.created_at = datetime()
"""

# ── Retrieval queries ─────────────────────────────────────────────────────────

# 1-hop neighbourhood around matched entities
NEIGHBOURHOOD_QUERY = """
MATCH (n:Entity)
WHERE any(kw IN $keywords WHERE toLower(n.name) CONTAINS toLower(kw))
WITH n LIMIT $limit
MATCH (n)-[r:RELATION]-(neighbour:Entity)
RETURN
    n.name          AS entity,
    n.type          AS entity_type,
    n.description   AS entity_desc,
    r.type          AS relation,
    r.context       AS rel_context,
    neighbour.name  AS neighbour,
    neighbour.type  AS neighbour_type
ORDER BY entity
"""

# Direct entity lookup (exact match)
EXACT_ENTITY_QUERY = """
MATCH (n:Entity {name: $name})
OPTIONAL MATCH (n)-[r:RELATION]-(m:Entity)
RETURN
    n.name         AS entity,
    n.type         AS entity_type,
    n.description  AS entity_desc,
    r.type         AS relation,
    m.name         AS neighbour
"""

# 2-hop path between two entities (used for relational questions)
TWO_HOP_PATH_QUERY = """
MATCH path = (a:Entity)-[:RELATION*1..2]-(b:Entity)
WHERE toLower(a.name) CONTAINS toLower($entity_a)
  AND toLower(b.name) CONTAINS toLower($entity_b)
RETURN [node in nodes(path) | node.name] AS path_nodes,
       [rel  in rels(path)  | rel.type]  AS path_rels
LIMIT $limit
"""

# Full-text style scan (fallback when keyword matching fails)
FUZZY_SCAN_QUERY = """
MATCH (n:Entity)-[r:RELATION]->(m:Entity)
WHERE toLower(n.description) CONTAINS toLower($keyword)
   OR toLower(r.context)     CONTAINS toLower($keyword)
RETURN
    n.name  AS entity,
    r.type  AS relation,
    m.name  AS neighbour,
    r.context AS context
LIMIT $limit
"""

# Schema introspection (debugging / visualisation)
SCHEMA_QUERY = """
CALL apoc.meta.stats()
YIELD labels, relTypesCount
RETURN labels, relTypesCount
"""


def build_neighbourhood_query(keywords: list[str], limit: int = 10) -> tuple[str, dict]:
    return NEIGHBOURHOOD_QUERY, {"keywords": keywords, "limit": limit}


def build_fuzzy_query(keyword: str, limit: int = 10) -> tuple[str, dict]:
    return FUZZY_SCAN_QUERY, {"keyword": keyword, "limit": limit}


def build_two_hop_query(entity_a: str, entity_b: str, limit: int = 5) -> tuple[str, dict]:
    return TWO_HOP_PATH_QUERY, {"entity_a": entity_a, "entity_b": entity_b, "limit": limit}
