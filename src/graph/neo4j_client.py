"""
neo4j_client.py
---------------
Thin wrapper around the official Neo4j Python driver.
Provides context-managed sessions and a helper to run Cypher queries.
"""

import logging
from typing import Any

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

from config import cfg

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Thread-safe Neo4j client.  Use as a context manager:

        with Neo4jClient() as client:
            results = client.query("MATCH (n) RETURN n LIMIT 5")
    """

    def __init__(self) -> None:
        self._driver: Driver | None = None

    def connect(self) -> None:
        try:
            self._driver = GraphDatabase.driver(
                cfg.neo4j_uri,
                auth=(cfg.neo4j_user, cfg.neo4j_password),
            )
            # Verify connectivity immediately so we fail fast
            self._driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", cfg.neo4j_uri)
        except ServiceUnavailable as exc:
            raise ConnectionError(
                f"Could not reach Neo4j at {cfg.neo4j_uri}. "
                "Is the database running?"
            ) from exc
        except AuthError as exc:
            raise PermissionError(
                "Neo4j authentication failed. Check NEO4J_USER / NEO4J_PASSWORD."
            ) from exc

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jClient":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return a list of record dicts."""
        if self._driver is None:
            raise RuntimeError("Client not connected. Use as a context manager.")

        db = database or cfg.neo4j_database
        with self._driver.session(database=db) as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]

    def execute_write(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> None:
        """Fire-and-forget write query (INSERT / MERGE / DELETE)."""
        if self._driver is None:
            raise RuntimeError("Client not connected.")

        db = database or cfg.neo4j_database
        with self._driver.session(database=db) as session:
            session.execute_write(lambda tx: tx.run(cypher, params or {}))

    def clear_graph(self) -> None:
        """Delete everything. Useful for testing — obviously don't call this in prod."""
        logger.warning("Clearing entire Neo4j graph!")
        self.execute_write("MATCH (n) DETACH DELETE n")

    def get_node_count(self) -> int:
        result = self.query("MATCH (n) RETURN count(n) AS cnt")
        return result[0]["cnt"] if result else 0

    def get_relationship_count(self) -> int:
        result = self.query("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return result[0]["cnt"] if result else 0
