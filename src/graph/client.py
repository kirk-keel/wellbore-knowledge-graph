"""Neo4j driver connection manager with resilient pooling and fallback."""

import logging
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.config import settings
from src.graph.schema import CONSTRAINT_STATEMENTS, INDEX_STATEMENTS, VECTOR_INDEX_STATEMENT

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Manages connections, schema initialization, and transactional Cypher execution."""

    def __init__(
        self,
        uri: str = settings.NEO4J_URI,
        user: str = settings.NEO4J_USER,
        password: str = settings.NEO4J_PASSWORD,
        database: str = settings.NEO4J_DATABASE,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver: Optional[Driver] = None
        self._connected: bool = False

    def connect(self) -> bool:
        """Establish connection to Neo4j cluster or instance."""
        if self._driver is not None and self._connected:
            return True

        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_pool_size=settings.NEO4J_MAX_CONNECTION_POOL_SIZE,
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            self._connected = True
            logger.info("Successfully connected to Neo4j at %s", self.uri)
            return True
        except (ServiceUnavailable, AuthError, Exception) as e:
            logger.warning("Could not establish live connection to Neo4j (%s): %s", self.uri, e)
            self._connected = False
            return False

    def close(self):
        """Close connection driver pool."""
        if self._driver:
            self._driver.close()
            self._driver = None
            self._connected = False
            logger.info("Neo4j driver connection closed.")

    def is_connected(self) -> bool:
        """Check if Neo4j is actively reachable."""
        if not self._driver:
            return self.connect()
        try:
            self._driver.verify_connectivity()
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    def init_schema(self) -> Dict[str, Any]:
        """Apply constraints, standard indexes, and vector indexes."""
        if not self.is_connected():
            logger.warning("Neo4j is not connected. Skipping schema setup.")
            return {"status": "skipped", "reason": "database_offline"}

        results = {"constraints": 0, "indexes": 0, "vector_index": "pending"}

        with self.get_session() as session:
            # 1. Apply constraints
            for stmt in CONSTRAINT_STATEMENTS:
                try:
                    session.run(stmt)
                    results["constraints"] += 1
                except Exception as e:
                    logger.debug("Constraint application notice: %s", e)

            # 2. Apply property indexes
            for stmt in INDEX_STATEMENTS:
                try:
                    session.run(stmt)
                    results["indexes"] += 1
                except Exception as e:
                    logger.debug("Index application notice: %s", e)

            # 3. Apply vector index
            try:
                session.run(VECTOR_INDEX_STATEMENT)
                results["vector_index"] = "active"
                logger.info("Neo4j Vector Index '%s' verified.", settings.VECTOR_INDEX_NAME)
            except Exception as e:
                logger.warning("Vector index creation note (requires Neo4j 5.11+): %s", e)
                results["vector_index"] = f"error: {str(e)}"

        return results

    def get_session(self) -> Session:
        """Create a new database session."""
        if not self._driver:
            if not self.connect():
                raise ConnectionError("Neo4j database is unreachable.")
        return self._driver.session(database=self.database)

    def execute_query(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a read/query Cypher statement and return results as dictionaries."""
        if not self.is_connected():
            return []

        with self.get_session() as session:
            result = session.run(cypher, parameters or {})
            return [record.data() for record in result]

    def execute_write(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a write/mutation Cypher statement."""
        if not self.is_connected():
            return []

        with self.get_session() as session:
            result = session.run(cypher, parameters or {})
            return [record.data() for record in result]


# Global client instance
neo4j_client = Neo4jClient()
