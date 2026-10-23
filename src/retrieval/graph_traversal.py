"""Multi-hop graph traversal and semantic path expansion engine."""

import logging
import re
from typing import Any, Dict, List, Optional

from src.config import settings
from src.graph.client import Neo4jClient, neo4j_client
from src.graph.queries import EXPAND_CHUNK_NEIGHBORHOOD, TRAVERSE_FAILURE_MITIGATION
from src.models.api import GraphPathSegment
from src.models.domain import Relationship

logger = logging.getLogger(__name__)


class MultiHopGraphTraverser:
    """Discovers semantic relationship paths connecting assets, failures, and SOPs."""

    def __init__(self, graph_client: Neo4jClient = neo4j_client):
        self.graph = graph_client

    def expand_from_chunks(
        self,
        chunk_ids: List[str],
        max_hops: int = settings.DEFAULT_GRAPH_HOPS,
        max_paths: int = settings.MAX_GRAPH_PATHS,
        fallback_relations: Optional[List[Relationship]] = None,
    ) -> List[GraphPathSegment]:
        """Expand graph neighborhood from seed chunk IDs."""
        if not chunk_ids:
            return []

        # 1. Query live Neo4j
        if self.graph.is_connected():
            try:
                results = self.graph.execute_query(
                    EXPAND_CHUNK_NEIGHBORHOOD,
                    {"chunk_ids": chunk_ids, "max_paths": max_paths},
                )
                if results:
                    return [
                        GraphPathSegment(
                            source_label=r["source_label"],
                            source_id=str(r["source_id"]),
                            source_name=str(r["source_name"]),
                            relation=str(r["relation"]),
                            target_label=r["target_label"],
                            target_id=str(r["target_id"]),
                            target_name=str(r["target_name"]),
                            properties=r.get("properties") or {},
                        )
                        for r in results
                    ]
            except Exception as e:
                logger.warning("Neo4j graph neighborhood expansion error: %s. Using fallback.", e)

        # 2. Fallback to in-memory traversal
        if fallback_relations:
            return self._traverse_in_memory(fallback_relations, max_paths=max_paths)

        return []

    def traverse_entity_neighborhood(
        self,
        entity_type: str,
        entity_id: str,
        max_hops: int = 2,
        rel_type: Optional[str] = None,
    ) -> List[GraphPathSegment]:
        """Traverse graph starting from a specific domain entity."""
        if not self.graph.is_connected():
            return []

        rel_filter = f":{rel_type}" if rel_type else ""
        query = f"""
        MATCH (s:{entity_type})
        WHERE (s.name = $entity_id OR s.serial_number = $entity_id OR s.code = $entity_id OR s.sop_id = $entity_id OR s.incident_id = $entity_id)
        MATCH path = (s)-[r{rel_filter}*1..{max_hops}]-(t)
        WHERE NOT t:Document AND NOT t:DocumentChunk
        UNWIND relationships(path) as rel
        WITH DISTINCT 
            startNode(rel) AS sn, 
            type(rel) AS r_type, 
            endNode(rel) AS tn,
            properties(rel) AS r_props
        RETURN 
            labels(sn)[0] AS source_label,
            coalesce(sn.name, sn.serial_number, sn.code, sn.sop_id, sn.incident_id, 'unknown') AS source_id,
            coalesce(sn.title, sn.name, sn.serial_number, sn.code, sn.sop_id, sn.incident_id, 'unknown') AS source_name,
            r_type AS relation,
            labels(tn)[0] AS target_label,
            coalesce(tn.name, tn.serial_number, tn.code, tn.sop_id, tn.incident_id, 'unknown') AS target_id,
            coalesce(tn.title, tn.name, tn.serial_number, tn.code, tn.sop_id, tn.incident_id, 'unknown') AS target_name,
            r_props AS properties
        LIMIT 30
        """
        try:
            records = self.graph.execute_query(query, {"entity_id": entity_id})
            return [
                GraphPathSegment(
                    source_label=r["source_label"],
                    source_id=str(r["source_id"]),
                    source_name=str(r["source_name"]),
                    relation=str(r["relation"]),
                    target_label=r["target_label"],
                    target_id=str(r["target_id"]),
                    target_name=str(r["target_name"]),
                    properties=r.get("properties") or {},
                )
                for r in records
            ]
        except Exception as e:
            logger.error("Error in traverse_entity_neighborhood: %s", e)
            return []

    def execute_safe_cypher(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Validate read-only query and execute against Neo4j."""
        # Security Guardrail: Reject any mutation statements
        forbidden = ["CREATE", "MERGE", "SET", "DELETE", "REMOVE", "DROP", "DETACH"]
        upper = cypher.upper()
        for kw in forbidden:
            if re.search(rf"\b{kw}\b", upper):
                raise ValueError(f"Query contains disallowed mutation keyword: {kw}. Read-only queries permitted.")

        return self.graph.execute_query(cypher, params or {})

    def _traverse_in_memory(self, relations: List[Relationship], max_paths: int = 25) -> List[GraphPathSegment]:
        """Extract path segments from in-memory relationship list."""
        segments: List[GraphPathSegment] = []
        for r in relations[:max_paths]:
            segments.append(
                GraphPathSegment(
                    source_label=r.source_type,
                    source_id=r.source_id,
                    source_name=r.source_id,
                    relation=r.relation_type,
                    target_label=r.target_type,
                    target_id=r.target_id,
                    target_name=r.target_id,
                    properties=r.properties,
                )
            )
        return segments
