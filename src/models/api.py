"""API schema models for FastAPI requests and responses."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.models.domain import WellboreExtractionResult


class QueryRequest(BaseModel):
    """Request payload for hybrid GraphRAG queries."""
    question: str = Field(..., description="Engineering query, e.g. 'What SOPs mitigate drillstring pack-off in shale formations?'")
    top_k_chunks: int = Field(default=5, ge=1, le=20, description="Number of vector chunks to retrieve")
    graph_hops: int = Field(default=2, ge=1, le=4, description="Depth of graph expansion from seed entities")
    min_similarity: float = Field(default=0.10, ge=0.0, le=1.0, description="Minimum cosine similarity cutoff")
    include_graph_paths: bool = Field(default=True, description="Whether to include structured graph traversal paths in response")


class GraphQueryRequest(BaseModel):
    """Direct graph traversal inquiry by entity or custom Cypher query."""
    entity_type: Optional[str] = Field(None, description="Entity type: 'Well', 'Equipment', 'Formation', 'FailureMode', 'SOP', 'Incident'")
    entity_id: Optional[str] = Field(None, description="Primary ID or key, e.g. 'BOP-HYD-5000-08' or 'Eagle Ford Shale'")
    relationship_type: Optional[str] = Field(None, description="Relationship filter, e.g. 'EXHIBITED_FAILURE' or 'MITIGATED_BY'")
    max_hops: int = Field(default=2, ge=1, le=4, description="Maximum traversal depth")
    custom_cypher: Optional[str] = Field(None, description="Optional read-only Cypher query override")


class IngestTextRequest(BaseModel):
    """Payload to ingest raw engineering procedures or incident memos."""
    title: str = Field(..., description="Document title")
    text: str = Field(..., description="Unstructured engineering text")
    doc_id: Optional[str] = Field(None, description="Optional custom document ID")
    source: str = Field(default="manual_entry", description="Source or reference of document")
    category: str = Field(default="engineering_report", description="Document category, e.g. 'IADC_standard', 'well_program', 'incident_report'")


class RetrievedChunk(BaseModel):
    """Vector retrieved chunk with relevance scoring."""
    chunk_id: str
    doc_id: str
    page: Optional[int] = None
    text: str
    similarity_score: float


class GraphPathSegment(BaseModel):
    """A directed edge in the knowledge graph traversal."""
    source_label: str
    source_id: str
    source_name: str
    relation: str
    target_label: str
    target_id: str
    target_name: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class EntityCitation(BaseModel):
    """Citation of an engineering entity discovered in the graph."""
    entity_type: str
    entity_id: str
    entity_name: str
    relation_to_context: Optional[str] = None


class QueryResponse(BaseModel):
    """Grounded hybrid GraphRAG response."""
    question: str
    answer: str
    confidence_score: float
    retrieved_chunks: List[RetrievedChunk]
    graph_paths: List[GraphPathSegment]
    entities_cited: List[EntityCitation]
    reasoning_summary: str
    sources_consulted: List[str]


class IngestResponse(BaseModel):
    """Result of document ingestion and graph population."""
    doc_id: str
    status: str
    chunks_count: int
    entities_extracted: Dict[str, int]
    relationships_created: int
    extraction_details: Optional[WellboreExtractionResult] = None
    message: str


class GraphStatsResponse(BaseModel):
    """Overview statistics of the Neo4j Knowledge Graph."""
    node_counts: Dict[str, int]
    relationship_counts: Dict[str, int]
    total_nodes: int
    total_relationships: int
    vector_index_status: str
