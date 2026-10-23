"""FastAPI route handlers for ingestion, vector search, graph traversal, and hybrid QA."""

import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.config import settings
from src.embeddings.provider import embedding_provider
from src.graph.client import neo4j_client
from src.graph.queries import GET_GRAPH_STATS, GET_RELATIONSHIP_STATS
from src.ingestion.pipeline import ingestion_pipeline
from src.models.api import (
    GraphPathSegment,
    GraphQueryRequest,
    GraphStatsResponse,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from src.retrieval.graph_traversal import MultiHopGraphTraverser
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.qa_engine import WellboreQAEngine
from src.retrieval.vector_search import Neo4jVectorRetriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Wellbore GraphRAG"])

# Retrival and QA singletons
vector_retriever = Neo4jVectorRetriever(graph_client=neo4j_client, embedder=embedding_provider)
graph_traverser = MultiHopGraphTraverser(graph_client=neo4j_client)
hybrid_retriever = HybridRetriever(vector_retriever=vector_retriever, graph_traverser=graph_traverser, pipeline=ingestion_pipeline)
qa_engine = WellboreQAEngine(retriever=hybrid_retriever)


@router.get("/health", summary="Service health status")
async def health_check():
    """Verify health of API, Neo4j connection, and embedding provider."""
    neo4j_ok = neo4j_client.is_connected()
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "database": {
            "type": "Neo4j",
            "connected": neo4j_ok,
            "uri": settings.NEO4J_URI,
        },
        "embeddings": {
            "model": settings.OLLAMA_EMBED_MODEL,
            "dimension": settings.EMBEDDING_DIMENSION,
        },
        "llm": {
            "model": settings.OLLAMA_LLM_MODEL,
            "provider": "Ollama / Grounded Fallback",
        },
    }


@router.post("/query/hybrid", response_model=QueryResponse, summary="Execute Hybrid GraphRAG QA Query")
async def query_hybrid(request: QueryRequest):
    """
    Execute multi-hop graph traversal alongside vector search.
    Returns grounded engineering response with exact equipment, failure, SOP citations.
    """
    try:
        response = qa_engine.answer_query(request)
        return response
    except Exception as e:
        logger.error("Error executing hybrid query: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}",
        )


@router.post("/query/vector", response_model=List[RetrievedChunk], summary="Execute Vector Search over Chunks")
async def query_vector(request: QueryRequest):
    """Retrieve top-K matching document chunks via dense vector cosine similarity."""
    try:
        cached_chunks = list(ingestion_pipeline.in_memory_chunks.values())
        results = vector_retriever.search(
            query=request.question,
            top_k=request.top_k_chunks,
            min_score=request.min_similarity,
            fallback_chunks=cached_chunks,
        )
        return results
    except Exception as e:
        logger.error("Error in vector search: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/query/graph", response_model=List[GraphPathSegment], summary="Traverse Knowledge Graph Neighborhood")
async def query_graph(request: GraphQueryRequest):
    """
    Traverse graph paths from a specific entity or execute a safe read-only Cypher query.
    """
    try:
        if request.custom_cypher:
            records = graph_traverser.execute_safe_cypher(request.custom_cypher)
            # Format generic Cypher output
            segments = []
            for r in records:
                segments.append(
                    GraphPathSegment(
                        source_label=r.get("source_label", "Entity"),
                        source_id=str(r.get("source_id", "id")),
                        source_name=str(r.get("source_name", "name")),
                        relation=str(r.get("relation", "RELATED_TO")),
                        target_label=r.get("target_label", "Entity"),
                        target_id=str(r.get("target_id", "id")),
                        target_name=str(r.get("target_name", "name")),
                    )
                )
            return segments

        if not request.entity_type or not request.entity_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'custom_cypher' or both 'entity_type' and 'entity_id' must be specified.",
            )

        paths = graph_traverser.traverse_entity_neighborhood(
            entity_type=request.entity_type,
            entity_id=request.entity_id,
            max_hops=request.max_hops,
            rel_type=request.relationship_type,
        )
        return paths
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as e:
        logger.error("Error in graph query: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/ingest/text", response_model=IngestResponse, summary="Ingest Unstructured Engineering Text")
async def ingest_text(payload: IngestTextRequest):
    """
    Ingest raw engineering text (e.g. daily drilling logs, SOPs, incident reports).
    Parses entities, chunks text, generates embeddings, and saves into Neo4j.
    """
    try:
        response = ingestion_pipeline.ingest_raw_text(
            text=payload.text,
            title=payload.title,
            doc_id=payload.doc_id,
            source=payload.source,
            category=payload.category,
        )
        return response
    except Exception as e:
        logger.error("Error ingesting text: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/ingest/file", response_model=IngestResponse, summary="Ingest Engineering PDF or Text Document")
async def ingest_file(
    file: UploadFile = File(...),
    doc_id: str = Form(None),
    category: str = Form("engineering_manual"),
):
    """
    Upload and ingest technical PDF or text document into the Knowledge Graph.
    Extracts equipment serial numbers, formation lithologies, failure modes, and SOPs.
    """
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        response = ingestion_pipeline.ingest_file(
            file_path=tmp_path,
            doc_id=doc_id or os.path.splitext(file.filename)[0],
            category=category,
        )
        return response
    except Exception as e:
        logger.error("Error ingesting file %s: %s", file.filename, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"File ingestion failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.get("/graph/stats", response_model=GraphStatsResponse, summary="Get Knowledge Graph Statistics")
async def graph_stats():
    """Return count of entities, relationship types, and vector index status."""
    if not neo4j_client.is_connected():
        # Return statistics from local in-memory cache
        mem_chunks = len(ingestion_pipeline.in_memory_chunks)
        mem_docs = len(ingestion_pipeline.in_memory_docs)
        mem_rels = sum(len(ext.relationships) for ext in ingestion_pipeline.in_memory_extractions.values())
        return GraphStatsResponse(
            node_counts={"Document": mem_docs, "DocumentChunk": mem_chunks},
            relationship_counts={"EXTRACTED_FROM": mem_chunks, "MENTIONS": mem_rels},
            total_nodes=mem_docs + mem_chunks,
            total_relationships=mem_chunks + mem_rels,
            vector_index_status="in_memory_cache",
        )

    try:
        node_records = neo4j_client.execute_query(GET_GRAPH_STATS)
        rel_records = neo4j_client.execute_query(GET_RELATIONSHIP_STATS)

        node_counts = {}
        total_nodes = 0
        if node_records and "nodes" in node_records[0]:
            for item in node_records[0]["nodes"]:
                if item["label"]:
                    node_counts[item["label"]] = item["count"]
                    total_nodes += item["count"]

        rel_counts = {}
        total_rels = 0
        if rel_records and "relationships" in rel_records[0]:
            for item in rel_records[0]["relationships"]:
                if item["type"]:
                    rel_counts[item["type"]] = item["count"]
                    total_rels += item["count"]

        return GraphStatsResponse(
            node_counts=node_counts,
            relationship_counts=rel_counts,
            total_nodes=total_nodes,
            total_relationships=total_rels,
            vector_index_status="online",
        )
    except Exception as e:
        logger.error("Error reading graph stats: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
