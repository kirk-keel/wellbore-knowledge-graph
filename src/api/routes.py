"""FastAPI route handlers for ingestion, vector search, graph traversal, and hybrid QA."""

import logging
import os
import json
import shutil
import tempfile
import asyncio
from typing import Any, Dict, List
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status, Query
from fastapi.responses import HTMLResponse, StreamingResponse

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
from src.retrieval.text_to_cypher import TextToCypherAgent
from src.retrieval.community_summary import CommunityDetector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Wellbore GraphRAG"])

# Retrival and QA singletons
vector_retriever = Neo4jVectorRetriever(graph_client=neo4j_client, embedder=embedding_provider)
graph_traverser = MultiHopGraphTraverser(graph_client=neo4j_client)
hybrid_retriever = HybridRetriever(vector_retriever=vector_retriever, graph_traverser=graph_traverser, pipeline=ingestion_pipeline)
qa_engine = WellboreQAEngine(retriever=hybrid_retriever)
text_to_cypher_agent = TextToCypherAgent(graph_client=neo4j_client, pipeline=ingestion_pipeline)
community_detector = CommunityDetector(graph_client=neo4j_client, pipeline=ingestion_pipeline)


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


@router.post("/query/cypher", summary="Execute Analytical Text-to-Cypher Query")
async def query_text_to_cypher(payload: Dict[str, str]):
    """
    Translates an analytical natural language engineering inquiry into safe Cypher
    and returns executed results (e.g. cumulative NPT, failure frequencies).
    """
    question = payload.get("question")
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Field 'question' is required.")
    try:
        result = text_to_cypher_agent.query(question)
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as e:
        logger.error("Error executing Text-to-Cypher: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/graph/communities", summary="List Global Graph Risk Communities")
async def list_communities():
    """Returns detected operational risk communities (stratigraphic & well integrity clusters)."""
    try:
        return community_detector.detect_communities()
    except Exception as e:
        logger.error("Error detecting communities: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/graph/communities/{community_id}/summary", summary="Get Community Risk Summary")
async def get_community_summary(community_id: str):
    """Returns executive risk synthesis for a specific formation/risk community."""
    try:
        return community_detector.generate_community_summary(community_id)
    except Exception as e:
        logger.error("Error generating community summary: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/query/stream", summary="Stream GraphRAG Response via SSE")
async def stream_query(question: str = Query(..., description="Engineering query to stream")):
    """
    Server-Sent Events (SSE) streaming endpoint delivering token-by-token synthesis
    alongside real-time graph traversal discovery events.
    """
    async def event_generator():
        yield f"event: start\ndata: {json.dumps({'status': 'retrieving', 'question': question})}\n\n"
        await asyncio.sleep(0.05)

        # 1. Retrieve
        req = QueryRequest(question=question, top_k_chunks=3, graph_hops=2)
        chunks, paths, citations = hybrid_retriever.retrieve(question, top_k=3, graph_hops=2)

        # Emit graph path discovery
        yield f"event: graph_paths\ndata: {json.dumps({'paths_discovered': len(paths), 'citations': [c.model_dump() for c in citations]})}\n\n"
        await asyncio.sleep(0.05)

        # 2. QA generation
        resp = qa_engine.answer_query(req)

        # Stream answer in chunks
        words = resp.answer.split(" ")
        for i in range(0, len(words), 4):
            token_slice = " ".join(words[i:i+4]) + " "
            yield f"event: token\ndata: {json.dumps({'token': token_slice})}\n\n"
            await asyncio.sleep(0.03)

        # Emit completion with metadata
        yield f"event: complete\ndata: {json.dumps({'confidence_score': resp.confidence_score, 'sources': resp.sources_consulted})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/graph/visual", summary="Cytoscape Graph Visualization JSON")
async def get_visual_graph():
    """Generates cytoscape-formatted nodes and edges for the web visualizer."""
    nodes = []
    edges = []
    seen_nodes = set()

    for ext in ingestion_pipeline.in_memory_extractions.values():
        for w in ext.wells:
            if w.name not in seen_nodes:
                nodes.append({"data": {"id": w.name, "label": w.name, "type": "Well", "color": "#8b5cf6"}})
                seen_nodes.add(w.name)
        for e in ext.equipment:
            if e.serial_number not in seen_nodes:
                nodes.append({"data": {"id": e.serial_number, "label": f"{e.equipment_type}\n({e.serial_number})", "type": "Equipment", "color": "#f59e0b"}})
                seen_nodes.add(e.serial_number)
        for f in ext.formations:
            if f.name not in seen_nodes:
                nodes.append({"data": {"id": f.name, "label": f"{f.name}\n({f.lithology})", "type": "Formation", "color": "#3b82f6"}})
                seen_nodes.add(f.name)
        for fm in ext.failure_modes:
            if fm.code not in seen_nodes:
                nodes.append({"data": {"id": fm.code, "label": f"{fm.failure_type}\n[{fm.code}]", "type": "FailureMode", "color": "#ef4444"}})
                seen_nodes.add(fm.code)
        for s in ext.sops:
            if s.sop_id not in seen_nodes:
                nodes.append({"data": {"id": s.sop_id, "label": f"SOP\n{s.sop_id}", "type": "SOP", "color": "#10b981"}})
                seen_nodes.add(s.sop_id)
        for inc in ext.incidents:
            if inc.incident_id not in seen_nodes:
                nodes.append({"data": {"id": inc.incident_id, "label": f"Incident\n{inc.incident_id}", "type": "Incident", "color": "#f97316"}})
                seen_nodes.add(inc.incident_id)

        for idx, rel in enumerate(ext.relationships):
            edges.append({
                "data": {
                    "id": f"e_{idx}_{rel.relation_type}",
                    "source": rel.source_id,
                    "target": rel.target_id,
                    "label": rel.relation_type,
                }
            })

    return {"elements": {"nodes": nodes, "edges": edges}}


@router.get("/dashboard", response_class=HTMLResponse, summary="Interactive Graph Dashboard")
async def get_dashboard():
    """Serves the interactive Subgraph Visualizer & GraphRAG web console."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Wellbore Knowledge Graph Explorer</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: #0f172a; color: #f8fafc; display: flex; flex-direction: column; height: 100vh; }
    header { background: #1e293b; padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #334155; }
    header h1 { font-size: 1.15rem; font-weight: 600; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
    .badge { background: #0284c7; color: white; padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; }
    .main-container { display: flex; flex: 1; overflow: hidden; }
    #cy { flex: 1; height: 100%; background: #0b1120; }
    .sidebar { width: 380px; background: #1e293b; border-left: 1px solid #334155; display: flex; flex-direction: column; }
    .panel { padding: 18px; border-bottom: 1px solid #334155; }
    .panel h2 { font-size: 0.95rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px; }
    .legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.8rem; }
    .dot { width: 10px; height: 10px; border-radius: 50%; }
    .query-box textarea { width: 100%; background: #0f172a; border: 1px solid #334155; color: white; padding: 10px; border-radius: 6px; resize: vertical; min-height: 70px; }
    .query-box button { margin-top: 8px; width: 100%; background: #0284c7; color: white; border: none; padding: 10px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: 0.2s; }
    .query-box button:hover { background: #0369a1; }
    .results-area { flex: 1; overflow-y: auto; padding: 18px; font-size: 0.85rem; line-height: 1.5; color: #cbd5e1; }
    .node-card { background: #0f172a; padding: 12px; border-radius: 6px; border: 1px solid #334155; margin-bottom: 12px; }
    .node-card h3 { color: #38bdf8; font-size: 0.9rem; margin-bottom: 4px; }
  </style>
</head>
<body>
  <header>
    <h1>🛢️ Wellbore Knowledge Graph <span class="badge">GraphRAG v0.1.0</span></h1>
    <div style="font-size: 0.85rem; color: #94a3b8;">Status: <span style="color: #4ade80;">Operational</span></div>
  </header>
  <div class="main-container">
    <div id="cy"></div>
    <div class="sidebar">
      <div class="panel">
        <h2>Ontology Legend</h2>
        <div class="legend">
          <div class="legend-item"><div class="dot" style="background:#8b5cf6;"></div>Well</div>
          <div class="legend-item"><div class="dot" style="background:#f59e0b;"></div>Equipment</div>
          <div class="legend-item"><div class="dot" style="background:#3b82f6;"></div>Formation</div>
          <div class="legend-item"><div class="dot" style="background:#ef4444;"></div>FailureMode</div>
          <div class="legend-item"><div class="dot" style="background:#10b981;"></div>SOP</div>
          <div class="legend-item"><div class="dot" style="background:#f97316;"></div>Incident</div>
        </div>
      </div>
      <div class="panel query-box">
        <h2>Ask Engineering Query</h2>
        <textarea id="queryInput" placeholder="e.g. What SOPs mitigate drillstring pack-off and BOP ram leaks?"></textarea>
        <button onclick="runQuery()">Run GraphRAG Query</button>
      </div>
      <div class="results-area" id="resultsArea">
        <div style="color: #64748b; text-align: center; margin-top: 40px;">
          Click any graph node to inspect properties or submit an engineering query.
        </div>
      </div>
    </div>
  </div>
  <script>
    async function initGraph() {
      try {
        const res = await fetch('/api/v1/graph/visual');
        const data = await res.json();
        const cy = cytoscape({
          container: document.getElementById('cy'),
          elements: data.elements,
          style: [
            {
              selector: 'node',
              style: {
                'background-color': 'data(color)',
                'label': 'data(label)',
                'color': '#ffffff',
                'font-size': '10px',
                'text-valign': 'center',
                'text-halign': 'center',
                'text-wrap': 'wrap',
                'width': '65px',
                'height': '65px',
                'border-width': 2,
                'border-color': '#ffffff',
                'text-outline-color': '#0f172a',
                'text-outline-width': 2
              }
            },
            {
              selector: 'edge',
              style: {
                'width': 2,
                'line-color': '#475569',
                'target-arrow-color': '#94a3b8',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'label': 'data(label)',
                'font-size': '8px',
                'color': '#94a3b8',
                'text-rotation': 'autorotate',
                'text-background-opacity': 0.8,
                'text-background-color': '#0f172a',
                'text-background-padding': '2px'
              }
            }
          ],
          layout: {
            name: 'cose',
            animate: false,
            nodeOverlap: 20,
            componentSpacing: 100
          }
        });

        cy.on('tap', 'node', function(evt){
          const node = evt.target;
          const d = node.data();
          document.getElementById('resultsArea').innerHTML = `
            <div class="node-card">
              <h3>[${d.type}] ${d.id}</h3>
              <p style="margin-top: 6px; font-size: 0.85rem; color: #94a3b8;">${d.label.replace('\\n', ' - ')}</p>
              <div style="margin-top: 10px; font-size: 0.75rem; color: #38bdf8;">Connected Edges: ${node.connectedEdges().length}</div>
            </div>
          `;
        });
      } catch(err) {
        console.error("Failed to load visual graph:", err);
      }
    }

    async function runQuery() {
      const q = document.getElementById('queryInput').value;
      if (!q) return;
      const resArea = document.getElementById('resultsArea');
      resArea.innerHTML = '<div style="color: #38bdf8;">Retrieving vector embeddings & traversing graph...</div>';
      try {
        const resp = await fetch('/api/v1/query/hybrid', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({question: q, top_k_chunks: 3, graph_hops: 2})
        });
        const data = await resp.json();
        resArea.innerHTML = `
          <div class="node-card">
            <h3 style="color:#4ade80;">Confidence Score: ${(data.confidence_score * 100).toFixed(1)}%</h3>
            <div style="margin-top: 8px; white-space: pre-wrap;">${data.answer}</div>
            <div style="margin-top: 12px; font-size: 0.75rem; color: #94a3b8;">Sources: ${data.sources_consulted.join(', ')}</div>
          </div>
        `;
      } catch(e) {
        resArea.innerHTML = '<div style="color:#ef4444;">Query failed: ' + e + '</div>';
      }
    }

    window.onload = initGraph;
  </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)
