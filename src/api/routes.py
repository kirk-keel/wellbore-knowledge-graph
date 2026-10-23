"""FastAPI route handlers for ingestion, vector search, graph traversal, and hybrid QA."""

import logging
import os
import json
import shutil
import tempfile
import asyncio
from typing import Any, Dict, List
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status, Query, Response
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
from src.retrieval.casing_calculator import CasingIntegrityCalculator, CasingIntegrityRequest, CasingIntegrityReport
from src.retrieval.rcfa_generator import RCFAGenerator, RCFARequest, RCFAReport
from src.ingestion.telemetry_stream import TelemetryWatchdog, TelemetryFrame, PrecursorAlert
from src.retrieval.multi_agent import MultiAgentDrillingPanel, MultiAgentAdvisoryReport
from src.retrieval.wellbore_schematic import WellboreSchematicGenerator, WellboreSchematicData
from src.retrieval.hydraulics_engine import HydraulicsEngine, HydraulicsRequest, HydraulicsReport
from src.retrieval.anti_collision import AntiCollisionScanner, AntiCollisionReport, SurveyStation
from src.retrieval.vibration_classifier import VibrationClassifier, VibrationDataPacket, VibrationClassificationReport
from src.ingestion.witsml_exporter import WITSML141Exporter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Wellbore GraphRAG"])

# Retrival and QA singletons
vector_retriever = Neo4jVectorRetriever(graph_client=neo4j_client, embedder=embedding_provider)
graph_traverser = MultiHopGraphTraverser(graph_client=neo4j_client)
hybrid_retriever = HybridRetriever(vector_retriever=vector_retriever, graph_traverser=graph_traverser, pipeline=ingestion_pipeline)
qa_engine = WellboreQAEngine(retriever=hybrid_retriever)
text_to_cypher_agent = TextToCypherAgent(graph_client=neo4j_client, pipeline=ingestion_pipeline)
community_detector = CommunityDetector(graph_client=neo4j_client, pipeline=ingestion_pipeline)
casing_calculator = CasingIntegrityCalculator()
rcfa_generator = RCFAGenerator(graph_client=neo4j_client, pipeline=ingestion_pipeline)
telemetry_watchdog = TelemetryWatchdog(pipeline=ingestion_pipeline)
multi_agent_panel = MultiAgentDrillingPanel(qa_engine=qa_engine, retriever=hybrid_retriever)
schematic_generator = WellboreSchematicGenerator()
hydraulics_engine = HydraulicsEngine()
anti_collision_scanner = AntiCollisionScanner()
vibration_classifier = VibrationClassifier()
witsml_exporter = WITSML141Exporter()


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


@router.post("/engineering/casing-check", response_model=CasingIntegrityReport, summary="Evaluate Casing Burst, Collapse & MASP")
async def evaluate_casing(payload: CasingIntegrityRequest):
    """Calculates downhole geomechanical loads, MASP, and verifies casing structural margins."""
    try:
        return casing_calculator.evaluate_casing_integrity(payload)
    except Exception as e:
        logger.error("Error evaluating casing integrity: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/reports/rcfa", response_model=RCFAReport, summary="Generate 5-Whys RCFA Post-Mortem Report")
async def generate_rcfa_report(payload: RCFARequest):
    """Constructs 5-Whys causal hierarchy and generates formal IADC post-mortem report."""
    try:
        return rcfa_generator.generate_rcfa(payload)
    except Exception as e:
        logger.error("Error generating RCFA: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/telemetry/stream", summary="Ingest Real-Time Drilling Telemetry")
async def ingest_telemetry(payload: Dict[str, Any]):
    """
    Ingests real-time drilling telemetry frame(s) and checks for active failure precursors
    against the Knowledge Graph (pack-off, kick, drilling break).
    """
    try:
        if "frames" in payload:
            frames = [TelemetryFrame(**f) for f in payload["frames"]]
            alerts = telemetry_watchdog.ingest_batch(frames)
            return {"status": "INGESTED", "frames_count": len(frames), "alerts_triggered": alerts}
        else:
            frame = TelemetryFrame(**payload)
            alert = telemetry_watchdog.ingest_frame(frame)
            return {
                "status": "ALERT_TRIGGERED" if alert else "NORMAL",
                "alert": alert,
                "frame_logged": frame,
            }
    except Exception as e:
        logger.error("Error ingesting telemetry: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/telemetry/alerts", response_model=List[PrecursorAlert], summary="List Active Precursor Alerts")
async def list_telemetry_alerts():
    """Returns active real-time precursor alerts."""
    return telemetry_watchdog.active_alerts


@router.post("/query/multi-agent", response_model=MultiAgentAdvisoryReport, summary="Multi-Agent Specialist Panel Consultation")
async def query_multi_agent(payload: Dict[str, str]):
    """
    Executes collaborative multi-agent inquiry across 4 drilling specialist roles:
    Superintendent, Geomechanics, Equipment Reliability, and Statutory Well Control Auditor.
    """
    inquiry = payload.get("inquiry")
    if not inquiry:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Field 'inquiry' is required.")
    well_name = payload.get("well_name", "Well 104-B")
    try:
        return multi_agent_panel.consult(inquiry=inquiry, well_name=well_name)
    except Exception as e:
        logger.error("Error in multi-agent consultation: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/wellbore/{well_name}/schematic", response_model=WellboreSchematicData, summary="Get Wellbore Schematic Data & SVG")
async def get_wellbore_schematic(well_name: str):
    """Returns 2D wellbore casing architecture and SVG rendering."""
    try:
        return schematic_generator.generate_schematic(well_name=well_name)
    except Exception as e:
        logger.error("Error generating wellbore schematic: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/wellbore/{well_name}/schematic.svg", summary="Direct SVG Rendering of Wellbore Schematic")
async def get_wellbore_schematic_svg(well_name: str):
    """Directly serves Scalable Vector Graphics (.svg) schematic."""
    try:
        data = schematic_generator.generate_schematic(well_name=well_name)
        return Response(content=data.svg_schematic, media_type="image/svg+xml")
    except Exception as e:
        logger.error("Error generating SVG schematic: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/engineering/hydraulics-ecd", response_model=HydraulicsReport, summary="Calculate Annular Hydraulics & Surge/Swab")
async def evaluate_hydraulics(payload: HydraulicsRequest):
    """Calculates Bingham Plastic annular friction, ECD, and surge/swab pressure margins."""
    try:
        return hydraulics_engine.simulate(payload)
    except Exception as e:
        logger.error("Error evaluating hydraulics: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/engineering/anti-collision", response_model=AntiCollisionReport, summary="Scan 3D Trajectory Anti-Collision")
async def scan_anti_collision(payload: Dict[str, Any]):
    """
    Computes 3D Minimum Curvature path and evaluates ISCWSA Separation Factor
    against offset wells in the knowledge graph.
    """
    active_well = payload.get("active_well_name", "Well 104-B")
    stations_raw = payload.get("stations", [])
    stations = [SurveyStation(**s) for s in stations_raw] if stations_raw else [
        SurveyStation(measured_depth_md_ft=0.0, inclination_deg=0.0, azimuth_deg=45.0),
        SurveyStation(measured_depth_md_ft=3800.0, inclination_deg=2.0, azimuth_deg=45.0),
        SurveyStation(measured_depth_md_ft=8200.0, inclination_deg=18.0, azimuth_deg=50.0),
        SurveyStation(measured_depth_md_ft=12500.0, inclination_deg=28.5, azimuth_deg=52.0),
    ]
    try:
        return anti_collision_scanner.scan_anti_collision(active_well=active_well, active_stations=stations)
    except Exception as e:
        logger.error("Error in anti-collision scan: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/engineering/vibration-classify", response_model=VibrationClassificationReport, summary="Classify Drillstring Dynamic Dysfunctions")
async def classify_vibrations(payload: VibrationDataPacket):
    """
    Classifies downhole drilling dysfunctions: Stick-Slip (SSSI %), Bit Bounce,
    and Backward Whirl, mapping threats to BHA components in the Knowledge Graph.
    """
    try:
        return vibration_classifier.classify(payload)
    except Exception as e:
        logger.error("Error classifying vibrations: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/export/witsml/{well_name}", summary="Export Trajectory to Energistics WITSML 1.4.1.1 XML")
async def export_witsml_trajectory(well_name: str, wellbore: str = "Wellbore-1"):
    """Generates official WITSML 1.4.1.1 XML document for commercial rig data aggregators."""
    try:
        xml_content = witsml_exporter.export_trajectory_witsml(
            well_name=well_name,
            wellbore_name=wellbore,
            survey_name=f"{well_name}_Definitive_MWD",
        )
        return Response(content=xml_content, media_type="application/xml")
    except Exception as e:
        logger.error("Error exporting WITSML: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/dashboard", response_class=HTMLResponse, summary="Interactive Graph Dashboard")
async def get_dashboard():
    """Serves the interactive Subgraph Visualizer & GraphRAG web console."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Wellbore Knowledge Graph Explorer & Engineering Console</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: #0f172a; color: #f8fafc; display: flex; flex-direction: column; height: 100vh; }
    header { background: #1e293b; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #334155; }
    header h1 { font-size: 1.15rem; font-weight: 600; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
    .badge { background: #0284c7; color: white; padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; }
    .nav-tabs { display: flex; gap: 8px; }
    .tab-btn { background: #334155; border: none; color: #94a3b8; padding: 6px 14px; border-radius: 6px; font-size: 0.82rem; font-weight: 500; cursor: pointer; transition: 0.2s; }
    .tab-btn.active { background: #0284c7; color: white; }
    .main-container { display: flex; flex: 1; overflow: hidden; position: relative; }
    .view-content { flex: 1; height: 100%; display: none; overflow-y: auto; background: #0b1120; }
    .view-content.active { display: flex; }
    #cy { flex: 1; height: 100%; }
    .sidebar { width: 400px; background: #1e293b; border-left: 1px solid #334155; display: flex; flex-direction: column; }
    .panel { padding: 16px; border-bottom: 1px solid #334155; }
    .panel h2 { font-size: 0.9rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px; }
    .legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.78rem; }
    .dot { width: 10px; height: 10px; border-radius: 50%; }
    .query-box textarea { width: 100%; background: #0f172a; border: 1px solid #334155; color: white; padding: 10px; border-radius: 6px; resize: vertical; min-height: 70px; font-size: 0.85rem; }
    .query-box button { margin-top: 8px; width: 100%; background: #0284c7; color: white; border: none; padding: 9px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: 0.2s; font-size: 0.85rem; }
    .query-box button:hover { background: #0369a1; }
    .results-area { flex: 1; overflow-y: auto; padding: 16px; font-size: 0.85rem; line-height: 1.5; color: #cbd5e1; }
    .node-card { background: #0f172a; padding: 12px; border-radius: 6px; border: 1px solid #334155; margin-bottom: 12px; }
    .node-card h3 { color: #38bdf8; font-size: 0.9rem; margin-bottom: 4px; }
    /* Schematic View */
    .schematic-container { flex: 1; display: flex; justify-content: center; align-items: center; padding: 20px; }
    /* Multi-Agent View */
    .agent-grid { flex: 1; padding: 24px; display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; overflow-y: auto; }
    .agent-card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; }
    .agent-card h3 { font-size: 1rem; color: #38bdf8; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center; }
    .agent-risk { font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; }
    .risk-high { background: #ef4444; color: white; }
    .risk-med { background: #f59e0b; color: white; }
    .risk-low { background: #10b981; color: white; }
  </style>
</head>
<body>
  <header>
    <h1>🛢️ Wellbore Knowledge Graph <span class="badge">GraphRAG & Subsurface AI</span></h1>
    <div class="nav-tabs">
      <button class="tab-btn active" onclick="switchTab('graphView', this)">Knowledge Graph</button>
      <button class="tab-btn" onclick="switchTab('schematicView', this)">Wellbore Schematic</button>
      <button class="tab-btn" onclick="switchTab('agentView', this)">Multi-Agent Panel</button>
      <button class="tab-btn" onclick="switchTab('casingView', this)">Casing Integrity (MASP)</button>
    </div>
    <div style="font-size: 0.85rem; color: #94a3b8;">Status: <span style="color: #4ade80;">Online</span></div>
  </header>

  <div class="main-container">
    <!-- View 1: Knowledge Graph -->
    <div id="graphView" class="view-content active">
      <div id="cy"></div>
    </div>

    <!-- View 2: Wellbore Schematic -->
    <div id="schematicView" class="view-content">
      <div class="schematic-container" id="schematicTarget">
        <div style="color: #64748b;">Loading 2D Casing & Trajectory Schematic...</div>
      </div>
    </div>

    <!-- View 3: Multi-Agent Panel -->
    <div id="agentView" class="view-content">
      <div class="agent-grid" id="agentGridTarget">
        <div class="agent-card">
          <h3>Lead Superintendent <span class="agent-risk risk-low">ACTIVE</span></h3>
          <p style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Role: Operational sequence coordinator & task triage.</p>
          <div style="margin-top: 10px; font-size: 0.85rem;">Ready to direct subsurface engineering inquiries.</div>
        </div>
        <div class="agent-card">
          <h3>Geomechanics Specialist <span class="agent-risk risk-med">ACTIVE</span></h3>
          <p style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Role: Lithology, pore pressure gradient & sloughing shale.</p>
          <div style="margin-top: 10px; font-size: 0.85rem;">Monitors Draupne, Norphlet, and Eagle Ford intervals.</div>
        </div>
        <div class="agent-card">
          <h3>Equipment Reliability <span class="agent-risk risk-high">ACTIVE</span></h3>
          <p style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Role: Asset serial records, thermal ratings & BHA fatigue.</p>
          <div style="margin-top: 10px; font-size: 0.85rem;">Tracking BOP-HYD-5000-08, MM-775-A1, and MWD-TENSOR-882.</div>
        </div>
        <div class="agent-card">
          <h3>Well Control Auditor <span class="agent-risk risk-low">COMPLIANT</span></h3>
          <p style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Role: API RP 53 / NORSOK D-010 Two-Barrier enforcement.</p>
          <div style="margin-top: 10px; font-size: 0.85rem;">Auditing primary hydrostatic & secondary mechanical barriers.</div>
        </div>
      </div>
    </div>

    <!-- View 4: Casing & MASP Calculator -->
    <div id="casingView" class="view-content" style="padding: 24px; flex-direction: column;">
      <h2 style="font-size: 1.1rem; color: #38bdf8; margin-bottom: 12px;">Casing Mechanical Stress & MASP Verification</h2>
      <div style="display: flex; gap: 16px; margin-bottom: 16px;">
        <button class="tab-btn active" onclick="runCasingAudit('CSG-958-L80')">Evaluate CSG-958-L80 (9-5/8")</button>
        <button class="tab-btn active" onclick="runCasingAudit('CSG-700-Q125')">Evaluate CSG-700-Q125 (7")</button>
      </div>
      <div id="casingResultTarget" class="node-card" style="max-width: 700px;">
        Click an evaluation button above to execute downhole burst, collapse, MASP, and kick tolerance audit.
      </div>
    </div>

    <!-- Persistent Engineering Sidebar -->
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
        <button onclick="runQuery()">Run Grounded GraphRAG</button>
        <button onclick="runMultiAgent()" style="background: #8b5cf6; margin-top: 6px;">Consult Multi-Agent Committee</button>
      </div>
      <div class="results-area" id="resultsArea">
        <div style="color: #64748b; text-align: center; margin-top: 30px;">
          Click any graph node to inspect properties or submit an engineering query.
        </div>
      </div>
    </div>
  </div>

  <script>
    function switchTab(viewId, btn) {
      document.querySelectorAll('.view-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(viewId).classList.add('active');
      btn.classList.add('active');
      if (viewId === 'schematicView') loadSchematic();
    }

    async function loadSchematic() {
      try {
        const resp = await fetch('/api/v1/wellbore/Well%20104-B/schematic');
        const data = await resp.json();
        document.getElementById('schematicTarget').innerHTML = data.svg_schematic;
      } catch(e) {
        document.getElementById('schematicTarget').innerHTML = '<div style="color:#ef4444">Failed loading schematic.</div>';
      }
    }

    async function runCasingAudit(tag) {
      const resTarget = document.getElementById('casingResultTarget');
      resTarget.innerHTML = '<div style="color:#38bdf8;">Evaluating downhole geomechanical stresses & API burst limits...</div>';
      try {
        const resp = await fetch('/api/v1/engineering/casing-check', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({casing_serial_or_tag: tag, shoe_depth_tvd_ft: 12500, target_formation_ppg: 14.8, mud_weight_ppg: 15.2})
        });
        const d = await resp.json();
        resTarget.innerHTML = `
          <h3>[${d.casing_tag}] ${d.grade} (${d.outer_diameter_in}") - Status: ${d.overall_integrity_status}</h3>
          <p style="margin-top: 8px;"><strong>MASP (Max Anticipated Surface Pressure):</strong> ${d.maximum_anticipated_surface_pressure_masp_psi} psi</p>
          <p><strong>Burst Rating:</strong> ${d.burst_rating_psi} psi | <strong>Safety Factor:</strong> ${d.burst_safety_factor} (${d.burst_status})</p>
          <p><strong>Collapse Rating:</strong> ${d.collapse_rating_psi} psi | <strong>Safety Factor:</strong> ${d.collapse_safety_factor} (${d.collapse_status})</p>
          <p><strong>Maximum Allowable Kick Volume:</strong> ${d.kick_tolerance_bbl} bbl</p>
          <div style="margin-top: 10px; padding: 8px; background: #1e293b; border-radius: 4px; font-size: 0.85rem; color: #38bdf8;">
            ${d.recommendation}
          </div>
        `;
      } catch(e) {
        resTarget.innerHTML = '<div style="color:#ef4444">Error calculating casing margins.</div>';
      }
    }

    async function runMultiAgent() {
      const q = document.getElementById('queryInput').value || 'What SOPs mitigate drillstring pack-off and BOP ram leaks?';
      const resArea = document.getElementById('resultsArea');
      resArea.innerHTML = '<div style="color: #a855f7;">Consulting 4 autonomous drilling specialists (Superintendent, Geomechanics, Reliability, Well Control)...</div>';
      try {
        const resp = await fetch('/api/v1/query/multi-agent', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({inquiry: q, well_name: 'Well 104-B'})
        });
        const data = await resp.json();
        resArea.innerHTML = `
          <div class="node-card" style="border-color: #8b5cf6;">
            <h3 style="color:#c084fc;">Committee Consensus: ${data.operational_classification}</h3>
            <p style="font-size: 0.75rem; color: #4ade80; margin-top: 4px;">Barrier Status: ${data.two_barrier_compliance_status}</p>
            <div style="margin-top: 8px; font-size: 0.82rem; white-space: pre-wrap;">${data.final_advisory}</div>
          </div>
        `;
      } catch(e) {
        resArea.innerHTML = '<div style="color:#ef4444;">Multi-agent consultation error.</div>';
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

    window.onload = initGraph;
  </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)
    return HTMLResponse(content=html_content)
