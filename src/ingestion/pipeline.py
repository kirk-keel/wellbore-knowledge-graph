"""End-to-end ingestion pipeline: Parsing -> Chunking -> Extraction -> Embedding -> Graph Storage."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings
from src.embeddings.provider import EmbeddingProvider, embedding_provider
from src.graph.client import Neo4jClient, neo4j_client
from src.graph.queries import (
    MERGE_CHUNK,
    MERGE_DOCUMENT,
    MERGE_EQUIPMENT,
    MERGE_FAILURE_MODE,
    MERGE_FORMATION,
    MERGE_INCIDENT,
    MERGE_SOP,
    MERGE_WELL,
)
from src.ingestion.chunker import WellboreDomainChunker
from src.ingestion.entity_extractor import WellboreEntityExtractor
from src.ingestion.pdf_parser import TechnicalDocumentParser
from src.ingestion.las_parser import LASDocumentParser
from src.ingestion.ddr_timeline import DDRTimelineParser
from src.models.api import IngestResponse
from src.models.domain import DocumentChunk, WellboreExtractionResult, Relationship, FailureModeEntity

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Coordinates parsing, semantic chunking, entity extraction, embeddings, and Neo4j graph writes."""

    def __init__(
        self,
        parser: Optional[TechnicalDocumentParser] = None,
        chunker: Optional[WellboreDomainChunker] = None,
        extractor: Optional[WellboreEntityExtractor] = None,
        embedder: Optional[EmbeddingProvider] = None,
        graph_client: Optional[Neo4jClient] = None,
    ):
        self.parser = parser or TechnicalDocumentParser()
        self.chunker = chunker or WellboreDomainChunker()
        self.extractor = extractor or WellboreEntityExtractor()
        self.embedder = embedder or embedding_provider
        self.graph = graph_client or neo4j_client

        # Local cache for offline/testing mode
        self.in_memory_docs: Dict[str, Any] = {}
        self.in_memory_chunks: Dict[str, DocumentChunk] = {}
        self.in_memory_extractions: Dict[str, WellboreExtractionResult] = {}

    def ingest_file(self, file_path: str | Path, doc_id: Optional[str] = None, category: str = "engineering_manual") -> IngestResponse:
        """Process and ingest a local file (PDF, TXT, MD, LAS, DDR) into the Knowledge Graph."""
        path = Path(file_path)
        actual_doc_id = doc_id or path.stem

        if path.suffix.lower() == ".las":
            las_parser = LASDocumentParser()
            well, curves, anomalies = las_parser.parse_las_file(path)

            summary_text = (
                f"# WELL LOG CURVE SUMMARY: {well.name}\n"
                f"Field: {well.field} | Operator: {well.operator} | Total Depth: {well.total_depth_ft} ft\n"
                f"Total Depth Intervals Logged: {len(curves)}\n\n"
                f"## DOWNHOLE ANOMALY FLAGS & PRECURSORS\n"
            )
            for anom in anomalies:
                summary_text += f"- Depth {anom['depth_ft']} ft: [{anom['severity']}] {anom['anomaly_type']} ({anom['failure_mode']}) - {anom['details']}\n"

            raw_docs = self.parser.parse_raw_text(
                summary_text,
                title=f"LWD Log {path.name}",
                doc_id=actual_doc_id,
                metadata={"source": path.name, "category": "well_log", "well_name": well.name},
            )
            response = self._process_langchain_documents(raw_docs, actual_doc_id, category="well_log", title=f"LWD Log {path.name}")
            
            # Inject LAS well and anomaly failure modes into extraction cache
            if actual_doc_id in self.in_memory_extractions:
                ext = self.in_memory_extractions[actual_doc_id]
                ext.wells.append(well)
                for anom in anomalies:
                    ext.failure_modes.append(FailureModeEntity(
                        code=anom["failure_mode"],
                        failure_type=anom["anomaly_type"],
                        severity=anom["severity"],
                        description=anom["details"],
                    ))
            return response

        elif "ddr" in path.stem.lower() or "daily_drilling_reports" in path.stem.lower():
            ddr_parser = DDRTimelineParser()
            text_content = path.read_text(encoding="utf-8", errors="replace")
            ddrs, ddr_rels = ddr_parser.parse_ddr_text(text_content)

            meta = {"source": path.name, "category": "daily_drilling_report"}
            raw_docs = self.parser.parse_file(path, doc_id=actual_doc_id, metadata=meta)
            response = self._process_langchain_documents(raw_docs, actual_doc_id, category="daily_drilling_report", title=path.name)

            # Store DDR entries and temporal sequence
            if actual_doc_id in self.in_memory_extractions:
                ext = self.in_memory_extractions[actual_doc_id]
                ext.ddrs.extend(ddrs)
                ext.relationships.extend(ddr_rels)
            return response

        meta = {"source": path.name, "category": category}
        raw_docs = self.parser.parse_file(path, doc_id=doc_id, metadata=meta)
        return self._process_langchain_documents(raw_docs, doc_id or path.stem, category=category, title=path.name)

    def ingest_raw_text(self, text: str, title: str, doc_id: Optional[str] = None, source: str = "manual_entry", category: str = "engineering_report") -> IngestResponse:
        """Process and ingest a raw engineering text string into the Knowledge Graph."""
        actual_doc_id = doc_id or f"doc_{abs(hash(title + text[:30]))}"
        meta = {"source": source, "category": category, "title": title}
        raw_docs = self.parser.parse_raw_text(text, title=title, doc_id=actual_doc_id, metadata=meta)
        return self._process_langchain_documents(raw_docs, actual_doc_id, category=category, title=title)

    def _process_langchain_documents(self, documents: List[Any], doc_id: str, category: str, title: str) -> IngestResponse:
        """Execute chunking, extraction, embedding, and Neo4j graph ingestion."""
        # 1. Chunk documents
        chunks = self.chunker.chunk_documents(documents, doc_id_override=doc_id)
        if not chunks:
            raise ValueError(f"No textual content could be parsed from document {doc_id}")

        # 2. Generate vector embeddings for chunks
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(chunk_texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # 3. Extract domain entities and relations across chunk texts
        combined_text = "\n\n".join(chunk_texts)
        extraction = self.extractor.extract_from_text(combined_text)

        # 4. Save to Neo4j (and local cache)
        relationships_count = self._store_in_graph(doc_id, title, category, chunks, extraction)

        # Store in local memory cache
        self.in_memory_docs[doc_id] = {"title": title, "category": category, "chunks_count": len(chunks)}
        for c in chunks:
            self.in_memory_chunks[c.id] = c
        self.in_memory_extractions[doc_id] = extraction

        entities_count = {
            "wells": len(extraction.wells),
            "equipment": len(extraction.equipment),
            "formations": len(extraction.formations),
            "failure_modes": len(extraction.failure_modes),
            "sops": len(extraction.sops),
            "incidents": len(extraction.incidents),
        }

        return IngestResponse(
            doc_id=doc_id,
            status="SUCCESS",
            chunks_count=len(chunks),
            entities_extracted=entities_count,
            relationships_created=relationships_count,
            extraction_details=extraction,
            message=f"Successfully ingested '{title}' into knowledge graph with {len(chunks)} chunks and {sum(entities_count.values())} entities.",
        )

    def _store_in_graph(
        self,
        doc_id: str,
        title: str,
        category: str,
        chunks: List[DocumentChunk],
        extraction: WellboreExtractionResult,
    ) -> int:
        """Write nodes and edges into Neo4j."""
        rel_counter = 0

        # If Neo4j is offline, return count of planned relationships
        if not self.graph.is_connected():
            logger.info("Neo4j offline. Graph writes buffered in in-memory state.")
            return len(extraction.relationships) + len(chunks)

        with self.graph.get_session() as session:
            # 1. Upsert Document Node
            session.run(
                MERGE_DOCUMENT,
                id=doc_id,
                title=title,
                source=chunks[0].metadata.get("source", "unknown"),
                category=category,
                metadata={},
            )

            # 2. Upsert Chunks
            for chunk in chunks:
                session.run(
                    MERGE_CHUNK,
                    id=chunk.id,
                    doc_id=chunk.doc_id,
                    chunk_index=chunk.chunk_index,
                    page=chunk.page or 1,
                    text=chunk.text,
                    embedding=chunk.embedding,
                    section_title=chunk.metadata.get("section_title") or "",
                )
                rel_counter += 1

            # 3. Upsert Entities
            for w in extraction.wells:
                session.run(
                    MERGE_WELL,
                    name=w.name,
                    api_number=w.api_number or "",
                    field=w.field or "",
                    water_depth_ft=w.water_depth_ft or 0.0,
                    total_depth_ft=w.total_depth_ft or 0.0,
                    operator=w.operator or "",
                )

            for eq in extraction.equipment:
                session.run(
                    MERGE_EQUIPMENT,
                    serial_number=eq.serial_number,
                    equipment_type=eq.equipment_type,
                    manufacturer=eq.manufacturer or "",
                    model=eq.model or "",
                    working_pressure_psi=eq.working_pressure_psi or 0.0,
                )

            for f in extraction.formations:
                session.run(
                    MERGE_FORMATION,
                    name=f.name,
                    lithology=f.lithology,
                    top_depth_ft=f.top_depth_ft or 0.0,
                    bottom_depth_ft=f.bottom_depth_ft or 0.0,
                    pore_pressure_ppg=f.pore_pressure_ppg or 0.0,
                    fracture_gradient_ppg=f.fracture_gradient_ppg or 0.0,
                    hazards=f.hazards,
                )

            for fm in extraction.failure_modes:
                session.run(
                    MERGE_FAILURE_MODE,
                    code=fm.code,
                    failure_type=fm.failure_type,
                    severity=fm.severity,
                    description=fm.description,
                    root_cause=fm.root_cause or "",
                    symptoms=fm.symptoms,
                )

            for s in extraction.sops:
                session.run(
                    MERGE_SOP,
                    sop_id=s.sop_id,
                    title=s.title,
                    category=s.category,
                    standard_reference=s.standard_reference or "",
                    required_barriers=s.required_barriers,
                    steps=s.steps,
                )

            for inc in extraction.incidents:
                session.run(
                    MERGE_INCIDENT,
                    incident_id=inc.incident_id,
                    well_name=inc.well_name or "",
                    date=inc.date or "",
                    npt_hours=inc.npt_hours or 0.0,
                    depth_ft=inc.depth_ft or 0.0,
                    description=inc.description,
                    corrective_actions=inc.corrective_actions,
                )

            # 4. Link Chunks to Entities mentioned
            entity_keys = []
            entity_keys.extend([w.name for w in extraction.wells])
            entity_keys.extend([e.serial_number for e in extraction.equipment])
            entity_keys.extend([f.name for f in extraction.formations])
            entity_keys.extend([fm.code for fm in extraction.failure_modes])
            entity_keys.extend([s.sop_id for s in extraction.sops])
            entity_keys.extend([i.incident_id for i in extraction.incidents])

            for chunk in chunks:
                for ek in entity_keys:
                    if ek.lower() in chunk.text.lower():
                        session.run(
                            """
                            MATCH (c:DocumentChunk {id: $chunk_id})
                            MATCH (e) WHERE (e.name = $ek OR e.serial_number = $ek OR e.code = $ek OR e.sop_id = $ek OR e.incident_id = $ek)
                            MERGE (c)-[:MENTIONS]->(e)
                            """,
                            chunk_id=chunk.id,
                            ek=ek,
                        )
                        rel_counter += 1

            # 5. Upsert Structured Domain Relationships
            for rel in extraction.relationships:
                # Dynamic cypher for typed relationship
                cypher_rel = f"""
                MATCH (s:{rel.source_type}) WHERE (s.name = $s_id OR s.serial_number = $s_id OR s.code = $s_id OR s.sop_id = $s_id OR s.incident_id = $s_id)
                MATCH (t:{rel.target_type}) WHERE (t.name = $t_id OR t.serial_number = $t_id OR t.code = $t_id OR t.sop_id = $t_id OR t.incident_id = $t_id)
                MERGE (s)-[r:{rel.relation_type}]->(t)
                SET r += $props
                """
                try:
                    session.run(cypher_rel, s_id=rel.source_id, t_id=rel.target_id, props=rel.properties)
                    rel_counter += 1
                except Exception as e:
                    logger.debug("Error inserting relation %s: %s", rel, e)

        return rel_counter


# Global pipeline instance
ingestion_pipeline = IngestionPipeline()
