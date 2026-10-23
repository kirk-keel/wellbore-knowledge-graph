"""Seeds the Wellbore Knowledge Graph with domain standards, incident reports, and SOPs."""

import logging
from pathlib import Path
from src.graph.client import neo4j_client
from src.ingestion.pipeline import ingestion_pipeline
from src.models.api import QueryRequest
from src.retrieval.qa_engine import WellboreQAEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed")


def seed():
    logger.info("==================================================================")
    logger.info("  WELLBORE KNOWLEDGE GRAPH: SEEDING DOMAIN KNOWLEDGE BASE       ")
    logger.info("==================================================================")

    # 1. Connect and initialize schema
    connected = neo4j_client.connect()
    if connected:
        logger.info("Neo4j is online. Initializing constraints and vector index...")
        schema_status = neo4j_client.init_schema()
        logger.info("Schema result: %s", schema_status)
    else:
        logger.warning("Neo4j offline. Documents will be seeded into in-memory Knowledge Graph.")

    # 2. Ingest sample documents
    sample_dir = Path("data/sample_docs")
    if not sample_dir.exists():
        logger.error("Sample documents directory not found: %s", sample_dir)
        return

    doc_files = list(sample_dir.glob("*.txt")) + list(sample_dir.glob("*.pdf")) + list(sample_dir.glob("*.md"))
    logger.info("Found %d engineering documents to ingest.", len(doc_files))

    total_chunks = 0
    total_entities = 0

    for file_path in doc_files:
        logger.info("Ingesting: %s", file_path.name)
        category = "incident_report" if "incident" in file_path.name.lower() else "engineering_standard"
        resp = ingestion_pipeline.ingest_file(file_path=file_path, category=category)
        logger.info("  -> Chunks: %d | Entities: %s", resp.chunks_count, resp.entities_extracted)
        total_chunks += resp.chunks_count
        total_entities += sum(resp.entities_extracted.values())

    logger.info("------------------------------------------------------------------")
    logger.info("Seeding complete! Total Chunks: %d | Total Entities: %d", total_chunks, total_entities)
    logger.info("------------------------------------------------------------------")

    # 3. Test verification query
    logger.info("Running verification GraphRAG query...")
    qa_engine = WellboreQAEngine()
    test_query = QueryRequest(
        question="What SOPs mitigate drillstring pack-off and BOP ram leak failures?",
        top_k_chunks=3,
        graph_hops=2,
    )
    result = qa_engine.answer_query(test_query)

    logger.info("Sample Verification Query: '%s'", test_query.question)
    logger.info("Confidence Score: %.2f", result.confidence_score)
    logger.info("Answer Summary:\n%s", result.answer)
    logger.info("Sources Consulted: %s", result.sources_consulted)
    logger.info("==================================================================")


if __name__ == "__main__":
    seed()
