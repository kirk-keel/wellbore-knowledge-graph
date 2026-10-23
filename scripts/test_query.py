"""Interactive or CLI query runner for Wellbore Knowledge Graph."""

import argparse
import json
import logging
from src.models.api import QueryRequest
from src.retrieval.qa_engine import WellboreQAEngine
from src.ingestion.pipeline import ingestion_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Query the Wellbore GraphRAG Engine")
    parser.add_argument("query", type=str, nargs="?", default="What SOPs mitigate drillstring pack-off in shale formations?", help="Engineering question to ask")
    parser.add_argument("--top-k", type=int, default=5, help="Number of vector chunks to retrieve")
    parser.add_argument("--hops", type=int, default=2, help="Graph hops to expand")
    args = parser.parse_args()

    # Pre-seed in-memory if needed
    if not ingestion_pipeline.in_memory_chunks:
        from scripts.seed_database import seed
        seed()

    qa_engine = WellboreQAEngine()
    req = QueryRequest(question=args.query, top_k_chunks=args.top_k, graph_hops=args.hops)
    res = qa_engine.answer_query(req)

    print("\n" + "=" * 60)
    print(f"QUERY: {res.question}")
    print("=" * 60)
    print(f"CONFIDENCE SCORE: {res.confidence_score}")
    print("\nGROUNDED ANSWER:\n" + res.answer)
    print("\nREASONING SUMMARY:\n" + res.reasoning_summary)
    print("\nENTITIES CITED:")
    for e in res.entities_cited:
        print(f"  • [{e.entity_type}] {e.entity_id} ({e.entity_name}) - {e.relation_to_context}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
