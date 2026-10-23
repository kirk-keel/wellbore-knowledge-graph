"""Automated RAG Evaluation benchmark for Wellbore Knowledge Graph (Faithfulness, Relevance, Recall)."""

import pytest
from src.ingestion.pipeline import ingestion_pipeline
from src.models.api import QueryRequest
from src.retrieval.qa_engine import WellboreQAEngine
from src.retrieval.text_to_cypher import TextToCypherAgent
from scripts.seed_database import seed


@pytest.fixture(scope="module", autouse=True)
def prepare_seeded_graph():
    """Ensure sample engineering documents are loaded in memory/graph."""
    seed()


def evaluate_faithfulness(answer: str, ground_truth_entities: list[str]) -> float:
    """Computes grounding score: checks if cited entities match verified entities."""
    if not ground_truth_entities:
        return 1.0
    matched = sum(1 for e in ground_truth_entities if e.lower() in answer.lower())
    return round(matched / len(ground_truth_entities), 2)


def evaluate_context_recall(retrieved_texts: list[str], target_keywords: list[str]) -> float:
    """Measures if relevant domain concepts were successfully retrieved in the text chunks."""
    all_text = " ".join(retrieved_texts).lower()
    matched = sum(1 for kw in target_keywords if kw.lower() in all_text)
    return round(matched / len(target_keywords), 2)


def test_eval_bop_ram_leak_scenario():
    """Scenario 1: Deepwater Titan BOP Ram Leak & Influx Incident."""
    qa_engine = WellboreQAEngine()
    req = QueryRequest(
        question="What failure mode occurred on BOP-HYD-5000-08 and what standards mitigate it?",
        top_k_chunks=3,
        graph_hops=2,
    )
    res = qa_engine.answer_query(req)

    # 1. Faithfulness: Cites verified failure and SOPs
    faithfulness = evaluate_faithfulness(res.answer, ["FAIL-RAMLEAK", "API-RP-53"])
    assert faithfulness >= 0.50

    # 2. Context Recall: Retrieved chunks must mention BOP and incident
    chunk_texts = [c.text for c in res.retrieved_chunks]
    recall = evaluate_context_recall(chunk_texts, ["BOP-HYD-5000-08", "FAIL-RAMLEAK"])
    assert recall >= 0.50

    # 3. Two-Barrier Rule Safety Check
    assert "Barrier Compliance" in res.answer or "Barrier" in res.answer


def test_eval_packoff_draupne_scenario():
    """Scenario 2: Drillstring Pack-Off in reactive shale."""
    qa_engine = WellboreQAEngine()
    req = QueryRequest(
        question="What SOP remediates drillstring pack-off in Well 104-B?",
        top_k_chunks=3,
        graph_hops=2,
    )
    res = qa_engine.answer_query(req)

    faithfulness = evaluate_faithfulness(res.answer, ["FAIL-PACKOFF", "SOP-SP-774"])
    assert faithfulness >= 0.50
    assert res.confidence_score > 0.40


def test_eval_text_to_cypher_quantitative():
    """Scenario 3: Quantitative Cypher evaluation."""
    agent = TextToCypherAgent()
    res = agent.query("What is the total NPT for Well 104-B?")

    assert "cypher_query" in res
    assert "MATCH" in res["cypher_query"]
    assert "DELETE" not in res["cypher_query"]
    assert res["results_count"] >= 1
