"""Integration tests for FastAPI endpoints."""

import io
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Wellbore" in data["service"]


def test_ingest_text_endpoint():
    payload = {
        "title": "Eagle Ford Drilling Memo",
        "text": "Well Titan-101 is drilling Eagle Ford Shale at 10,200 ft with mud motor MM-775-A1. Risk of pack-off FAIL-PACKOFF mitigated by SOP-SP-774.",
        "category": "drilling_memo",
    }
    response = client.post("/api/v1/ingest/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["chunks_count"] >= 1
    assert data["entities_extracted"]["equipment"] >= 1


def test_ingest_file_endpoint():
    content = b"INCIDENT: BOP-HYD-5000-08 experienced FAIL-RAMLEAK at Norphlet Formation. Mitigated via API-RP-53."
    file_payload = ("test_incident.txt", io.BytesIO(content), "text/plain")
    response = client.post(
        "/api/v1/ingest/file",
        files={"file": file_payload},
        data={"category": "incident_report"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"


def test_query_vector_endpoint():
    payload = {
        "question": "What equipment is used in Eagle Ford?",
        "top_k_chunks": 3,
        "min_similarity": 0.05,
    }
    response = client.post("/api/v1/query/vector", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_query_hybrid_endpoint():
    payload = {
        "question": "What SOPs mitigate drillstring pack-off and BOP ram leaks?",
        "top_k_chunks": 3,
        "graph_hops": 2,
        "min_similarity": 0.05,
        "include_graph_paths": True,
    }
    response = client.post("/api/v1/query/hybrid", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "confidence_score" in data
    assert len(data["entities_cited"]) >= 1


def test_graph_stats_endpoint():
    response = client.get("/api/v1/graph/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_nodes" in data
    assert "vector_index_status" in data
