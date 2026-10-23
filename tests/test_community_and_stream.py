"""Tests for Community Detection, Safety Guardrails, Text-to-Cypher, and Visualization."""

from fastapi.testclient import TestClient
from src.api.main import app
from src.retrieval.community_summary import CommunityDetector
from src.retrieval.safety_guardrails import TwoBarrierSafetyGuardrail

client = TestClient(app)


def test_community_detection():
    detector = CommunityDetector()
    communities = detector.detect_communities()
    assert len(communities) >= 1
    comm = communities[0]
    assert "community_id" in comm
    assert "formation" in comm

    # Summary generation
    summary = detector.generate_community_summary(comm["community_id"])
    assert "executive_summary" in summary


def test_two_barrier_safety_guardrail():
    guardrail = TwoBarrierSafetyGuardrail()

    # Well control with both barriers
    compliant_text = "Primary barrier maintained with 12.5 ppg hydrostatic mud weight. Secondary barrier secured by BOP pipe rams."
    audit1 = guardrail.audit_engineering_response("How to shut in on a kick?", compliant_text)
    assert audit1["two_barrier_status"] == "VERIFIED_COMPLIANT"

    # Deficient barrier
    deficient_text = "Close the pipe rams immediately."
    audit2 = guardrail.audit_engineering_response("How to shut in on a kick?", deficient_text)
    assert audit2["two_barrier_status"] == "SAFETY_WARNING_DEFICIENT_BARRIER"


def test_text_to_cypher_endpoint():
    res = client.post("/api/v1/query/cypher", json={"question": "What is the total NPT for Well 104-B?"})
    assert res.status_code == 200
    data = res.json()
    assert "cypher_query" in data
    assert "results" in data


def test_communities_endpoints():
    res1 = client.get("/api/v1/graph/communities")
    assert res1.status_code == 200
    comms = res1.json()
    assert isinstance(comms, list)
    if comms:
        cid = comms[0]["community_id"]
        res2 = client.get(f"/api/v1/graph/communities/{cid}/summary")
        assert res2.status_code == 200


def test_dashboard_endpoint():
    res = client.get("/api/v1/dashboard")
    assert res.status_code == 200
    assert "cytoscape" in res.text
    assert "Wellbore Knowledge Graph" in res.text


def test_visual_graph_endpoint():
    res = client.get("/api/v1/graph/visual")
    assert res.status_code == 200
    data = res.json()
    assert "elements" in data
    assert "nodes" in data["elements"]
    assert "edges" in data["elements"]
