"""Tests for Cypher query templates and safety guardrails."""

import pytest
from src.retrieval.graph_traversal import MultiHopGraphTraverser
from src.graph.queries import MERGE_WELL, MERGE_EQUIPMENT, VECTOR_SEARCH_CHUNKS


def test_cypher_templates_contain_expected_labels():
    assert "MERGE (w:Well" in MERGE_WELL
    assert "MERGE (e:Equipment" in MERGE_EQUIPMENT
    assert "CALL db.index.vector.queryNodes" in VECTOR_SEARCH_CHUNKS


def test_safe_cypher_guardrail_mutation_rejection():
    traverser = MultiHopGraphTraverser()

    # Mutation queries must raise ValueError
    with pytest.raises(ValueError, match="disallowed mutation keyword"):
        traverser.execute_safe_cypher("MATCH (n:Well) DETACH DELETE n")

    with pytest.raises(ValueError, match="disallowed mutation keyword"):
        traverser.execute_safe_cypher("CREATE (n:Test {name: 'fake'})")

    with pytest.raises(ValueError, match="disallowed mutation keyword"):
        traverser.execute_safe_cypher("MATCH (n:Well) SET n.name = 'hacked'")
