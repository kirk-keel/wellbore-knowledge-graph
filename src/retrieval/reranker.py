"""Reciprocal Rank Fusion (RRF) and Cross-Channel Reranker for Hybrid GraphRAG."""

import logging
from typing import Dict, List, Optional
from src.models.api import GraphPathSegment, RetrievedChunk

logger = logging.getLogger(__name__)


class HybridReranker:
    """
    Reranks candidate retrieved chunks using Reciprocal Rank Fusion (RRF)
    between vector semantic similarity and graph topological centrality.
    """

    def __init__(self, rrf_k: int = 60, graph_weight: float = 0.4):
        self.rrf_k = rrf_k
        self.graph_weight = graph_weight

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        graph_paths: List[GraphPathSegment],
        top_n: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Apply Reciprocal Rank Fusion to rerank chunks based on vector rank and graph alignment."""
        if not chunks:
            return []

        # 1. Vector Channel Ranking
        vector_sorted = sorted(chunks, key=lambda c: c.similarity_score, reverse=True)
        vector_ranks = {c.chunk_id: rank + 1 for rank, c in enumerate(vector_sorted)}

        # 2. Graph Centrality Channel
        # Count mentions of entities that appear in active graph traversal paths
        active_entities = set()
        for p in graph_paths:
            active_entities.add(p.source_id.lower())
            active_entities.add(p.source_name.lower())
            active_entities.add(p.target_id.lower())
            active_entities.add(p.target_name.lower())

        graph_scores: Dict[str, float] = {}
        for c in chunks:
            c_text_lower = c.text.lower()
            matched_entities = sum(1 for e in active_entities if e and e in c_text_lower)
            graph_scores[c.chunk_id] = matched_entities

        graph_sorted = sorted(chunks, key=lambda c: graph_scores.get(c.chunk_id, 0), reverse=True)
        graph_ranks = {c.chunk_id: rank + 1 for rank, c in enumerate(graph_sorted)}

        # 3. Compute Reciprocal Rank Fusion (RRF) Score
        combined_scores: Dict[str, float] = {}
        for c in chunks:
            cid = c.chunk_id
            v_rank = vector_ranks.get(cid, len(chunks))
            g_rank = graph_ranks.get(cid, len(chunks))

            v_rrf = 1.0 / (self.rrf_k + v_rank)
            g_rrf = 1.0 / (self.rrf_k + g_rank)

            combined_scores[cid] = (1.0 - self.graph_weight) * v_rrf + self.graph_weight * g_rrf

        # Update similarity_score with normalized RRF score
        max_rrf = max(combined_scores.values()) if combined_scores else 1.0
        for c in chunks:
            # Calibrate combined score
            normalized_score = round(combined_scores[c.chunk_id] / max_rrf, 4)
            c.similarity_score = normalized_score

        chunks.sort(key=lambda c: c.similarity_score, reverse=True)
        return chunks[:top_n] if top_n else chunks
