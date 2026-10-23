"""Hybrid retriever combining vector semantic search with multi-hop graph expansion."""

import logging
from typing import Dict, List, Optional, Tuple

from src.config import settings
from src.ingestion.pipeline import IngestionPipeline, ingestion_pipeline
from src.models.api import EntityCitation, GraphPathSegment, RetrievedChunk
from src.retrieval.graph_traversal import MultiHopGraphTraverser
from src.retrieval.vector_search import Neo4jVectorRetriever
from src.retrieval.reranker import HybridReranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Orchestrates dual-channel retrieval: dense vector similarity + symbolic graph exploration."""

    def __init__(
        self,
        vector_retriever: Optional[Neo4jVectorRetriever] = None,
        graph_traverser: Optional[MultiHopGraphTraverser] = None,
        pipeline: Optional[IngestionPipeline] = None,
        reranker: Optional[HybridReranker] = None,
    ):
        self.vector_retriever = vector_retriever or Neo4jVectorRetriever()
        self.graph_traverser = graph_traverser or MultiHopGraphTraverser()
        self.pipeline = pipeline or ingestion_pipeline
        self.reranker = reranker or HybridReranker()

    def retrieve(
        self,
        query: str,
        top_k: int = settings.DEFAULT_TOP_K_CHUNKS,
        graph_hops: int = settings.DEFAULT_GRAPH_HOPS,
        min_similarity: float = settings.MIN_SIMILARITY_SCORE,
    ) -> Tuple[List[RetrievedChunk], List[GraphPathSegment], List[EntityCitation]]:
        """Execute hybrid retrieval returning vector chunks, graph paths, and entity citations."""
        # 1. Vector Search
        cached_chunks = list(self.pipeline.in_memory_chunks.values())
        retrieved_chunks = self.vector_retriever.search(
            query=query,
            top_k=top_k * 2,  # Retrieve more candidates for reranking
            min_score=min_similarity,
            fallback_chunks=cached_chunks,
        )

        chunk_ids = [c.chunk_id for c in retrieved_chunks]

        # 2. Multi-Hop Graph Traversal
        all_cached_relations = []
        for ext in self.pipeline.in_memory_extractions.values():
            all_cached_relations.extend(ext.relationships)

        graph_paths = self.graph_traverser.expand_from_chunks(
            chunk_ids=chunk_ids,
            max_hops=graph_hops,
            fallback_relations=all_cached_relations,
        )

        # 3. Reciprocal Rank Fusion (RRF) Reranking
        reranked_chunks = self.reranker.rerank(
            query=query,
            chunks=retrieved_chunks,
            graph_paths=graph_paths,
            top_n=top_k,
        )

        # 4. Entity Citation and Grounding Resolution
        entity_citations = self._extract_citations(reranked_chunks, graph_paths)

        return reranked_chunks, graph_paths, entity_citations

    def _extract_citations(
        self,
        chunks: List[RetrievedChunk],
        paths: List[GraphPathSegment],
    ) -> List[EntityCitation]:
        """Collect and deduplicate all engineering entities grounded in the retrieved graph paths."""
        citations_map: Dict[str, EntityCitation] = {}

        for p in paths:
            # Source entity
            s_key = f"{p.source_label}:{p.source_id}"
            if s_key not in citations_map:
                citations_map[s_key] = EntityCitation(
                    entity_type=p.source_label,
                    entity_id=p.source_id,
                    entity_name=p.source_name,
                    relation_to_context=f"Connected via {p.relation} to {p.target_name}",
                )

            # Target entity
            t_key = f"{p.target_label}:{p.target_id}"
            if t_key not in citations_map:
                citations_map[t_key] = EntityCitation(
                    entity_type=p.target_label,
                    entity_id=p.target_id,
                    entity_name=p.target_name,
                    relation_to_context=f"Target of {p.relation} from {p.source_name}",
                )

        return list(citations_map.values())
