"""Vector search module using Neo4j Vector Index with cosine fallback."""

import logging
from typing import List
import numpy as np

from src.config import settings
from src.embeddings.provider import EmbeddingProvider, embedding_provider
from src.graph.client import Neo4jClient, neo4j_client
from src.graph.queries import VECTOR_SEARCH_CHUNKS
from src.models.api import RetrievedChunk
from src.models.domain import DocumentChunk

logger = logging.getLogger(__name__)


class Neo4jVectorRetriever:
    """Executes approximate nearest neighbor vector search on DocumentChunks."""

    def __init__(
        self,
        graph_client: Neo4jClient = neo4j_client,
        embedder: EmbeddingProvider = embedding_provider,
        index_name: str = settings.VECTOR_INDEX_NAME,
    ):
        self.graph = graph_client
        self.embedder = embedder
        self.index_name = index_name

    def search(
        self,
        query: str,
        top_k: int = settings.DEFAULT_TOP_K_CHUNKS,
        min_score: float = settings.MIN_SIMILARITY_SCORE,
        fallback_chunks: List[DocumentChunk] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve top-k relevant document chunks for the query."""
        query_vector = self.embedder.embed_text(query)

        # 1. Try Neo4j Vector Search if live
        if self.graph.is_connected():
            try:
                results = self.graph.execute_query(
                    VECTOR_SEARCH_CHUNKS,
                    {
                        "index_name": self.index_name,
                        "top_k": top_k,
                        "query_vector": query_vector,
                        "min_score": min_score,
                    },
                )
                if results:
                    return [
                        RetrievedChunk(
                            chunk_id=r["chunk_id"],
                            doc_id=r["doc_id"],
                            page=r.get("page", 1),
                            text=r["text"],
                            similarity_score=float(r["similarity_score"]),
                        )
                        for r in results
                    ]
            except Exception as e:
                logger.warning("Neo4j vector search execution note: %s. Using in-memory fallback.", e)

        # 2. In-memory cosine fallback for offline/test environments
        if fallback_chunks:
            return self._search_in_memory(query_vector, fallback_chunks, top_k, min_score)

        return []

    def _search_in_memory(
        self,
        query_vec: List[float],
        chunks: List[DocumentChunk],
        top_k: int,
        min_score: float,
    ) -> List[RetrievedChunk]:
        """Compute cosine similarity over cached chunks."""
        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-9:
            q = q / q_norm

        scored: List[RetrievedChunk] = []
        for c in chunks:
            if not c.embedding:
                c.embedding = self.embedder.embed_text(c.text)

            v = np.array(c.embedding, dtype=np.float32)
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-9:
                v = v / v_norm
            score = float(np.dot(q, v))

            if score >= min_score or len(chunks) <= top_k:
                scored.append(
                    RetrievedChunk(
                        chunk_id=c.id,
                        doc_id=c.doc_id,
                        page=c.page or 1,
                        text=c.text,
                        similarity_score=round(score, 4),
                    )
                )

        scored.sort(key=lambda x: x.similarity_score, reverse=True)
        return scored[:top_k]
