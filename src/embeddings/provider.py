"""Embeddings provider abstraction supporting Ollama and deterministic fallback."""

import hashlib
import logging
import math
from typing import List
import httpx
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Provides vector embeddings for chunks and user queries."""

    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model_name: str = settings.OLLAMA_EMBED_MODEL,
        dimension: int = settings.EMBEDDING_DIMENSION,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts with fallback."""
        if not texts:
            return []

        # Try Ollama endpoint first if reachable
        try:
            return self._embed_ollama(texts)
        except Exception as e:
            if settings.ALLOW_MOCK_LLM_FALLBACK:
                logger.warning("Ollama embedding failed or unavailable (%s). Using deterministic provider.", e)
                return [self._embed_deterministic(t) for t in texts]
            raise RuntimeError(f"Ollama embedding failed and fallback is disabled: {e}") from e

    def _embed_ollama(self, texts: List[str]) -> List[List[float]]:
        """Query Ollama /api/embeddings or /api/embed endpoint."""
        embeddings: List[List[float]] = []
        with httpx.Client(base_url=self.base_url, timeout=10.0) as client:
            for text in texts:
                resp = client.post(
                    "/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    vec = data.get("embedding", [])
                    if len(vec) == self.dimension:
                        embeddings.append(vec)
                    elif len(vec) > 0:
                        # Resize or adjust if model dimension differs
                        resized = self._project_dimension(vec, self.dimension)
                        embeddings.append(resized)
                    else:
                        embeddings.append(self._embed_deterministic(text))
                else:
                    raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")
        return embeddings

    def _embed_deterministic(self, text: str) -> List[float]:
        """
        Deterministic, semantic-preserving token-hash embedding.
        Ensures unit vector output and reproducible vector distance calculations
        even in test/offline environments.
        """
        words = [w.lower().strip(".,;:?!()[]\"'") for w in text.split() if w.strip()]
        if not words:
            return [0.0] * self.dimension

        vec = np.zeros(self.dimension, dtype=np.float32)

        for i, word in enumerate(words):
            if not word:
                continue
            # Double hashing to produce multiple distinct coordinate activations per word
            h1 = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            h2 = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)

            idx1 = h1 % self.dimension
            idx2 = (h1 >> 16) % self.dimension
            idx3 = h2 % self.dimension

            sign1 = 1.0 if ((h1 >> 8) & 1) else -1.0
            sign2 = 1.0 if ((h2 >> 8) & 1) else -1.0
            sign3 = 1.0 if ((h1 >> 24) & 1) else -1.0

            # Weight early words slightly higher for title/heading emphasis
            pos_weight = 1.0 / math.sqrt(1.0 + 0.05 * i)

            vec[idx1] += sign1 * pos_weight
            vec[idx2] += sign2 * pos_weight * 0.7
            vec[idx3] += sign3 * pos_weight * 0.5

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return [float(x) for x in vec]

    def _project_dimension(self, vector: List[float], target_dim: int) -> List[float]:
        """Project or pad/truncate vector to target dimension."""
        arr = np.array(vector, dtype=np.float32)
        if len(arr) == target_dim:
            return [float(x) for x in arr]
        if len(arr) > target_dim:
            arr = arr[:target_dim]
        else:
            arr = np.pad(arr, (0, target_dim - len(arr)), mode="constant")
        norm = np.linalg.norm(arr)
        if norm > 1e-9:
            arr = arr / norm
        return [float(x) for x in arr]


# Default singleton instance
embedding_provider = EmbeddingProvider()
