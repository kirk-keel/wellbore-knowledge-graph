"""Configuration settings for wellbore-knowledge-graph."""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application settings
    APP_NAME: str = "Wellbore Engineering GraphRAG"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: Literal["development", "production", "test"] = "development"
    DEBUG: bool = False

    # Neo4j Settings
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "wellborepassword123"
    NEO4J_DATABASE: str = "neo4j"
    NEO4J_MAX_CONNECTION_POOL_SIZE: int = 50

    # Vector Index Settings
    VECTOR_INDEX_NAME: str = "chunk_vector_index"
    EMBEDDING_DIMENSION: int = 384
    SIMILARITY_METRIC: str = "cosine"

    # Ollama / LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_LLM_MODEL: str = "llama3.2:latest"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    FALLBACK_EMBED_PROVIDER: Literal["deterministic", "ollama"] = "deterministic"
    ALLOW_MOCK_LLM_FALLBACK: bool = True

    # Ingestion & Chunking Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Retrieval & Search Settings
    DEFAULT_TOP_K_CHUNKS: int = 5
    DEFAULT_GRAPH_HOPS: int = 2
    MAX_GRAPH_PATHS: int = 25
    MIN_SIMILARITY_SCORE: float = 0.10


settings = Settings()
