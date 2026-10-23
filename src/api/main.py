"""Main FastAPI Application Entrypoint for Wellbore Knowledge Graph."""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import settings
from src.graph.client import neo4j_client

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("wellbore_kg")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown procedures."""
    logger.info("Initializing %s v%s...", settings.APP_NAME, settings.APP_VERSION)

    # 1. Initialize Neo4j connection and schema
    connected = neo4j_client.connect()
    if connected:
        logger.info("Connected to Neo4j. Applying schema constraints and vector indexes...")
        schema_results = neo4j_client.init_schema()
        logger.info("Schema initialization results: %s", schema_results)
    else:
        logger.warning("Neo4j database not reachable at startup (%s). Running in resilient buffer mode.", settings.NEO4J_URI)

    yield

    # Shutdown
    logger.info("Shutting down %s...", settings.APP_NAME)
    neo4j_client.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Advanced GraphRAG (Graph Retrieval-Augmented Generation) system for subsurface & wellbore engineering. "
            "Combines vector search with multi-hop Cypher graph traversals over equipment assets, failure modes, "
            "formation lithology, and IADC operating standards."
        ),
        lifespan=lifespan,
    )

    # Enable CORS for frontend/dashboard integrations
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    application.include_router(router)

    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
