#!/usr/bin/env bash
set -e

echo "=== Starting Wellbore Knowledge Graph API Service ==="

# Wait for Neo4j if configured
if [ -n "$NEO4J_URI" ]; then
    echo "Checking Neo4j connectivity at $NEO4J_URI..."
fi

# Wait for Ollama if configured
if [ -n "$OLLAMA_BASE_URL" ]; then
    echo "Checking Ollama LLM provider at $OLLAMA_BASE_URL..."
fi

# Run database schema / migration initialization if scripts exist
if [ -f "scripts/seed_database.py" ] && [ "$AUTO_SEED" = "true" ]; then
    echo "Auto-seeding database schema..."
    python scripts/seed_database.py || echo "Warning: Seed script returned non-zero exit code, continuing..."
fi

echo "Starting Uvicorn web server on port 8000..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
