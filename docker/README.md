# Docker Deployment & Orchestration Guide

This directory contains container definitions and orchestration configurations for the Wellbore Engineering Knowledge Graph stack.

## Architecture

```
                   ┌────────────────────────────────────────┐
                   │        Reverse Proxy / Ingress         │
                   └──────────────────┬─────────────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               │                                             │
               ▼                                             ▼
  ┌──────────────────────────┐                  ┌──────────────────────────┐
  │  FastAPI GraphRAG Engine │                  │   Neo4j Browser GUI      │
  │      (Port 8000)         │                  │      (Port 7474)         │
  └──────┬─────────────┬─────┘                  └──────────────────────────┘
         │             │
         ▼             ▼
┌─────────────────┐  ┌──────────────────┐
│  Neo4j Graph DB │  │   Ollama LLM     │
│  (Port 7687)    │  │  (Port 11434)    │
│  Vector + APOC  │  │  Embedding + Gen │
└─────────────────┘  └──────────────────┘
```

## Quick Commands

### 1. Launch Production Stack
```bash
docker compose -f docker/docker-compose.yml up -d --build
```
Or from the project root:
```bash
docker compose up -d --build
```

### 2. Launch Development Stack (with live reload)
```bash
docker compose -f docker/docker-compose.dev.yml up --build
```

### 3. Pull Required Ollama Models
Once the Ollama container is healthy:
```bash
# Pull Llama 3.2 reasoning model
docker exec -it wellbore-ollama ollama pull llama3.2:latest

# Pull Nomic Embeddings vector model
docker exec -it wellbore-ollama ollama pull nomic-embed-text
```

### 4. GPU Acceleration (NVIDIA CUDA)
To enable GPU pass-through for Ollama, add the `deploy` reservations block to the `ollama` service in `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

### 5. Check Service Logs
```bash
docker compose logs -f api
docker compose logs -f neo4j
docker compose logs -f ollama
```

### 6. Clean Up Volumes
```bash
docker compose down -v
```
