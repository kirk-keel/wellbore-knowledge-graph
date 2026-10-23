# Production Deployment Guide

This guide covers deployment architectures for the **Wellbore Knowledge Graph** system across development, single-node Docker Compose, and enterprise Kubernetes production environments.

---

## 1. Prerequisites & Resource Allocation

### Minimum System Requirements
- **CPU**: 4 Cores (x86_64 or ARM64)
- **RAM**: 8 GB RAM (16 GB recommended when hosting local 7B/8B LLM weights)
- **Storage**: 50 GB SSD
- **GPU (Optional)**: NVIDIA GPU with CUDA 12+ (8GB+ VRAM) for real-time Ollama acceleration

---

## 2. Docker Compose Deployment (Recommended for Quick Production)

The fastest way to deploy the complete stack (Neo4j with APOC, Ollama LLM provider, and FastAPI GraphRAG) is via Docker Compose:

```bash
# Clone the repository
git clone https://github.com/kirk-keel/wellbore-knowledge-graph.git
cd wellbore-knowledge-graph

# Configure environment variables
cp .env.example .env

# Start all microservices in detached mode
docker compose up -d --build
```

### Pull Required Neural Network Weights
```bash
# Pull Llama 3.2 instruction-tuned model
docker exec -it wellbore-ollama ollama pull llama3.2:latest

# Pull Nomic Embeddings vector model
docker exec -it wellbore-ollama ollama pull nomic-embed-text
```

### Seed Initial Knowledge Base
```bash
docker exec -it wellbore-graphrag-api python scripts/seed_database.py
```

---

## 3. GPU Acceleration for Ollama (CUDA)

For high-throughput LLM answer synthesis, update `docker-compose.yml` to allocate GPU hardware to Ollama:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: wellbore-ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Verify GPU allocation inside the container:
```bash
docker exec -it wellbore-ollama nvidia-smi
```

---

## 4. Neo4j Performance Tuning

For production databases handling millions of nodes and high-frequency vector searches:

```ini
# Add to neo4j configuration (configs/neo4j.conf)
server.memory.heap.initial_size=4G
server.memory.heap.max_size=8G
server.memory.pagecache.size=4G

# Vector Index Configuration
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*
```

---

## 5. Kubernetes Deployment Manifest

Below is a standard Kubernetes manifest (`k8s-deployment.yaml`) for deploying the FastAPI GraphRAG application:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wellbore-graphrag-api
  labels:
    app: wellbore-graphrag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: wellbore-graphrag
  template:
    metadata:
      labels:
        app: wellbore-graphrag
    spec:
      containers:
      - name: api
        image: wellbore-knowledge-graph:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: wellbore-config
        - secretRef:
            name: wellbore-secrets
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 20
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: wellbore-service
spec:
  type: ClusterIP
  selector:
    app: wellbore-graphrag
  ports:
  - port: 80
    targetPort: 8000
```

---

## 6. Resilient Standalone / Offline Mode

The application contains built-in zero-dependency fallbacks:
- If Neo4j is offline or unreachable, the system automatically falls back to an in-memory graph buffer.
- If Ollama is offline or unreachable, the system activates a deterministic normalized token embedding provider and rule-based Cypher generator.

This guarantees uninterrupted service during network partitions, container reboots, or isolated edge rig deployments.
