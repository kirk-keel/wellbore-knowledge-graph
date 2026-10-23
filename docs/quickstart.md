# Quickstart Guide

Get up and running with the **Wellbore Engineering Knowledge Graph** in under 5 minutes.

---

## 1. Local Environment Setup

### Prerequisites
- Python 3.10+ (or 3.11, 3.12, 3.14)
- Git
- Docker & Docker Compose (optional for full microservices)

### Clone and Install
```bash
# Clone the repository
git clone https://github.com/kirk-keel/wellbore-knowledge-graph.git
cd wellbore-knowledge-graph

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Environment Configuration

Copy the example environment file:
```bash
cp .env.example .env
```

If running standalone without Docker, the application automatically runs in **resilient mode** using an in-memory graph buffer and deterministic token hash embeddings—no external database setup is strictly required to test features!

---

## 3. Seed Database & Sample Data

Generate sample synthetic drilling engineering documents, LAS well logs, and DDR timelines:
```bash
python scripts/generate_sample_pdf.py
python scripts/seed_database.py
```

---

## 4. Run the API Server

Launch the FastAPI application with Uvicorn:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 5. Explore the Application

Open your browser:
- **Interactive Visual Graph Dashboard**: [http://localhost:8000/api/v1/dashboard](http://localhost:8000/api/v1/dashboard)
- **Interactive OpenAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Healthcheck & Status**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## 6. Run Test Suite

Verify all unit and integration tests pass cleanly:
```bash
pytest -v
```
All 41 tests validate ingestion, RRF hybrid retrieval, safe AST Cypher generation, physics simulators (Bingham-Plastic, ISCWSA anti-collision, casing mechanics), and WITSML export.
