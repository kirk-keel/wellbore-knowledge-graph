# REST API Reference

The Wellbore Knowledge Graph exposes a comprehensive FastAPI interface for semantic retrieval, Cypher querying, domain physics simulations, real-time telemetry streaming, and automated multi-agent safety reviews.

**Base URL**: `http://localhost:8000/api/v1`  
**OpenAPI Swagger UI**: `http://localhost:8000/docs`  
**Interactive Visual Dashboard**: `http://localhost:8000/api/v1/dashboard`

---

## Endpoint Summary

| Category | Method | Endpoint | Description |
| :--- | :--- | :--- | :--- |
| **System** | `GET` | `/health` | Healthcheck and connected services readiness |
| **System** | `GET` | `/stats` | Neo4j knowledge graph entity counts and stats |
| **System** | `GET` | `/dashboard` | HTML5/Cytoscape.js interactive graph dashboard |
| **System** | `GET` | `/graph/visual` | Graph nodes & edges payload formatted for Cytoscape |
| **Ingestion** | `POST` | `/ingest/text` | Ingest raw text or Markdown into knowledge graph |
| **Ingestion** | `POST` | `/ingest/file` | Ingest PDF, LAS, or TXT file into knowledge graph |
| **Retrieval** | `POST` | `/query/vector` | Pure semantic vector cosine similarity search |
| **Retrieval** | `POST` | `/query/hybrid` | Hybrid Vector + Multi-Hop Graph Traversal + RRF |
| **Retrieval** | `GET` | `/query/stream` | Server-Sent Events (SSE) streaming hybrid QA |
| **Retrieval** | `POST` | `/query/cypher` | Text-to-Cypher agent with AST safety verification |
| **Intelligence** | `GET` | `/communities` | Global GraphRAG entity communities & risk summaries |
| **Integrity** | `POST` | `/safety/two-barrier` | Two-Barrier rule compliance check (API RP 53 / NORSOK) |
| **Physics** | `POST` | `/physics/casing-integrity` | API Spec 5C3 burst, collapse, and tension verification |
| **Physics** | `POST` | `/physics/hydraulics` | Bingham-Plastic ECD, pressure drops, surge/swab |
| **Physics** | `POST` | `/physics/anti-collision` | Minimum Curvature 3D scan & ISCWSA Separation Factor |
| **Physics** | `POST` | `/physics/vibrations` | Drillstring stick-slip, bit bounce, & whirl classifier |
| **RCFA** | `POST` | `/rcfa/generate` | 5-Whys Root Cause Failure Analysis incident report |
| **Telemetry** | `POST` | `/telemetry/evaluate` | Real-time watchdog evaluating kick/pack-off precursors |
| **Schematic** | `POST` | `/schematic/svg` | 2D wellbore architecture SVG rendering |
| **Multi-Agent** | `POST` | `/agents/committee` | Autonomous multi-agent drilling engineering review |
| **Interoperability**| `POST` | `/export/witsml` | Energistics WITSML 1.4.1.1 XML trajectory exporter |

---

## Detailed Endpoint Examples

### 1. Hybrid Retrieval & QA (`POST /query/hybrid`)
Executes vector similarity search, multi-hop graph expansion, RRF reranking, and grounded answer synthesis.

**Request:**
```json
{
  "question": "What is the primary mitigation for pack-off in the Draupne shale?",
  "top_k": 5,
  "graph_hops": 2,
  "min_similarity": 0.10
}
```

**Response:**
```json
{
  "question": "What is the primary mitigation for pack-off in the Draupne shale?",
  "answer": "When experiencing pack-off in reactive Draupne shale, immediately reduce pump rate to 300 gpm to prevent broaching, work the drillstring with maximum allowable overpull (up to 80 klbf), and spot a high-viscosity lubricating pill while back-reaming.",
  "grounding_score": 0.94,
  "is_grounded": true,
  "chunks": [
    {
      "chunk_id": "sop-draupne-01",
      "text": "SOP-DR-402: Pack-off mitigation in Draupne formation requires dropping pump rate to 300 gpm...",
      "similarity_score": 0.88,
      "rrf_rank": 1
    }
  ],
  "graph_paths": [
    {
      "path": "(Formation: Draupne)-[:PRONE_TO]->(FailureMode: PackOff)-[:MITIGATED_BY]->(SOP: SOP-DR-402)"
    }
  ]
}
```

---

### 2. Bingham-Plastic Hydraulics & ECD Simulator (`POST /physics/hydraulics`)
Simulates downhole rheology, frictional pressure drop, and Equivalent Circulating Density (ECD).

**Request:**
```json
{
  "depth_ft": 10500.0,
  "mud_density_ppg": 11.2,
  "plastic_viscosity_cp": 22.0,
  "yield_point_lbf_100sqft": 16.0,
  "flow_rate_gpm": 550.0,
  "wellbore_diameter_in": 8.5,
  "drillpipe_od_in": 5.0
}
```

**Response:**
```json
{
  "depth_ft": 10500.0,
  "mud_density_ppg": 11.2,
  "annular_velocity_fpm": 273.47,
  "flow_regime": "Laminar",
  "frictional_pressure_loss_psi": 535.12,
  "ecd_ppg": 12.18,
  "surge_margin_ppg": 0.45,
  "swab_margin_ppg": 0.42,
  "is_within_drilling_window": true
}
```

---

### 3. 3D Offset-Well Anti-Collision Scanner (`POST /physics/anti-collision`)
Calculates ISCWSA Minimum Curvature trajectory interpolation and 3D center-to-center Separation Factor (SF).

**Request:**
```json
{
  "active_trajectory": [
    {"md": 0.0, "inc": 0.0, "azi": 0.0},
    {"md": 3000.0, "inc": 15.0, "azi": 45.0},
    {"md": 6000.0, "inc": 35.0, "azi": 50.0}
  ],
  "offset_trajectories": [
    {
      "well_name": "Offset-Well-02",
      "slot_north_ft": 45.0,
      "slot_east_ft": 30.0,
      "trajectory": [
        {"md": 0.0, "inc": 0.0, "azi": 0.0},
        {"md": 3000.0, "inc": 14.5, "azi": 44.0},
        {"md": 6000.0, "inc": 34.0, "azi": 48.0}
      ]
    }
  ],
  "error_uncertainty_ft": 12.0
}
```

**Response:**
```json
{
  "minimum_separation_factor": 1.42,
  "closest_approach_md_ft": 3000.0,
  "closest_offset_well": "Offset-Well-02",
  "center_to_center_distance_ft": 43.8,
  "collision_risk_level": "WARNING",
  "anti_collision_clearance_passed": true
}
```

---

### 4. Energistics WITSML 1.4.1.1 XML Exporter (`POST /export/witsml`)
Converts wellbore trajectory data into compliant WITSML 1.4.1.1 XML documents.

**Request:**
```json
{
  "well_name": "Well-31-A",
  "wellbore_name": "Wellbore-01-OH",
  "trajectory_stations": [
    {"md": 0.0, "tvd": 0.0, "inc": 0.0, "azi": 0.0},
    {"md": 2500.0, "tvd": 2480.0, "inc": 12.4, "azi": 42.1}
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "version": "1.4.1.1",
  "witsml_xml": "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<trajectorys xmlns=\"http://www.witsml.org/schemas/1series\" version=\"1.4.1.1\">..."
}
```
