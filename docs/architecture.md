# System Architecture

The **Wellbore Engineering Knowledge Graph (GraphRAG)** integrates structural graph topologies with high-dimensional vector embeddings, generative LLM reasoning, deterministic engineering physics simulators, and safety guardrails.

## High-Level Architecture Diagram

```mermaid
flowchart TD
    subgraph Ingestion ["Ingestion Subsystem"]
        A1["Drilling PDFs & Manuals"] --> B1["PDF Parser & Structure Extractor"]
        A2["CWLS LAS 2.0 Logs"] --> B2["LAS Depth & Curve Parser"]
        A3["IADC Daily Drilling Reports"] --> B3["DDR Sequential Timeline Builder"]
        A4["Rig Telemetry Data"] --> B4["Telemetry Stream Watchdog"]

        B1 & B2 & B3 --> C["Hierarchical Markdown Chunker"]
        C --> D["Entity & Relationship Extractor"]
        D --> E["Embedding Provider (Nomic / Deterministic)"]
    end

    subgraph KnowledgeGraph ["Knowledge Store (Neo4j 5.20)"]
        E --> F1[("Neo4j Graph Database")]
        F1 --> F2["Vector Index (384-dim Cosine)"]
        F1 --> F3["Entity Constraints & Indexes"]
        F1 --> F4["Sequential DDR Temporal Chains"]
        F1 --> F5["Ontology: Wells, BHA, Formations, Incidents"]
    end

    subgraph RetrievalEngine ["Hybrid GraphRAG Engine"]
        Q["User / Rig Query"] --> G1["Vector Cosine Search (Top-K)"]
        Q --> G2["Multi-Hop Cypher Traversal (1-3 Hops)"]
        Q --> G3["Text-to-Cypher Generator (AST Validated)"]

        G1 & G2 --> H["Reciprocal Rank Fusion (RRF) Reranker"]
        H --> I["Two-Barrier Safety Guardrail (API RP 53 / NORSOK D-010)"]
        I --> J["Grounding QA Engine & Anti-Hallucination Verifier"]
    end

    subgraph PhysicsAndAgents ["Engineering Simulators & Multi-Agent Committee"]
        K1["Bingham-Plastic Hydraulics & ECD Simulator"]
        K2["3D Offset Anti-Collision Scanner (ISCWSA)"]
        K3["Drillstring Vibration Dysfunction Classifier"]
        K4["Casing Integrity & MASP Calculator (API Spec 5C3)"]
        K5["5-Whys RCFA Incident Post-Mortem Generator"]
        K6["Autonomous Multi-Agent Drilling Committee"]
        K7["Energistics WITSML 1.4.1.1 XML Exporter"]
    end

    subgraph Presentation ["Presentation & API Layer (FastAPI)"]
        J & PhysicsAndAgents --> API["FastAPI REST Endpoints (17 Routes)"]
        API --> L1["Interactive Cytoscape.js Knowledge Graph Dashboard"]
        API --> L2["Server-Sent Events (SSE) Query Stream"]
        API --> L3["OpenAPI Swagger UI Documentation"]
    end
```

---

## Subsystem Details

### 1. Ingestion Pipeline
- **Hierarchical Document Parsing**: Extracts section headers, tables, bullet points, equipment serial numbers, lithology descriptions, and SOP instructions.
- **CWLS LAS 2.0 Parsing**: Reads well log headers (~WELL, ~CURVE, ~PARAMETER, ~ASCII) and detects log anomalies (e.g., severe washouts where $CALI > 14''$, or hydrocarbon gas shows with $RES > 50\,\Omega\cdot\text{m}$ and $GR < 40\,\text{API}$).
- **DDR Timeline Chaining**: Connects daily drilling reports chronologically via `(:DDR)-[:NEXT_REPORT]->(:DDR)` to allow sequential querying of NPT, mud weight changes, and drilling problems over time.

### 2. Neo4j Knowledge Graph Schema
- **Nodes**: `Well`, `Wellbore`, `BHA`, `BHAComponent`, `Formation`, `Equipment`, `Incident`, `SOP`, `DailyDrillingReport`, `WellLogCurve`, `Chunk`.
- **Relationships**: `LOCATED_IN`, `HAS_BHA`, `CONTAINS_COMPONENT`, `PENETRATES`, `HAS_INCIDENT`, `INVOLVED_EQUIPMENT`, `MITIGATED_BY`, `HAS_SOP`, `RECORDED_DDR`, `NEXT_REPORT`, `HAS_LOG_CURVE`, `EXTRACTED_FROM`.
- **Vector Index**: 384-dimensional cosine index on `Chunk(embedding)`.

### 3. Hybrid Retrieval & RRF Ranking
Queries leverage two complementary retrieval pathways:
1. **Vector Cosine Similarity**: Scans indexed semantic document chunks for contextual nuance.
2. **Multi-Hop Cypher Traversal**: Traverses entity paths up to $N$ hops to gather connected domain relationships (e.g., equipment $\to$ failure mode $\to$ mitigation SOP).
3. **Reciprocal Rank Fusion (RRF)**: Merges both sets of results:
   $$RRF(d) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$$
   where $k=60$, vector weight $w_v = 0.6$, and graph weight $w_g = 0.4$.

### 4. Safety Guardrails & Anti-Hallucination
- **Two-Barrier Rule Enforcement**: Strict compliance checks against NORSOK D-010 and API RP 53. If a proposed operational intervention leaves fewer than 2 independent, verified well barriers in place (e.g., only one casing/fluid column or compromised BOP ram), execution is intercepted and critical warnings are flagged.
- **Grounding Verification**: The QA engine calculates the lexical and semantic overlap between the LLM generation and the retrieved context chunks, annotating any ungrounded assertions.
- **AST Cypher Validator**: Inspects generated Cypher queries against an abstract syntax tree allowlist to reject mutations (`DELETE`, `DROP`, `CREATE`, `SET`, `MERGE`, `CALL`) before execution.

### 5. Deterministic Engineering Physics Engines
- **Hydraulics & ECD Engine**: Bingham-Plastic flow model computing annular velocities, frictional pressure losses, Equivalent Circulating Density, and surge/swab margins.
- **3D Offset Anti-Collision Scanner**: Minimum Curvature algorithm for 3D trajectory interpolation ($N, E, TVD$), combined with ISCWSA ellipsoid error modeling and Separation Factor ($SF = \frac{CC}{S_1 + S_2}$) warnings.
- **Drillstring Vibration Dysfunction Classifier**: Surface Stick-Slip Index (SSSI %) classification, bit bounce axial vibration detection, and backward whirl centrifugal acceleration estimation.
- **Casing Mechanics & MASP Calculator**: API Spec 5C3 burst (Barlow), collapse (plastic/yield), and axial tension calculations with Maximum Allowable Surface Pressure (MASP) safety factors.
- **WITSML 1.4.1.1 XML Exporter**: Standardized Energistics-compliant well, wellbore, and trajectory export for integration with Halliburton Landmark, Schlumberger Petrel, and Baker Hughes JewelSuite.
