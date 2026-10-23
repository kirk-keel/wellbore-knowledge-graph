"""Neo4j graph schema constraints and index definitions."""

from typing import List
from src.config import settings

# Uniqueness Constraints
CONSTRAINT_STATEMENTS: List[str] = [
    # Document and Chunks
    "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:DocumentChunk) REQUIRE c.id IS UNIQUE",

    # Domain Entities
    "CREATE CONSTRAINT well_name_unique IF NOT EXISTS FOR (w:Well) REQUIRE w.name IS UNIQUE",
    "CREATE CONSTRAINT equipment_serial_unique IF NOT EXISTS FOR (e:Equipment) REQUIRE e.serial_number IS UNIQUE",
    "CREATE CONSTRAINT formation_name_unique IF NOT EXISTS FOR (f:Formation) REQUIRE f.name IS UNIQUE",
    "CREATE CONSTRAINT failure_code_unique IF NOT EXISTS FOR (fm:FailureMode) REQUIRE fm.code IS UNIQUE",
    "CREATE CONSTRAINT sop_id_unique IF NOT EXISTS FOR (s:SOP) REQUIRE s.sop_id IS UNIQUE",
    "CREATE CONSTRAINT incident_id_unique IF NOT EXISTS FOR (i:Incident) REQUIRE i.incident_id IS UNIQUE",
]

# Standard Property Indexes for high-throughput lookup
INDEX_STATEMENTS: List[str] = [
    "CREATE INDEX equipment_type_idx IF NOT EXISTS FOR (e:Equipment) ON (e.equipment_type)",
    "CREATE INDEX formation_lithology_idx IF NOT EXISTS FOR (f:Formation) ON (f.lithology)",
    "CREATE INDEX failure_type_idx IF NOT EXISTS FOR (fm:FailureMode) ON (fm.failure_type)",
    "CREATE INDEX sop_category_idx IF NOT EXISTS FOR (s:SOP) ON (s.category)",
    "CREATE INDEX chunk_doc_idx IF NOT EXISTS FOR (c:DocumentChunk) ON (c.doc_id)",
]

# Neo4j 5+ Vector Index Definition
VECTOR_INDEX_STATEMENT: str = f"""
CREATE VECTOR INDEX {settings.VECTOR_INDEX_NAME} IF NOT EXISTS
FOR (c:DocumentChunk) ON (c.embedding)
OPTIONS {{
  indexConfig: {{
    `vector.dimensions`: {settings.EMBEDDING_DIMENSION},
    `vector.similarity_function`: '{settings.SIMILARITY_METRIC}'
  }}
}}
"""
