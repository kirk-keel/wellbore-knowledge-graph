"""Cypher queries and graph traversal builders for Wellbore Knowledge Graph."""

# Document & Chunk Ingestion Queries
MERGE_DOCUMENT = """
MERGE (d:Document {id: $id})
ON CREATE SET 
    d.title = $title,
    d.source = $source,
    d.category = $category,
    d.ingested_at = datetime(),
    d.metadata = $metadata
RETURN d.id as doc_id
"""

MERGE_CHUNK = """
MERGE (c:DocumentChunk {id: $id})
ON CREATE SET 
    c.doc_id = $doc_id,
    c.chunk_index = $chunk_index,
    c.page = $page,
    c.text = $text,
    c.embedding = $embedding,
    c.section_title = $section_title
WITH c
MATCH (d:Document {id: $doc_id})
MERGE (c)-[:EXTRACTED_FROM]->(d)
RETURN c.id as chunk_id
"""

# Domain Entity Upsert Queries
MERGE_WELL = """
MERGE (w:Well {name: $name})
ON CREATE SET 
    w.api_number = $api_number,
    w.field = $field,
    w.water_depth_ft = $water_depth_ft,
    w.total_depth_ft = $total_depth_ft,
    w.operator = $operator
RETURN w.name as name
"""

MERGE_EQUIPMENT = """
MERGE (e:Equipment {serial_number: $serial_number})
ON CREATE SET 
    e.equipment_type = $equipment_type,
    e.manufacturer = $manufacturer,
    e.model = $model,
    e.working_pressure_psi = $working_pressure_psi
RETURN e.serial_number as serial_number
"""

MERGE_FORMATION = """
MERGE (f:Formation {name: $name})
ON CREATE SET 
    f.lithology = $lithology,
    f.top_depth_ft = $top_depth_ft,
    f.bottom_depth_ft = $bottom_depth_ft,
    f.pore_pressure_ppg = $pore_pressure_ppg,
    f.fracture_gradient_ppg = $fracture_gradient_ppg,
    f.hazards = $hazards
RETURN f.name as name
"""

MERGE_FAILURE_MODE = """
MERGE (fm:FailureMode {code: $code})
ON CREATE SET 
    fm.failure_type = $failure_type,
    fm.severity = $severity,
    fm.description = $description,
    fm.root_cause = $root_cause,
    fm.symptoms = $symptoms
RETURN fm.code as code
"""

MERGE_SOP = """
MERGE (s:SOP {sop_id: $sop_id})
ON CREATE SET 
    s.title = $title,
    s.category = $category,
    s.standard_reference = $standard_reference,
    s.required_barriers = $required_barriers,
    s.steps = $steps
RETURN s.sop_id as sop_id
"""

MERGE_INCIDENT = """
MERGE (i:Incident {incident_id: $incident_id})
ON CREATE SET 
    i.well_name = $well_name,
    i.date = $date,
    i.npt_hours = $npt_hours,
    i.depth_ft = $depth_ft,
    i.description = $description,
    i.corrective_actions = $corrective_actions
RETURN i.incident_id as incident_id
"""

# Link Chunk to Entities Mentioned
LINK_CHUNK_TO_ENTITY = """
MATCH (c:DocumentChunk {id: $chunk_id})
CALL {
    WITH c
    MATCH (w:Well {name: $entity_id})
    MERGE (c)-[:MENTIONS]->(w)
    RETURN count(*) as cnt1
}
CALL {
    WITH c
    MATCH (e:Equipment {serial_number: $entity_id})
    MERGE (c)-[:MENTIONS]->(e)
    RETURN count(*) as cnt2
}
CALL {
    WITH c
    MATCH (f:Formation {name: $entity_id})
    MERGE (c)-[:MENTIONS]->(f)
    RETURN count(*) as cnt3
}
CALL {
    WITH c
    MATCH (fm:FailureMode {code: $entity_id})
    MERGE (c)-[:MENTIONS]->(fm)
    RETURN count(*) as cnt4
}
CALL {
    WITH c
    MATCH (s:SOP {sop_id: $entity_id})
    MERGE (c)-[:MENTIONS]->(s)
    RETURN count(*) as cnt5
}
CALL {
    WITH c
    MATCH (i:Incident {incident_id: $entity_id})
    MERGE (c)-[:MENTIONS]->(i)
    RETURN count(*) as cnt6
}
RETURN true as success
"""

# Vector Search Query
VECTOR_SEARCH_CHUNKS = """
CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
YIELD node, score
WHERE score >= $min_score
RETURN 
    node.id AS chunk_id,
    node.doc_id AS doc_id,
    node.page AS page,
    node.text AS text,
    score AS similarity_score
ORDER BY score DESC
"""

# Multi-Hop Neighborhood Expansion from Seed Chunks
EXPAND_CHUNK_NEIGHBORHOOD = """
MATCH (c:DocumentChunk) WHERE c.id IN $chunk_ids
MATCH (c)-[:MENTIONS]->(seed)
OPTIONAL MATCH path = (seed)-[r*1..2]-(target)
WHERE target <> c AND NOT target:Document
WITH seed, r, target
UNWIND relationships(path) as rel
WITH DISTINCT 
    startNode(rel) AS s, 
    type(rel) AS rel_type, 
    endNode(rel) AS t,
    properties(rel) AS rel_props
RETURN 
    labels(s)[0] AS source_label,
    coalesce(s.name, s.serial_number, s.code, s.sop_id, s.incident_id, s.id, 'unknown') AS source_id,
    coalesce(s.title, s.name, s.serial_number, s.code, s.sop_id, s.incident_id, 'unknown') AS source_name,
    rel_type AS relation,
    labels(t)[0] AS target_label,
    coalesce(t.name, t.serial_number, t.code, t.sop_id, t.incident_id, t.id, 'unknown') AS target_id,
    coalesce(t.title, t.name, t.serial_number, t.code, t.sop_id, t.incident_id, 'unknown') AS target_name,
    rel_props AS properties
LIMIT $max_paths
"""

# Domain-Specific Multi-Hop Traversal: Failure Modes & Mitigating SOPs
TRAVERSE_FAILURE_MITIGATION = """
MATCH (fm:FailureMode)
WHERE toLower(fm.failure_type) CONTAINS toLower($keyword) OR toLower(fm.code) CONTAINS toLower($keyword)
OPTIONAL MATCH (fm)-[r1:MITIGATED_BY]->(sop:SOP)
OPTIONAL MATCH (fm)-[r2:OCCURS_IN]->(form:Formation)
OPTIONAL MATCH (inc:Incident)-[r3:EXHIBITED_FAILURE]->(fm)
OPTIONAL MATCH (inc)-[r4:INVOLVED_EQUIPMENT]->(eq:Equipment)
RETURN 
    fm.code AS failure_code,
    fm.failure_type AS failure_type,
    fm.severity AS severity,
    fm.description AS description,
    collect(DISTINCT sop.sop_id) AS mitigating_sops,
    collect(DISTINCT form.name) AS affected_formations,
    collect(DISTINCT inc.incident_id) AS past_incidents,
    collect(DISTINCT eq.serial_number) AS involved_equipment
"""

# Graph Schema Statistics
GET_GRAPH_STATS = """
CALL {
    MATCH (n)
    RETURN labels(n)[0] AS label, count(n) AS node_count
}
RETURN 
    collect({label: label, count: node_count}) AS nodes
"""

GET_RELATIONSHIP_STATS = """
CALL {
    MATCH ()-[r]->()
    RETURN type(r) AS rel_type, count(r) AS rel_count
}
RETURN 
    collect({type: rel_type, count: rel_count}) AS relationships
"""
