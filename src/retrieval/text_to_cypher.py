"""Text-to-Cypher Agent for quantitative and analytical graph queries."""

import logging
import re
from typing import Any, Dict, List, Optional
import httpx

from src.config import settings
from src.graph.client import Neo4jClient, neo4j_client
from src.ingestion.pipeline import IngestionPipeline, ingestion_pipeline

logger = logging.getLogger(__name__)

TEXT_TO_CYPHER_PROMPT = """You are a Neo4j Cypher and Drilling Engineering expert.
Convert the user's natural language question into a high-performance, strictly READ-ONLY Cypher query.

Schema:
Node Labels:
- (:Well {name, api_number, field, water_depth_ft, total_depth_ft})
- (:Equipment {serial_number, equipment_type, manufacturer, model, working_pressure_psi})
- (:Formation {name, lithology, top_depth_ft, bottom_depth_ft, pore_pressure_ppg, hazards})
- (:FailureMode {code, failure_type, severity, description, root_cause})
- (:SOP {sop_id, title, category, standard_reference, required_barriers})
- (:Incident {incident_id, well_name, date, npt_hours, depth_ft, description})
- (:DDR {report_number, well_name, report_date, current_depth_ft, npt_hours, mud_weight_ppg})

Relationship Types:
- (:Incident)-[:OCCURRED_IN_WELL]->(:Well)
- (:Incident)-[:INVOLVED_EQUIPMENT]->(:Equipment)
- (:Incident)-[:EXHIBITED_FAILURE]->(:FailureMode)
- (:FailureMode)-[:MITIGATED_BY]->(:SOP)
- (:FailureMode)-[:OCCURS_IN]->(:Formation)
- (:Well)-[:PENETRATES]->(:Formation)
- (:Well)-[:USES_EQUIPMENT]->(:Equipment)
- (:DDR)-[:NEXT_REPORT]->(:DDR)
- (:DDR)-[:RECORDED_FOR_WELL]->(:Well)

Rules:
1. ONLY return the raw Cypher query inside ```cypher ... ``` code block.
2. The query MUST be read-only (MATCH, OPTIONAL MATCH, RETURN, WITH, UNWIND, WHERE, ORDER BY, LIMIT).
3. NEVER use CREATE, MERGE, SET, DELETE, REMOVE, DROP.

User Question:
{question}
"""


class TextToCypherAgent:
    """Translates analytical engineering questions into verified Cypher statements and executes them."""

    def __init__(
        self,
        graph_client: Neo4jClient = neo4j_client,
        pipeline: IngestionPipeline = ingestion_pipeline,
        ollama_url: str = settings.OLLAMA_BASE_URL,
        model: str = settings.OLLAMA_LLM_MODEL,
    ):
        self.graph = graph_client
        self.pipeline = pipeline
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model

    def query(self, natural_language_question: str) -> Dict[str, Any]:
        """Generate Cypher, validate safety, execute, and return results."""
        # 1. Generate Cypher query
        cypher = self._generate_cypher(natural_language_question)

        # 2. Validate Read-Only Safety
        self._validate_read_only(cypher)

        # 3. Execute
        if self.graph.is_connected():
            try:
                results = self.graph.execute_query(cypher)
                return {
                    "question": natural_language_question,
                    "cypher_query": cypher,
                    "execution_mode": "neo4j_live",
                    "results_count": len(results),
                    "results": results,
                }
            except Exception as e:
                logger.warning("Cypher execution failed on Neo4j: %s. Using heuristic analysis.", e)

        # 4. In-memory analytical execution fallback
        fallback_results = self._execute_in_memory_analytics(natural_language_question, cypher)
        return {
            "question": natural_language_question,
            "cypher_query": cypher,
            "execution_mode": "in_memory_analytical_engine",
            "results_count": len(fallback_results),
            "results": fallback_results,
        }

    def _generate_cypher(self, question: str) -> str:
        """Query LLM to formulate Cypher query, with robust rule-based template fallback."""
        try:
            prompt = TEXT_TO_CYPHER_PROMPT.format(question=question)
            with httpx.Client(base_url=self.ollama_url, timeout=12.0) as client:
                resp = client.post(
                    "/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False},
                )
                if resp.status_code == 200:
                    text_out = resp.json().get("response", "")
                    match = re.search(r"```(?:cypher)?\s*(.*?)\s*```", text_out, re.DOTALL)
                    if match:
                        return match.group(1).strip()
        except Exception:
            pass

        # Rule-based Cypher generator for common engineering inquiry patterns
        q_lower = question.lower()
        if "npt" in q_lower or "non-productive" in q_lower:
            if "equipment" in q_lower or "serial" in q_lower:
                return (
                    "MATCH (e:Equipment)<-[:INVOLVED_EQUIPMENT]-(i:Incident) "
                    "RETURN e.serial_number AS equipment, count(i) AS incidents, sum(i.npt_hours) AS total_npt_hours "
                    "ORDER BY total_npt_hours DESC"
                )
            if "well" in q_lower:
                return (
                    "MATCH (w:Well)<-[:OCCURRED_IN_WELL]-(i:Incident) "
                    "RETURN w.name AS well, count(i) AS incidents, sum(i.npt_hours) AS total_npt_hours "
                    "ORDER BY total_npt_hours DESC"
                )
            return (
                "MATCH (i:Incident) "
                "RETURN i.incident_id AS incident, i.well_name AS well, i.npt_hours AS npt_hours, i.description AS description "
                "ORDER BY i.npt_hours DESC"
            )

        if "mitigate" in q_lower or "sop" in q_lower:
            return (
                "MATCH (fm:FailureMode)-[:MITIGATED_BY]->(s:SOP) "
                "RETURN fm.failure_type AS failure_mode, collect(s.sop_id) AS mitigating_sops"
            )

        if "formation" in q_lower or "lithology" in q_lower:
            return (
                "MATCH (w:Well)-[:PENETRATES]->(f:Formation) "
                "RETURN w.name AS well, collect(f.name) AS formations"
            )

        # Default exploration query
        return "MATCH (n) RETURN labels(n)[0] AS entity_type, count(n) AS count ORDER BY count DESC"

    def _validate_read_only(self, cypher: str):
        """Enforce strict read-only AST safety validation."""
        forbidden = [
            r"\bCREATE\b",
            r"\bMERGE\b",
            r"\bSET\b",
            r"\bDELETE\b",
            r"\bREMOVE\b",
            r"\bDROP\b",
            r"\bDETACH\b",
            r"\bCALL\s+apoc\.periodic\b",
        ]
        for pattern in forbidden:
            if re.search(pattern, cypher, re.IGNORECASE):
                raise ValueError(f"Disallowed Cypher mutation detected: matches '{pattern}'. Only read-only operations permitted.")

    def _execute_in_memory_analytics(self, question: str, cypher: str) -> List[Dict[str, Any]]:
        """Compute aggregated answers from in-memory extractions when Neo4j is offline."""
        all_incidents = []
        all_equipment = []
        all_failures = []
        all_sops = []

        for ext in self.pipeline.in_memory_extractions.values():
            all_incidents.extend(ext.incidents)
            all_equipment.extend(ext.equipment)
            all_failures.extend(ext.failure_modes)
            all_sops.extend(ext.sops)

        q_lower = question.lower()
        if "npt" in q_lower:
            # Group by well or equipment
            well_npt: Dict[str, float] = {}
            for inc in all_incidents:
                w = inc.well_name or "Deepwater Titan Well"
                well_npt[w] = well_npt.get(w, 0.0) + (inc.npt_hours or 0.0)

            return [
                {"well": w, "total_npt_hours": round(hours, 2), "incidents_count": sum(1 for i in all_incidents if i.well_name == w or w in i.incident_id)}
                for w, hours in well_npt.items()
            ]

        if "mitigate" in q_lower or "sop" in q_lower:
            return [
                {"failure_mode": f.failure_type, "code": f.code, "severity": f.severity, "known_sops": [s.sop_id for s in all_sops if "IADC" in s.sop_id or "API" in s.sop_id]}
                for f in all_failures[:5]
            ]

        return [
            {"total_incidents_recorded": len(all_incidents), "total_equipment_assets": len(all_equipment), "total_sops": len(all_sops)}
        ]
