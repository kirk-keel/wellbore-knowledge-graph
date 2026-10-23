"""Entity and relation extraction engine for wellbore engineering documents using LangChain and domain rules."""

import json
import logging
import re
from typing import Dict, List, Optional
import httpx
from langchain_core.prompts import PromptTemplate

from src.config import settings
from src.models.domain import (
    EquipmentEntity,
    FailureModeEntity,
    FormationEntity,
    IncidentEntity,
    Relationship,
    SOPEntity,
    WellEntity,
    WellboreExtractionResult,
)

logger = logging.getLogger(__name__)

SYSTEM_EXTRACTION_PROMPT = """You are an expert Petroleum, Drilling, and Subsurface Engineer extracting structured knowledge from technical documents.
Extract all instances of:
1. Wells (name, api_number, field, water_depth_ft, total_depth_ft, operator)
2. Equipment (serial_number, equipment_type, manufacturer, model, working_pressure_psi)
3. Formations & Lithology (name, lithology, top_depth_ft, bottom_depth_ft, pore_pressure_ppg, hazards)
4. Failure Modes (code, failure_type, severity, description, root_cause, symptoms)
5. SOPs & Standards (sop_id, title, category, standard_reference, required_barriers, steps)
6. Incidents (incident_id, well_name, date, npt_hours, depth_ft, description, corrective_actions)
7. Semantic Relationships between the above entities (e.g. EXHIBITED_FAILURE, MITIGATED_BY, PENETRATES, USES_EQUIPMENT, INVOLVED_EQUIPMENT).

Format your response as valid JSON strictly adhering to the schema:
{{
  "wells": [...],
  "equipment": [...],
  "formations": [...],
  "failure_modes": [...],
  "sops": [...],
  "incidents": [...],
  "relationships": [
    {{"source_type": "...", "source_id": "...", "relation_type": "...", "target_type": "...", "target_id": "..."}}
  ]
}}

Document Text:
{text}
"""


class WellboreEntityExtractor:
    """Extracts domain entities and relationships from engineering text chunks."""

    def __init__(self, ollama_url: str = settings.OLLAMA_BASE_URL, model: str = settings.OLLAMA_LLM_MODEL):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.prompt_template = PromptTemplate(
            input_variables=["text"],
            template=SYSTEM_EXTRACTION_PROMPT,
        )

    def extract_from_text(self, text: str) -> WellboreExtractionResult:
        """Extract entities and relations using LLM with deterministic regex fallback."""
        llm_result: Optional[WellboreExtractionResult] = None

        # Attempt LLM extraction if Ollama is accessible
        try:
            llm_result = self._extract_with_ollama(text)
        except Exception as e:
            logger.debug("LLM extraction skipped or unavailable (%s). Using specialized domain heuristics.", e)

        # Apply deterministic rule-based extractor
        rule_result = self._extract_with_rules(text)

        if llm_result:
            return self._merge_results(llm_result, rule_result)
        return rule_result

    def _extract_with_ollama(self, text: str) -> Optional[WellboreExtractionResult]:
        """Call Ollama chat/generate API with JSON format instruction."""
        prompt = self.prompt_template.format(text=text[:3500])
        with httpx.Client(base_url=self.ollama_url, timeout=12.0) as client:
            resp = client.post(
                "/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
            )
            if resp.status_code == 200:
                raw_json = resp.json().get("response", "{}")
                data = json.loads(raw_json)
                return WellboreExtractionResult(**data)
        return None

    def _extract_with_rules(self, text: str) -> WellboreExtractionResult:
        """
        Specialized domain parser for drilling engineering terminology,
        API/IADC standards, equipment serial numbers, lithology tops, and failure modes.
        """
        wells: Dict[str, WellEntity] = {}
        equipment: Dict[str, EquipmentEntity] = {}
        formations: Dict[str, FormationEntity] = {}
        failure_modes: Dict[str, FailureModeEntity] = {}
        sops: Dict[str, SOPEntity] = {}
        incidents: Dict[str, IncidentEntity] = {}
        relationships: List[Relationship] = []

        # --- 1. Wells Extraction ---
        well_patterns = [
            r"(?:Well\s+|Wellbore\s+)([A-Z0-9]+(?:-[A-Z0-9]+)+)",
            r"(Deepwater\s+Titan(?:\s+Well)?)",
            r"(Titan-\d+)",
            r"Well\s+([0-9]{2,4}-[A-Z0-9]+)",
        ]
        for pattern in well_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                w_name = match.group(1).strip()
                if w_name and w_name not in wells:
                    # Check for depth
                    depth_match = re.search(rf"{re.escape(w_name)}.*?(\d{{1,2}},\d{{3}}|\d{{4,5}})\s*(?:ft|feet|m)", text, re.IGNORECASE)
                    td = float(depth_match.group(1).replace(",", "")) if depth_match else None
                    wells[w_name] = WellEntity(
                        name=w_name,
                        field="North Sea / Offshore" if "sea" in text.lower() or "offshore" in text.lower() else "Onshore Basin",
                        total_depth_ft=td,
                    )

        # --- 2. Equipment & Serial Numbers Extraction ---
        # Matches formats like: BOP-HYD-5000-08, MM-775-A1, MWD-TENSOR-882, CSG-958-L80, SN: 98421-BOP
        eq_patterns = [
            r"\b(BOP-[A-Z0-9-]+)\b",
            r"\b(MM-[A-Z0-9-]+)\b",
            r"\b(MWD-[A-Z0-9-]+)\b",
            r"\b(RSS-[A-Z0-9-]+)\b",
            r"\b(CSG-[A-Z0-9-]+)\b",
            r"\b(PUMP-[A-Z0-9-]+)\b",
            r"Serial\s*(?:Number|No\.?|#):?\s*([A-Z0-9-]+)",
            r"S/N:?\s*([A-Z0-9-]+)",
        ]
        for pattern in eq_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                serial = match.group(1).strip()
                if len(serial) >= 4 and serial not in equipment:
                    eq_type = "Drilling Equipment"
                    if "BOP" in serial.upper() or "ram" in text.lower():
                        eq_type = "Blowout Preventer (BOP) Stack"
                    elif "MM" in serial.upper() or "motor" in text.lower():
                        eq_type = "Positive Displacement Mud Motor"
                    elif "MWD" in serial.upper():
                        eq_type = "Measurement While Drilling (MWD) Tool"
                    elif "CSG" in serial.upper():
                        eq_type = "Casing String"
                    elif "RSS" in serial.upper():
                        eq_type = "Rotary Steerable System"

                    # Rating detection (e.g. 15,000 psi, 10k psi, 5000 psi)
                    press_match = re.search(r"(\d{1,2}(?:,\d{3})?|\d{4,5})\s*(?:psi|PSI)", text)
                    press = float(press_match.group(1).replace(",", "")) if press_match else None

                    equipment[serial] = EquipmentEntity(
                        serial_number=serial,
                        equipment_type=eq_type,
                        working_pressure_psi=press,
                    )

        # --- 3. Formation & Lithology Extraction ---
        formation_vocab = [
            ("Eagle Ford", "Organic-rich Shale"),
            ("Brent", "Deltaic Sandstone"),
            ("Norphlet", "Aeolian Sandstone"),
            ("Austin Chalk", "Chalk / Limestone"),
            ("Haynesville", "Overpressured Shale"),
            ("Wilcox", "Deepwater Sandstone"),
            ("Heimdal", "Porous Sandstone"),
            ("Draupne", "Source Rock Shale"),
            ("Permian Basin Wolfcamp", "Unconventional Shale"),
        ]
        for name, default_lith in formation_vocab:
            if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
                # Look for depth range e.g. "at 10,250 ft", "10,250 - 11,800 ft"
                depth_match = re.search(rf"{re.escape(name)}.*?(?:top|at|depth)?\s*[:\s]?\s*(\d{{1,2}},\d{{3}}|\d{{4,5}})\s*(?:ft|feet)?", text, re.IGNORECASE)
                top_depth = float(depth_match.group(1).replace(",", "")) if depth_match else None

                hazards = []
                if re.search(r"overpressure|gas kick|influx", text, re.IGNORECASE):
                    hazards.append("Overpressured Gas")
                if re.search(r"lost circulation|loss zone|fracture", text, re.IGNORECASE):
                    hazards.append("Lost Circulation")
                if re.search(r"swelling|reactive|sloughing shale", text, re.IGNORECASE):
                    hazards.append("Reactive Sloughing Shale")

                form_key = f"{name} Formation"
                formations[form_key] = FormationEntity(
                    name=form_key,
                    lithology=default_lith,
                    top_depth_ft=top_depth,
                    hazards=hazards,
                )

        # Generic formation match if not in list
        generic_form_matches = re.finditer(r"([A-Z][a-zA-Z0-9\s]{2,20})\s+(Formation|Shale|Sandstone|Carbonate|Limestone)", text)
        for match in generic_form_matches:
            f_full = f"{match.group(1).strip()} {match.group(2).strip()}"
            if f_full not in formations and len(f_full) < 35:
                formations[f_full] = FormationEntity(
                    name=f_full,
                    lithology=match.group(2).strip(),
                )

        # --- 4. Failure Modes Extraction ---
        failure_vocab = [
            ("Pack-off", "FAIL-PACKOFF", "Critical", "Annular packing off around drill collars leading to sudden SPP spike and loss of rotation"),
            ("Twist-off", "FAIL-TWISTOFF", "Critical", "Parting of drillstring caused by excessive torsional stress or fatigue"),
            ("Washout", "FAIL-WASHOUT", "High", "Fluid erosion creating a hole in drill pipe or tool joints"),
            ("Ram Leak", "FAIL-RAMLEAK", "Critical", "Hydraulic seal breach in BOP pipe or blind shear ram"),
            ("Gas Kick", "FAIL-KICK", "Critical", "Uncontrolled influx of formation hydrocarbons into the wellbore"),
            ("Differential Sticking", "FAIL-DIFFSTICK", "High", "Drillstring stuck against permeable formation due to overbalanced hydrostatic pressure"),
            ("Lost Circulation", "FAIL-LOSTCIRC", "High", "Partial or total loss of drilling mud into fractured formation"),
        ]
        for f_term, code, sev, desc in failure_vocab:
            if re.search(rf"\b{re.escape(f_term)}\b", text, re.IGNORECASE):
                failure_modes[code] = FailureModeEntity(
                    code=code,
                    failure_type=f_term,
                    severity=sev,
                    description=desc,
                )

        # --- 5. SOPs and Standards Extraction ---
        sop_patterns = [
            r"\b(IADC-[A-Z0-9-]+)\b",
            r"\b(API-RP-[A-Z0-9-]+)\b",
            r"\b(SOP-[A-Z0-9-]+)\b",
            r"\b(API-SPEC-[A-Z0-9-]+)\b",
        ]
        for pattern in sop_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                sop_code = match.group(1).strip()
                if sop_code not in sops:
                    title = f"Operational Standard {sop_code}"
                    if "IADC" in sop_code:
                        title = f"IADC Standard Guideline {sop_code}"
                    elif "API" in sop_code:
                        title = f"API Recommended Practice {sop_code}"

                    sops[sop_code] = SOPEntity(
                        sop_id=sop_code,
                        title=title,
                        category="Well Control" if "WCC" in sop_code or "SHUTIN" in sop_code else "Drilling SOP",
                        standard_reference=sop_code,
                    )

        # Look for named procedures e.g. "Soft Shut-In Procedure", "Tripping In Procedure"
        named_sops = re.finditer(r"((?:Soft|Hard)\s+Shut-In\s+Procedure|Tripping\s+(?:In|Out)\s+Procedure|Drillstring\s+Packoff\s+Remediation)", text, re.IGNORECASE)
        for match in named_sops:
            s_title = match.group(1).strip()
            sop_id = "SOP-" + re.sub(r"[^A-Za-z0-9]", "-", s_title.upper())[:20]
            if sop_id not in sops:
                sops[sop_id] = SOPEntity(
                    sop_id=sop_id,
                    title=s_title,
                    category="Well Control" if "Shut-In" in s_title else "Drilling Operations",
                )

        # --- 6. Incident Extraction ---
        inc_matches = re.finditer(r"(?:Incident|INC|NPT)[\s-]*#?:?\s*([A-Z0-9-]+)", text, re.IGNORECASE)
        for match in inc_matches:
            inc_code = match.group(1).strip()
            if len(inc_code) >= 3 and inc_code not in incidents:
                # Look for NPT hours
                npt_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours|hrs)\s*(?:of\s+)?NPT", text, re.IGNORECASE)
                npt_val = float(npt_match.group(1)) if npt_match else None

                incidents[inc_code] = IncidentEntity(
                    incident_id=f"INC-{inc_code}",
                    description=text[:200].replace("\n", " "),
                    npt_hours=npt_val,
                )

        # --- 7. Relationships Construction ---
        # Connect Incidents to Wells
        for inc_id, inc_obj in incidents.items():
            for w_name in wells:
                relationships.append(Relationship(
                    source_type="Incident",
                    source_id=inc_obj.incident_id,
                    relation_type="OCCURRED_IN_WELL",
                    target_type="Well",
                    target_id=w_name,
                ))
            # Connect Incidents to Failure Modes
            for f_code in failure_modes:
                relationships.append(Relationship(
                    source_type="Incident",
                    source_id=inc_obj.incident_id,
                    relation_type="EXHIBITED_FAILURE",
                    target_type="FailureMode",
                    target_id=f_code,
                ))
            # Connect Incidents to Equipment
            for eq_serial in equipment:
                relationships.append(Relationship(
                    source_type="Incident",
                    source_id=inc_obj.incident_id,
                    relation_type="INVOLVED_EQUIPMENT",
                    target_type="Equipment",
                    target_id=eq_serial,
                ))

        # Connect Wells to Formations and Equipment
        for w_name in wells:
            for f_name in formations:
                relationships.append(Relationship(
                    source_type="Well",
                    source_id=w_name,
                    relation_type="PENETRATES",
                    target_type="Formation",
                    target_id=f_name,
                ))
            for eq_serial in equipment:
                relationships.append(Relationship(
                    source_type="Well",
                    source_id=w_name,
                    relation_type="USES_EQUIPMENT",
                    target_type="Equipment",
                    target_id=eq_serial,
                ))

        # Connect Failure Modes to Mitigating SOPs
        for f_code, f_obj in failure_modes.items():
            for s_id in sops:
                # Match mitigation logic
                if (f_obj.failure_type == "Pack-off" and "Packoff" in s_id) or \
                   ("Shut-In" in s_id and f_obj.failure_type in ["Gas Kick", "Ram Leak"]) or \
                   ("API-RP-53" in s_id and f_obj.failure_type == "Ram Leak") or \
                   ("IADC" in s_id):
                    relationships.append(Relationship(
                        source_type="FailureMode",
                        source_id=f_code,
                        relation_type="MITIGATED_BY",
                        target_type="SOP",
                        target_id=s_id,
                    ))
            for f_name in formations:
                relationships.append(Relationship(
                    source_type="FailureMode",
                    source_id=f_code,
                    relation_type="OCCURS_IN",
                    target_type="Formation",
                    target_id=f_name,
                ))

        # Connect Equipment to SOPs
        for eq_serial in equipment:
            for s_id in sops:
                if "API-RP-53" in s_id and "BOP" in eq_serial:
                    relationships.append(Relationship(
                        source_type="Equipment",
                        source_id=eq_serial,
                        relation_type="MAINTAINED_UNDER",
                        target_type="SOP",
                        target_id=s_id,
                    ))

        return WellboreExtractionResult(
            wells=list(wells.values()),
            equipment=list(equipment.values()),
            formations=list(formations.values()),
            failure_modes=list(failure_modes.values()),
            sops=list(sops.values()),
            incidents=list(incidents.values()),
            relationships=relationships,
        )

    def _merge_results(self, r1: WellboreExtractionResult, r2: WellboreExtractionResult) -> WellboreExtractionResult:
        """Merge LLM-extracted and rule-extracted results without duplicates."""
        wells = {w.name: w for w in (r1.wells + r2.wells)}
        equipment = {e.serial_number: e for e in (r1.equipment + r2.equipment)}
        formations = {f.name: f for f in (r1.formations + r2.formations)}
        failures = {f.code: f for f in (r1.failure_modes + r2.failure_modes)}
        sops = {s.sop_id: s for s in (r1.sops + r2.sops)}
        incidents = {i.incident_id: i for i in (r1.incidents + r2.incidents)}

        rel_keys = set()
        merged_rels: List[Relationship] = []
        for r in r1.relationships + r2.relationships:
            key = (r.source_type, r.source_id, r.relation_type, r.target_type, r.target_id)
            if key not in rel_keys:
                rel_keys.add(key)
                merged_rels.append(r)

        return WellboreExtractionResult(
            wells=list(wells.values()),
            equipment=list(equipment.values()),
            formations=list(formations.values()),
            failure_modes=list(failures.values()),
            sops=list(sops.values()),
            incidents=list(incidents.values()),
            relationships=merged_rels,
        )
