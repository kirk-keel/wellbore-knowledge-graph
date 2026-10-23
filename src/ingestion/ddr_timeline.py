"""Daily Drilling Report (DDR) parser and temporal sequence builder."""

import logging
import re
from typing import List, Tuple
from src.models.domain import DDREntry, IncidentEntity, Relationship

logger = logging.getLogger(__name__)


class DDRTimelineParser:
    """Parses multi-day DDR logs and constructs ordered temporal relationship chains."""

    def __init__(self):
        pass

    def parse_ddr_text(self, text: str, well_name_default: str = "Well 104-B") -> Tuple[List[DDREntry], List[Relationship]]:
        """Extract daily drilling reports and assemble sequential NEXT_REPORT graph edges."""
        entries: List[DDREntry] = []
        relationships: List[Relationship] = []

        # Split on DDR markers e.g. "--- DDR #" or "DAILY DRILLING REPORT #..."
        blocks = re.split(r"(?:^|\n)(?:[-=]+\s*)?(?:DAILY DRILLING REPORT|DDR)\s*#?\s*(\d+)", text, flags=re.IGNORECASE)

        if len(blocks) > 1:
            # blocks format: [preamble, number1, text1, number2, text2...]
            for i in range(1, len(blocks), 2):
                rpt_num = int(blocks[i])
                block_content = blocks[i + 1].strip()
                entry = self._parse_single_ddr(rpt_num, block_content, well_name_default)
                entries.append(entry)
        else:
            # Single DDR fallback
            entries.append(self._parse_single_ddr(1, text, well_name_default))

        # Sort chronologically by report number
        entries.sort(key=lambda d: d.report_number)

        # Build temporal sequence relationships: (:DDR)-[:NEXT_REPORT]->(:DDR)
        for i in range(len(entries) - 1):
            curr = entries[i]
            nxt = entries[i + 1]
            relationships.append(
                Relationship(
                    source_type="DDR",
                    source_id=f"{curr.well_name}:DDR_{curr.report_number}",
                    relation_type="NEXT_REPORT",
                    target_type="DDR",
                    target_id=f"{nxt.well_name}:DDR_{nxt.report_number}",
                    properties={"days_elapsed": 1},
                )
            )

        # Link each DDR to Well
        for e in entries:
            relationships.append(
                Relationship(
                    source_type="DDR",
                    source_id=f"{e.well_name}:DDR_{e.report_number}",
                    relation_type="RECORDED_FOR_WELL",
                    target_type="Well",
                    target_id=e.well_name,
                )
            )
            # Link reported incidents
            for inc in e.incidents_reported:
                relationships.append(
                    Relationship(
                        source_type="DDR",
                        source_id=f"{e.well_name}:DDR_{e.report_number}",
                        relation_type="REPORTED_INCIDENT",
                        target_type="Incident",
                        target_id=inc,
                    )
                )

        logger.info("Parsed %d DDR entries and generated %d temporal edges", len(entries), len(relationships))
        return entries, relationships

    def _parse_single_ddr(self, report_num: int, content: str, default_well: str) -> DDREntry:
        """Parse attributes of a single 24-hour DDR block."""
        # Date
        date_match = re.search(r"Date:?\s*([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{2}/[0-9]{2}/[0-9]{4})", content)
        report_date = date_match.group(1) if date_match else "2024-08-14"

        # Well name
        well_match = re.search(r"Well(?:\s+Name)?:?\s*([^\n\r]+)", content)
        well_name = well_match.group(1).strip() if well_match else default_well

        # Depth
        depth_match = re.search(r"(?:Midnight\s+Depth|Current\s+Depth|Depth):?\s*(\d{1,2}(?:,\d{3})?|\d{4,5})\s*(?:ft|m)?", content, re.IGNORECASE)
        cur_depth = float(depth_match.group(1).replace(",", "")) if depth_match else 10000.0

        # Progress
        prog_match = re.search(r"(?:Progress|Footage|Drilled):?\s*(\d+(?:\.\d+)?)\s*(?:ft|m)?", content, re.IGNORECASE)
        prog = float(prog_match.group(1)) if prog_match else 0.0

        # Mud weight
        mw_match = re.search(r"(?:Mud\s*Weight|MW):?\s*(\d+(?:\.\d+)?)\s*ppg", content, re.IGNORECASE)
        mw = float(mw_match.group(1)) if mw_match else None

        # NPT
        npt_match = re.search(r"(?:NPT|Non-Productive\s+Time):?\s*(\d+(?:\.\d+)?)\s*(?:hrs|hours)?", content, re.IGNORECASE)
        npt = float(npt_match.group(1)) if npt_match else 0.0

        # Incidents
        incidents = []
        for inc_match in re.finditer(r"(INC-[A-Z0-9-]+)", content):
            incidents.append(inc_match.group(1))

        # Clean summary
        summary = content.split("Operations Summary:")[-1].strip() if "Operations Summary:" in content else content[:300].strip()

        return DDREntry(
            report_number=report_num,
            well_name=well_name,
            report_date=report_date,
            current_depth_ft=cur_depth,
            progress_24hr_ft=prog,
            mud_weight_ppg=mw,
            operations_summary=summary,
            npt_hours=npt,
            incidents_reported=incidents,
        )
