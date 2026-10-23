"""Parser for CWLS Log ASCII Standard (.las) drilling and wireline logs."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from src.models.domain import TrajectoryPoint, WellEntity, WellLogCurve

logger = logging.getLogger(__name__)


class LASDocumentParser:
    """Parses standard .las files and survey logs into structured well log curves and trajectory points."""

    def __init__(self):
        pass

    def parse_las_file(self, file_path: str | Path) -> Tuple[WellEntity, List[WellLogCurve], List[Dict[str, Any]]]:
        """Parse a .las file, return WellEntity, curve rows, and detected downhole anomalies."""
        path = Path(file_path)
        content = path.read_text(encoding="utf-8", errors="replace")
        return self.parse_las_text(content, filename=path.name)

    def parse_las_text(self, text: str, filename: str = "drilling_log.las") -> Tuple[WellEntity, List[WellLogCurve], List[Dict[str, Any]]]:
        """Parse raw LAS string."""
        well_meta: Dict[str, str] = {}
        curve_names: List[str] = []
        data_rows: List[str] = []

        section = None
        for line in text.splitlines():
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue

            if line_str.startswith("~"):
                s_char = line_str[1].upper()
                if s_char == "W":
                    section = "WELL"
                elif s_char == "C":
                    section = "CURVE"
                elif s_char == "A":
                    section = "ASCII_DATA"
                else:
                    section = "OTHER"
                continue

            if section == "WELL":
                # Line format: MNEM.UNIT  VALUE : DESCRIPTION
                match = re.match(r"^([A-Za-z0-9_]+)\s*\.([^\s]*)\s*([^:]*)\s*:\s*(.*)", line_str)
                if match:
                    mnem = match.group(1).upper()
                    val = match.group(3).strip()
                    well_meta[mnem] = val
            elif section == "CURVE":
                match = re.match(r"^([A-Za-z0-9_]+)\s*\.", line_str)
                if match:
                    curve_names.append(match.group(1).upper())
            elif section == "ASCII_DATA":
                data_rows.append(line_str)

        # Build Well Entity
        well_name = well_meta.get("WELL", filename.replace(".las", ""))
        well = WellEntity(
            name=well_name,
            api_number=well_meta.get("API"),
            field=well_meta.get("FLD", "North Sea / Deepwater"),
            total_depth_ft=float(well_meta.get("STOP", 0.0)) if well_meta.get("STOP") else None,
            operator=well_meta.get("COMP", "Drilling Operator"),
        )

        curves: List[WellLogCurve] = []
        anomalies: List[Dict[str, Any]] = []

        # Map curve indices
        idx_map = {name: i for i, name in enumerate(curve_names)}

        for row in data_rows:
            parts = row.split()
            if len(parts) < len(curve_names):
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue

            depth = vals[idx_map.get("DEPT", 0)] if "DEPT" in idx_map else vals[0]
            rop = vals[idx_map["ROP"]] if "ROP" in idx_map else None
            spp = vals[idx_map["SPP"]] if "SPP" in idx_map else None
            torq = vals[idx_map["TORQ"]] if "TORQ" in idx_map else None
            flow = vals[idx_map["FLOW"]] if "FLOW" in idx_map else None
            pit = vals[idx_map["PITG"]] if "PITG" in idx_map else None
            gr = vals[idx_map["GR"]] if "GR" in idx_map else None

            curve = WellLogCurve(
                depth_md_ft=depth,
                rate_of_penetration_fph=rop,
                standpipe_pressure_psi=spp,
                torque_kft_lb=torq,
                flow_rate_gpm=flow,
                pit_gain_bbl=pit,
                gamma_ray_api=gr,
            )
            curves.append(curve)

            # Anomaly rules (drilling precursor analysis)
            if spp and spp > 3400.0:
                anomalies.append({
                    "depth_ft": depth,
                    "anomaly_type": "SPP_SPIKE",
                    "failure_mode": "FAIL-PACKOFF",
                    "severity": "Critical",
                    "details": f"Standpipe pressure reached {spp:.1f} psi exceeding 3,400 psi threshold (pack-off precursor).",
                })
            if pit and pit > 5.0:
                anomalies.append({
                    "depth_ft": depth,
                    "anomaly_type": "PIT_GAIN",
                    "failure_mode": "FAIL-KICK",
                    "severity": "Critical",
                    "details": f"Active pit volume gain of {pit:.1f} bbl recorded at {depth} ft (well influx/kick indicator).",
                })

        logger.info("Parsed LAS file %s: %d depth points, %d anomalies flagged", filename, len(curves), len(anomalies))
        return well, curves, anomalies
