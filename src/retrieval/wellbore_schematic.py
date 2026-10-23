"""2D/3D Wellbore Trajectory and Casing Schematic generator."""

import logging
import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CasingSection(BaseModel):
    """Casing string interval in the wellbore architecture."""
    name: str
    outer_diameter_in: float
    top_md_ft: float
    shoe_md_ft: float
    shoe_tvd_ft: float
    grade: str
    color: str = "#94a3b8"


class FormationHorizon(BaseModel):
    """Stratigraphic horizon depth marker."""
    name: str
    top_md_ft: float
    top_tvd_ft: float
    lithology: str
    color: str = "#3b82f6"


class WellboreSchematicData(BaseModel):
    """Complete architectural schematic data model for a well."""
    well_name: str
    total_depth_md_ft: float
    current_bit_depth_md_ft: float
    water_depth_ft: float
    casing_strings: List[CasingSection]
    formations: List[FormationHorizon]
    bha_tool_name: str
    svg_schematic: str


class WellboreSchematicGenerator:
    """Computes wellbore geometry and renders 2D/3D technical SVG schematics."""

    def __init__(self):
        pass

    def generate_schematic(self, well_name: str = "Well 104-B") -> WellboreSchematicData:
        """Construct wellbore casing architecture and generate technical SVG graphic."""
        casing_strings = [
            CasingSection(
                name='Conductor 30"',
                outer_diameter_in=30.0,
                top_md_ft=0.0,
                shoe_md_ft=450.0,
                shoe_tvd_ft=450.0,
                grade="X-52",
                color="#64748b",
            ),
            CasingSection(
                name='Surface Casing 13-3/8"',
                outer_diameter_in=13.375,
                top_md_ft=0.0,
                shoe_md_ft=3800.0,
                shoe_tvd_ft=3750.0,
                grade="P-110",
                color="#0ea5e9",
            ),
            CasingSection(
                name='Intermediate Casing 9-5/8"',
                outer_diameter_in=9.625,
                top_md_ft=0.0,
                shoe_md_ft=8200.0,
                shoe_tvd_ft=8000.0,
                grade="L-80",
                color="#f59e0b",
            ),
            CasingSection(
                name='Production Liner 7"',
                outer_diameter_in=7.0,
                top_md_ft=7900.0,
                shoe_md_ft=12500.0,
                shoe_tvd_ft=11800.0,
                grade="Q-125",
                color="#10b981",
            ),
        ]

        formations = [
            FormationHorizon(name="Brent Sandstone", top_md_ft=8400.0, top_tvd_ft=8200.0, lithology="Sandstone", color="#facc15"),
            FormationHorizon(name="Draupne Formation", top_md_ft=9850.0, top_tvd_ft=9500.0, lithology="Reactive Shale", color="#f87171"),
            FormationHorizon(name="Norphlet Sandstone", top_md_ft=11200.0, top_tvd_ft=10800.0, lithology="HPHT Sandstone", color="#fb923c"),
        ]

        bit_depth = 10500.0
        svg = self._render_svg(well_name, casing_strings, formations, bit_depth)

        return WellboreSchematicData(
            well_name=well_name,
            total_depth_md_ft=12500.0,
            current_bit_depth_md_ft=bit_depth,
            water_depth_ft=380.0,
            casing_strings=casing_strings,
            formations=formations,
            bha_tool_name="Mud Motor MM-775-A1 + 8-1/2 PDC Bit",
            svg_schematic=svg,
        )

    def _render_svg(self, well_name: str, casings: List[CasingSection], formations: List[FormationHorizon], bit_depth: float) -> str:
        """Draw scaled 2D wellbore schematic with casing shoes and formation tops."""
        # Scale: 12,500 ft mapped to ~600px height
        height = 700
        width = 460
        max_depth = 13000.0
        scale = 550.0 / max_depth
        y_offset = 60.0
        cx = 230.0

        svg = [
            f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background:#0f172a; border-radius:8px;">',
            f'  <text x="{cx}" y="32" fill="#38bdf8" font-size="14" font-weight="bold" text-anchor="middle">{well_name} - Wellbore Schematic</text>',
            f'  <text x="{cx}" y="48" fill="#94a3b8" font-size="10" text-anchor="middle">RKB Datum 0.0 ft | Total Depth 12,500 ft MD</text>',
            f'  <line x1="40" y1="{y_offset}" x2="{width - 40}" y2="{y_offset}" stroke="#475569" stroke-width="2"/>',
            f'  <text x="25" y="{y_offset + 4}" fill="#64748b" font-size="9" text-anchor="end">Surface</text>',
        ]

        # Draw Formation Horizons
        for f in formations:
            fy = y_offset + (f.top_md_ft * scale)
            svg.append(f'  <line x1="60" y1="{fy}" x2="{width - 60}" y2="{fy}" stroke="{f.color}" stroke-dasharray="4,4" stroke-width="1.5" opacity="0.7"/>')
            svg.append(f'  <text x="{width - 50}" y="{fy + 3}" fill="{f.color}" font-size="9">{f.name} ({int(f.top_md_ft)} ft)</text>')

        # Draw Open Hole
        hole_bottom_y = y_offset + (12500.0 * scale)
        svg.append(f'  <rect x="{cx - 22}" y="{y_offset}" width="44" height="{12500.0 * scale}" fill="#1e293b" opacity="0.6"/>')

        # Draw Casing Strings
        for c in casings:
            top_y = y_offset + (c.top_md_ft * scale)
            shoe_y = y_offset + (c.shoe_md_ft * scale)
            c_width = c.outer_diameter_in * 1.8
            half_w = c_width / 2.0

            # Casing Pipe Walls
            svg.append(f'  <!-- {c.name} -->')
            svg.append(f'  <rect x="{cx - half_w}" y="{top_y}" width="{c_width}" height="{shoe_y - top_y}" fill="none" stroke="{c.color}" stroke-width="2.5"/>')
            # Casing Shoe notch
            svg.append(f'  <polygon points="{cx - half_w},{shoe_y} {cx - half_w - 6},{shoe_y + 8} {cx - half_w},{shoe_y + 8}" fill="{c.color}"/>')
            svg.append(f'  <polygon points="{cx + half_w},{shoe_y} {cx + half_w + 6},{shoe_y + 8} {cx + half_w},{shoe_y + 8}" fill="{c.color}"/>')
            # Annotation
            svg.append(f'  <text x="50" y="{shoe_y + 4}" fill="{c.color}" font-size="8.5" text-anchor="end">{c.name} Shoe @ {int(c.shoe_md_ft)} ft</text>')

        # Draw Drillstring & BHA
        bha_y = y_offset + (bit_depth * scale)
        svg.append(f'  <!-- Drillstring -->')
        svg.append(f'  <line x1="{cx}" y1="{y_offset}" x2="{cx}" y2="{bha_y - 25}" stroke="#e2e8f0" stroke-width="3"/>')
        # Mud Motor
        svg.append(f'  <rect x="{cx - 6}" y="{bha_y - 25}" width="12" height="20" fill="#f59e0b" rx="2"/>')
        # PDC Bit
        svg.append(f'  <polygon points="{cx - 9},{bha_y - 5} {cx + 9},{bha_y - 5} {cx},{bha_y + 5}" fill="#ef4444"/>')
        svg.append(f'  <text x="{cx + 18}" y="{bha_y}" fill="#ef4444" font-size="9" font-weight="bold">Bit @ {int(bit_depth)} ft (Drilling Ahead)</text>')

        svg.append('</svg>')
        return "\n".join(svg)
