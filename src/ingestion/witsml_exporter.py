"""WITSML 1.4.1.1 XML Exporter for Trajectory Surveys and Wellbore Schematics."""

import logging
from xml.etree import ElementTree as ET
from xml.dom import minidom
from typing import List, Optional

from src.retrieval.anti_collision import SurveyStation, AntiCollisionScanner

logger = logging.getLogger(__name__)


class WITSML141Exporter:
    """Exports wellbore trajectories and casing programs into Energistics WITSML 1.4.1.1 XML standard."""

    def __init__(self):
        self.scanner = AntiCollisionScanner()

    def export_trajectory_witsml(
        self,
        well_name: str,
        wellbore_name: str,
        survey_name: str,
        stations: Optional[List[SurveyStation]] = None,
    ) -> str:
        """Generate official WITSML 1.4.1.1 <trajectorys> XML document."""
        if not stations:
            # Default trajectory stations
            stations = [
                SurveyStation(measured_depth_md_ft=0.0, inclination_deg=0.0, azimuth_deg=0.0),
                SurveyStation(measured_depth_md_ft=3800.0, inclination_deg=0.0, azimuth_deg=0.0),
                SurveyStation(measured_depth_md_ft=6500.0, inclination_deg=14.5, azimuth_deg=52.0),
                SurveyStation(measured_depth_md_ft=8200.0, inclination_deg=22.0, azimuth_deg=52.0),
                SurveyStation(measured_depth_md_ft=10500.0, inclination_deg=28.5, azimuth_deg=54.0),
                SurveyStation(measured_depth_md_ft=12500.0, inclination_deg=28.5, azimuth_deg=54.0),
            ]

        # Compute 3D path coordinates
        spatial_points = self.scanner.calculate_3d_path(stations)

        # Build XML hierarchy
        root = ET.Element(
            "trajectorys",
            xmlns="http://www.witsml.org/schemas/1series",
            version="1.4.1.1",
        )

        traj = ET.SubElement(root, "trajectory", uid=f"traj_{abs(hash(well_name))}")
        ET.SubElement(traj, "nameWell").text = well_name
        ET.SubElement(traj, "nameWellbore").text = wellbore_name
        ET.SubElement(traj, "name").text = survey_name

        # Trajectory Stations
        for idx, pt in enumerate(spatial_points):
            stn_elem = ET.SubElement(traj, "trajectoryStation", uid=f"stn_{idx + 1}")
            
            md_elem = ET.SubElement(stn_elem, "md", uom="ft")
            md_elem.text = f"{pt.md_ft:.2f}"

            tvd_elem = ET.SubElement(stn_elem, "tvd", uom="ft")
            tvd_elem.text = f"{pt.tvd_ft:.2f}"

            inc_elem = ET.SubElement(stn_elem, "incl", uom="dega")
            inc_elem.text = f"{pt.inclination_deg:.2f}"

            azi_elem = ET.SubElement(stn_elem, "azi", uom="dega")
            azi_elem.text = f"{pt.azimuth_deg:.2f}"

            disp_n = ET.SubElement(stn_elem, "dispNs", uom="ft")
            disp_n.text = f"{pt.north_ft:.2f}"

            disp_e = ET.SubElement(stn_elem, "dispEw", uom="ft")
            disp_e.text = f"{pt.east_ft:.2f}"

            ET.SubElement(stn_elem, "typeTrajStation").text = "tie in" if idx == 0 else "MWD"

        # Pretty print XML
        rough_string = ET.tostring(root, encoding="utf-8")
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
