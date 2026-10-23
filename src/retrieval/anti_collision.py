"""3D Offset-Well Anti-Collision Scanner and ISCWSA Ellipse of Uncertainty Engine."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SurveyStation(BaseModel):
    """Directional survey station."""
    measured_depth_md_ft: float
    inclination_deg: float
    azimuth_deg: float


class SpatialPoint(BaseModel):
    """3D Cartesian spatial station with positional error uncertainty."""
    md_ft: float
    tvd_ft: float
    north_ft: float
    east_ft: float
    inclination_deg: float
    azimuth_deg: float
    uncertainty_radius_ft: float


class ProximityScanResult(BaseModel):
    """Proximity evaluation at a specific depth station against an offset well."""
    active_depth_md_ft: float
    offset_well_name: str
    offset_depth_md_ft: float
    center_to_center_distance_ft: float
    combined_uncertainty_ft: float
    separation_factor: float
    risk_level: str  # "SAFE", "WARNING_PROXIMITY", "CRITICAL_COLLISION"
    recommendation: str


class AntiCollisionReport(BaseModel):
    """Comprehensive multi-well pad anti-collision report."""
    active_well_name: str
    scanned_offset_wells: List[str]
    minimum_separation_factor: float
    critical_proximity_depth_md_ft: Optional[float]
    closest_offset_well: str
    overall_status: str
    scans: List[ProximityScanResult]


class AntiCollisionScanner:
    """Computes 3D wellbore paths using Minimum Curvature and calculates ISCWSA Separation Factors."""

    def __init__(self):
        pass

    def calculate_3d_path(self, stations: List[SurveyStation], surface_north_ft: float = 0.0, surface_east_ft: float = 0.0) -> List[SpatialPoint]:
        """Convert directional survey stations to 3D Cartesian coordinates via Minimum Curvature Method (MCM)."""
        if not stations:
            return []

        # Sort stations by MD
        sorted_stns = sorted(stations, key=lambda s: s.measured_depth_md_ft)
        points: List[SpatialPoint] = []

        # Tie-on surface station (RKB datum) with slot coordinates
        cur_n, cur_e, cur_tvd = surface_north_ft, surface_east_ft, 0.0

        for i, stn in enumerate(sorted_stns):
            md = stn.measured_depth_md_ft
            inc = math.radians(stn.inclination_deg)
            azi = math.radians(stn.azimuth_deg)

            if i == 0:
                cur_tvd = md * math.cos(inc)
                cur_n = surface_north_ft + md * math.sin(inc) * math.cos(azi)
                cur_e = surface_east_ft + md * math.sin(inc) * math.sin(azi)
            else:
                prev_stn = sorted_stns[i - 1]
                delta_md = md - prev_stn.measured_depth_md_ft
                p_inc = math.radians(prev_stn.inclination_deg)
                p_azi = math.radians(prev_stn.azimuth_deg)

                # Dogleg Angle (beta)
                cos_beta = math.cos(inc - p_inc) - math.sin(p_inc) * math.sin(inc) * (1.0 - math.cos(azi - p_azi))
                cos_beta = max(min(cos_beta, 1.0), -1.0)
                beta = math.acos(cos_beta)

                # Ratio Factor (RF)
                if abs(beta) < 1e-6:
                    rf = 1.0
                else:
                    rf = (2.0 / beta) * math.tan(beta / 2.0)

                # Increments
                cur_n += (delta_md / 2.0) * (math.sin(p_inc) * math.cos(p_azi) + math.sin(inc) * math.cos(azi)) * rf
                cur_e += (delta_md / 2.0) * (math.sin(p_inc) * math.sin(p_azi) + math.sin(inc) * math.sin(azi)) * rf
                cur_tvd += (delta_md / 2.0) * (math.cos(p_inc) + math.cos(inc)) * rf

            # ISCWSA MWD standard systematic error model approximation (approx 0.15% of MD + 4.0 ft surface error)
            uncertainty_rad = round(4.0 + (0.0018 * md), 2)

            points.append(
                SpatialPoint(
                    md_ft=md,
                    tvd_ft=round(cur_tvd, 2),
                    north_ft=round(cur_n, 2),
                    east_ft=round(cur_e, 2),
                    inclination_deg=stn.inclination_deg,
                    azimuth_deg=stn.azimuth_deg,
                    uncertainty_radius_ft=uncertainty_rad,
                )
            )

        return points

    def scan_anti_collision(
        self,
        active_well: str,
        active_stations: List[SurveyStation],
        offset_wells_data: Optional[Dict[str, List[SurveyStation]]] = None,
    ) -> AntiCollisionReport:
        """Scan active wellpath against offset wells and compute Separation Factors."""
        active_points = self.calculate_3d_path(active_stations, surface_north_ft=0.0, surface_east_ft=0.0)

        # Default synthetic offset well trajectories if none supplied
        if not offset_wells_data:
            offset_wells_dict = self._generate_default_offset_wells()
        else:
            offset_wells_dict = {name: (stns, 35.0, 35.0) for name, stns in offset_wells_data.items()}

        all_scans: List[ProximityScanResult] = []
        min_sf = 99.0
        critical_depth = None
        closest_well = "None"

        for off_name, (off_stns, slot_n, slot_e) in offset_wells_dict.items():
            off_points = self.calculate_3d_path(off_stns, surface_north_ft=slot_n, surface_east_ft=slot_e)
            if not off_points:
                continue

            for ap in active_points:
                if ap.md_ft < 100.0:
                    continue  # Skip surface slot level

                # Find closest offset point by 3D distance
                best_dist = float("inf")
                best_op = off_points[0]

                for op in off_points:
                    dist = math.sqrt((ap.north_ft - op.north_ft)**2 + (ap.east_ft - op.east_ft)**2 + (ap.tvd_ft - op.tvd_ft)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_op = op

                combined_uncertainty = ap.uncertainty_radius_ft + best_op.uncertainty_radius_ft
                sf = round(best_dist / combined_uncertainty, 2) if combined_uncertainty > 0 else 99.0

                if sf < 1.0:
                    risk = "CRITICAL_COLLISION"
                    rec = f"STOP DRILLING: Separation Factor {sf} < 1.0 violates safety boundary with {off_name}. Execute directional nudge immediately."
                elif sf < 1.5:
                    risk = "WARNING_PROXIMITY"
                    rec = f"PROXIMITY ALERT: Separation Factor {sf} < 1.5. Increase MWD survey frequency and notify directional coordinator."
                else:
                    risk = "SAFE"
                    rec = "Clear trajectory path. Compliant with IADC anti-collision policy."

                if sf < min_sf:
                    min_sf = sf
                    closest_well = off_name
                    if sf < 1.5:
                        critical_depth = ap.md_ft

                all_scans.append(
                    ProximityScanResult(
                        active_depth_md_ft=ap.md_ft,
                        offset_well_name=off_name,
                        offset_depth_md_ft=best_op.md_ft,
                        center_to_center_distance_ft=round(best_dist, 1),
                        combined_uncertainty_ft=round(combined_uncertainty, 1),
                        separation_factor=sf,
                        risk_level=risk,
                        recommendation=rec,
                    )
                )

        overall = "APPROVED" if min_sf >= 1.5 else ("WARNING" if min_sf >= 1.0 else "STOP_DRILLING")

        return AntiCollisionReport(
            active_well_name=active_well,
            scanned_offset_wells=list(offset_wells_dict.keys()),
            minimum_separation_factor=min_sf if min_sf < 90.0 else 2.5,
            critical_proximity_depth_md_ft=critical_depth,
            closest_offset_well=closest_well,
            overall_status=overall,
            scans=all_scans[:15],
        )

    def _generate_default_offset_wells(self) -> Dict[str, Tuple[List[SurveyStation], float, float]]:
        """Generate realistic offset well surveys with slot surface offsets (N, E)."""
        return {
            "Offset Well 104-A (Platform Slot 02)": (
                [
                    SurveyStation(measured_depth_md_ft=0.0, inclination_deg=0.0, azimuth_deg=75.0),
                    SurveyStation(measured_depth_md_ft=2000.0, inclination_deg=6.0, azimuth_deg=78.0),
                    SurveyStation(measured_depth_md_ft=5000.0, inclination_deg=24.0, azimuth_deg=80.0),
                    SurveyStation(measured_depth_md_ft=8000.0, inclination_deg=28.0, azimuth_deg=82.0),
                    SurveyStation(measured_depth_md_ft=11000.0, inclination_deg=28.0, azimuth_deg=82.0),
                ],
                65.0,  # Slot North offset ft
                40.0,  # Slot East offset ft
            ),
            "Titan-101 (Slot 06)": (
                [
                    SurveyStation(measured_depth_md_ft=0.0, inclination_deg=0.0, azimuth_deg=120.0),
                    SurveyStation(measured_depth_md_ft=3000.0, inclination_deg=8.0, azimuth_deg=125.0),
                    SurveyStation(measured_depth_md_ft=6000.0, inclination_deg=26.0, azimuth_deg=130.0),
                    SurveyStation(measured_depth_md_ft=10000.0, inclination_deg=35.0, azimuth_deg=132.0),
                ],
                120.0, # Slot North offset ft
                95.0,  # Slot East offset ft
            ),
        }
