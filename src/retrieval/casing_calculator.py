"""Quantitative Well Integrity, MASP, and Casing Stress Calculator."""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.models.domain import EquipmentEntity, FormationEntity, WellEntity

logger = logging.getLogger(__name__)

# Standard API 5CT Casing mechanical ratings reference
CASING_SPEC_DATABASE: Dict[str, Dict[str, float]] = {
    "CSG-958-L80": {
        "outer_diameter_in": 9.625,
        "nominal_weight_ppf": 47.0,
        "grade": "L-80",
        "burst_rating_psi": 6870.0,
        "collapse_rating_psi": 4760.0,
        "yield_strength_psi": 80000.0,
        "inner_diameter_in": 8.681,
    },
    "CSG-1338-P110": {
        "outer_diameter_in": 13.375,
        "nominal_weight_ppf": 68.0,
        "grade": "P-110",
        "burst_rating_psi": 8220.0,
        "collapse_rating_psi": 3930.0,
        "yield_strength_psi": 110000.0,
        "inner_diameter_in": 12.415,
    },
    "CSG-700-Q125": {
        "outer_diameter_in": 7.0,
        "nominal_weight_ppf": 32.0,
        "grade": "Q-125",
        "burst_rating_psi": 13830.0,
        "collapse_rating_psi": 12460.0,
        "yield_strength_psi": 125000.0,
        "inner_diameter_in": 6.094,
    },
}


class CasingIntegrityRequest(BaseModel):
    """Input payload for casing stress and well integrity verification."""
    well_name: str = Field(default="Deepwater Titan Well")
    casing_serial_or_tag: str = Field(default="CSG-958-L80")
    shoe_depth_tvd_ft: float = Field(default=12500.0, ge=100.0)
    target_formation_ppg: float = Field(default=14.8, ge=8.0, le=22.0)
    formation_fracture_gradient_ppg: float = Field(default=17.2, ge=8.0, le=24.0)
    mud_weight_ppg: float = Field(default=15.2, ge=8.0, le=22.0)
    gas_gradient_psi_per_ft: float = Field(default=0.10, ge=0.05, le=0.25)


class CasingIntegrityReport(BaseModel):
    """Detailed casing mechanics and barrier integrity evaluation."""
    casing_tag: str
    outer_diameter_in: float
    grade: str
    shoe_depth_tvd_ft: float
    formation_pore_pressure_psi: float
    maximum_anticipated_surface_pressure_masp_psi: float
    hydrostatic_pressure_psi: float
    burst_rating_psi: float
    collapse_rating_psi: float
    burst_safety_factor: float
    collapse_safety_factor: float
    burst_status: str
    collapse_status: str
    kick_tolerance_bbl: float
    overall_integrity_status: str
    recommendation: str


class CasingIntegrityCalculator:
    """Calculates downhole geomechanical loads, MASP, and verifies casing structural margins."""

    def __init__(self):
        pass

    def evaluate_casing_integrity(self, request: CasingIntegrityRequest) -> CasingIntegrityReport:
        """Execute full wellbore casing burst, collapse, MASP, and kick tolerance audit."""
        specs = CASING_SPEC_DATABASE.get(
            request.casing_serial_or_tag,
            CASING_SPEC_DATABASE["CSG-958-L80"],
        )

        tvd = request.shoe_depth_tvd_ft
        pp_ppg = request.target_formation_ppg
        fg_ppg = request.formation_fracture_gradient_ppg
        mw_ppg = request.mud_weight_ppg
        gas_grad = request.gas_gradient_psi_per_ft

        # 1. Pore Pressure & Hydrostatic Load
        p_pore = round(0.052 * pp_ppg * tvd, 1)
        p_hyd = round(0.052 * mw_ppg * tvd, 1)

        # 2. Maximum Anticipated Surface Pressure (MASP)
        # MASP = Formation Pore Pressure - Hydrostatic of full gas column to surface
        gas_hydrostatic = gas_grad * tvd
        masp = round(max(p_pore - gas_hydrostatic, 0.0), 1)

        # 3. Burst Load & Safety Factor
        burst_rating = specs["burst_rating_psi"]
        burst_sf = round(burst_rating / masp, 2) if masp > 0 else 99.0
        burst_status = "PASS (Compliant)" if burst_sf >= 1.125 else "FAIL (Sub-standard API Margin)"

        # 4. Collapse Load & Safety Factor
        # Conservative full evacuation collapse load at casing shoe
        collapse_rating = specs["collapse_rating_psi"]
        ext_pressure_at_shoe = round(0.052 * mw_ppg * tvd, 1)
        collapse_sf = round(collapse_rating / ext_pressure_at_shoe, 2) if ext_pressure_at_shoe > 0 else 99.0
        collapse_status = "PASS (Compliant)" if collapse_sf >= 1.00 else "WARNING (Risk under full evacuation)"

        # 5. Kick Tolerance (Max allowable influx volume before fracturing casing shoe)
        # Margin at shoe: (FG - MW) * 0.052 * TVD_shoe
        shoe_margin_psi = max(0.052 * (fg_ppg - mw_ppg) * tvd, 0.0)
        # Approximate annular capacity (bbl/ft) for 8.5" hole in 9.625" casing ~ 0.045 bbl/ft
        annular_capacity_bbl_per_ft = 0.045
        diff_gradient = max(0.052 * (mw_ppg - (gas_grad / 0.052)), 0.05)
        kick_tolerance_bbl = round(shoe_margin_psi / diff_gradient * annular_capacity_bbl_per_ft, 1)

        # Overall Status
        compliant = (burst_sf >= 1.125)
        overall_status = "APPROVED_BARRIER" if compliant else "INTEGRITY_RISK"

        recommendation = (
            f"Casing {request.casing_serial_or_tag} meets API Spec 5C3 burst criterion (SF={burst_sf} >= 1.125) "
            f"for penetrating {pp_ppg} ppg horizon. Maximum kick volume allowable without shoe breakdown is {kick_tolerance_bbl} bbl."
            if compliant else
            f"CRITICAL DEFICIENCY: Calculated MASP ({masp} psi) exceeds allowable API safety margin for {request.casing_serial_or_tag} "
            f"(SF={burst_sf} < 1.125). Upgrade to heavyweight casing or set an intermediate liner string."
        )

        return CasingIntegrityReport(
            casing_tag=request.casing_serial_or_tag,
            outer_diameter_in=specs["outer_diameter_in"],
            grade=specs["grade"],
            shoe_depth_tvd_ft=tvd,
            formation_pore_pressure_psi=p_pore,
            maximum_anticipated_surface_pressure_masp_psi=masp,
            hydrostatic_pressure_psi=p_hyd,
            burst_rating_psi=burst_rating,
            collapse_rating_psi=collapse_rating,
            burst_safety_factor=burst_sf,
            collapse_safety_factor=collapse_sf,
            burst_status=burst_status,
            collapse_status=collapse_status,
            kick_tolerance_bbl=kick_tolerance_bbl,
            overall_integrity_status=overall_status,
            recommendation=recommendation,
        )
