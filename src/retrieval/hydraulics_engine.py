"""Downhole Hydraulics, ECD, and Surge/Swab Pressure Simulation Engine."""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HydraulicsRequest(BaseModel):
    """Input parameters for downhole hydraulics and surge/swab simulation."""
    well_name: str = Field(default="Well 104-B")
    depth_tvd_ft: float = Field(default=10500.0, ge=100.0)
    mud_weight_ppg: float = Field(default=12.2, ge=8.0, le=20.0)
    flow_rate_gpm: float = Field(default=650.0, ge=50.0, le=1500.0)
    hole_diameter_in: float = Field(default=8.5, ge=4.0, le=36.0)
    drillpipe_od_in: float = Field(default=5.0, ge=2.375, le=9.625)
    plastic_viscosity_cp: float = Field(default=22.0, ge=1.0, le=100.0)
    yield_point_lb_per_100sqft: float = Field(default=16.0, ge=1.0, le=80.0)
    tripping_speed_fpm: float = Field(default=90.0, ge=10.0, le=300.0, description="Pipe running/pulling speed in ft/min")
    formation_pore_pressure_ppg: float = Field(default=11.5, ge=8.0, le=20.0)
    formation_fracture_gradient_ppg: float = Field(default=14.8, ge=8.0, le=22.0)


class HydraulicsReport(BaseModel):
    """Simulated downhole annular hydraulics and wellbore pressure margins."""
    well_name: str
    depth_tvd_ft: float
    static_hydrostatic_pressure_psi: float
    annular_velocity_fpm: float
    annular_pressure_loss_psi: float
    equivalent_circulating_density_ecd_ppg: float
    swab_pressure_drop_psi: float
    surge_pressure_increase_psi: float
    effective_swab_density_ppg: float
    effective_surge_density_ppg: float
    swab_margin_safe: bool
    surge_margin_safe: bool
    operating_window_summary: str
    recommendations: List[str]


class HydraulicsEngine:
    """Calculates rheological annular frictional losses, ECD, and transient swab/surge margins."""

    def __init__(self):
        pass

    def simulate(self, req: HydraulicsRequest) -> HydraulicsReport:
        """Run Bingham Plastic hydraulics and swab/surge simulation."""
        tvd = req.depth_tvd_ft
        mw = req.mud_weight_ppg
        gpm = req.flow_rate_gpm
        dh = req.hole_diameter_in
        dp = req.drillpipe_od_in
        pv = req.plastic_viscosity_cp
        yp = req.yield_point_lb_per_100sqft
        v_trip = req.tripping_speed_fpm
        pp = req.formation_pore_pressure_ppg
        fg = req.formation_fracture_gradient_ppg

        # 1. Static Hydrostatic Pressure
        p_hyd = round(0.052 * mw * tvd, 1)

        # 2. Annular Velocity (ft/min)
        # AV = (24.51 * Q) / (Dh^2 - Dp^2)
        hydraulic_diameter = dh - dp
        annular_area_factor = (dh**2 - dp**2)
        av_fpm = round((24.51 * gpm) / annular_area_factor, 1)

        # 3. Annular Frictional Pressure Loss (Bingham Plastic)
        # dP/dL = (PV * AV) / (60,000 * (Dh - Dp)^2) + YP / (200 * (Dh - Dp))
        dp_viscous_per_ft = (pv * av_fpm) / (60000.0 * (hydraulic_diameter**2))
        dp_yield_per_ft = yp / (200.0 * hydraulic_diameter)
        dp_ann_per_ft = dp_viscous_per_ft + dp_yield_per_ft
        p_ann_loss = round(dp_ann_per_ft * tvd, 1)

        # 4. Equivalent Circulating Density (ECD)
        # ECD = MW + (dP_ann) / (0.052 * TVD)
        delta_ecd = round(p_ann_loss / (0.052 * tvd), 2)
        ecd = round(mw + delta_ecd, 2)

        # 5. Surge and Swab Pressure
        # Effective mud clinging constant for plugged pipe ~ 0.45
        k_cling = 0.45
        v_eff = k_cling * v_trip
        dp_surge_per_ft = (pv * v_eff) / (60000.0 * (hydraulic_diameter**2)) + yp / (200.0 * hydraulic_diameter)
        delta_p_trip = round(dp_surge_per_ft * tvd, 1)

        # Equivalent density offsets
        trip_delta_ppg = round(delta_p_trip / (0.052 * tvd), 2)
        swab_ppg = round(mw - trip_delta_ppg, 2)
        surge_ppg = round(mw + trip_delta_ppg, 2)

        # 6. Safety Margin Evaluation against Formation Horizons
        # Swab must not drop below pore pressure (otherwise swabbed kick occurs)
        swab_safe = swab_ppg >= pp
        # Surge must not exceed fracture gradient (otherwise lost circulation occurs)
        surge_safe = surge_ppg <= fg

        recommendations = []
        if not swab_safe:
            max_safe_trip_speed = round(v_trip * max(mw - pp, 0.1) / max(trip_delta_ppg, 0.1), 0)
            recommendations.append(
                f"SWAB HAZARD: Pulling pipe at {v_trip} fpm reduces bottomhole pressure to {swab_ppg} ppg "
                f"below formation pore pressure ({pp} ppg). Limit tripping speed to <= {max(max_safe_trip_speed, 30.0)} fpm to prevent swabbed kick."
            )
        else:
            recommendations.append(f"Swab margin is safe ({swab_ppg} ppg > {pp} ppg pore pressure).")

        if not surge_safe:
            recommendations.append(
                f"SURGE HAZARD: Running pipe at {v_trip} fpm elevates bottomhole pressure to {surge_ppg} ppg "
                f"exceeding formation fracture gradient ({fg} ppg). Reduce running speed and condition mud to lower yield point."
            )
        else:
            recommendations.append(f"Surge margin is safe ({surge_ppg} ppg < {fg} ppg fracture gradient).")

        if ecd >= fg:
            recommendations.append(
                f"CIRCULATION LOSS RISK: Dynamic ECD ({ecd} ppg) approaches/exceeds fracture gradient ({fg} ppg). "
                f"Reduce pump flow rate from {gpm} GPM to lower annular friction."
            )

        window_summary = (
            f"Drilling Window: Pore Pressure {pp} ppg < Static MW {mw} ppg < Dynamic ECD {ecd} ppg < Fracture Gradient {fg} ppg. "
            f"Status: {'STABLE' if (swab_safe and surge_safe and ecd < fg) else 'CRITICAL_WINDOW_RESTRICTION'}."
        )

        return HydraulicsReport(
            well_name=req.well_name,
            depth_tvd_ft=tvd,
            static_hydrostatic_pressure_psi=p_hyd,
            annular_velocity_fpm=av_fpm,
            annular_pressure_loss_psi=p_ann_loss,
            equivalent_circulating_density_ecd_ppg=ecd,
            swab_pressure_drop_psi=delta_p_trip,
            surge_pressure_increase_psi=delta_p_trip,
            effective_swab_density_ppg=swab_ppg,
            effective_surge_density_ppg=surge_ppg,
            swab_margin_safe=swab_safe,
            surge_margin_safe=surge_safe,
            operating_window_summary=window_summary,
            recommendations=recommendations,
        )
