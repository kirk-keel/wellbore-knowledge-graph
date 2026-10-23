"""Downhole drillstring vibration and dynamic drilling dysfunction classifier."""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VibrationDataPacket(BaseModel):
    """High-frequency telemetry packet capturing dynamic vibrations."""
    well_name: str = Field(default="Well 104-B")
    depth_ft: float = Field(default=10480.0)
    surface_rpm_mean: float = Field(default=120.0, ge=10.0)
    rpm_max: float = Field(default=220.0, ge=0.0)
    rpm_min: float = Field(default=10.0, ge=0.0)
    torque_mean_kft_lb: float = Field(default=15.0)
    torque_std_kft_lb: float = Field(default=6.5)
    wob_mean_klb: float = Field(default=25.0)
    wob_variance_klb: float = Field(default=9.0)
    lateral_vibration_g_rms: float = Field(default=3.5, description="RMS lateral acceleration in g")
    axial_vibration_g_rms: float = Field(default=1.8, description="RMS axial acceleration in g")


class VibrationClassificationReport(BaseModel):
    """Diagnosed drilling dysfunctions, severity ratings, and mitigation actions."""
    well_name: str
    depth_ft: float
    stick_slip_severity_index_pct: float
    bit_bounce_severity: str
    lateral_whirl_severity: str
    primary_dysfunction: str
    risk_level: str  # "SAFE", "MODERATE_FATIGUE", "CRITICAL_TOOL_FAILURE"
    threatened_bha_assets: List[str]
    mitigating_actions: List[str]
    recommended_parameters: Dict[str, str]


class VibrationClassifier:
    """Classifies drillstring dynamic dysfunctions: Stick-Slip, Bit Bounce, and Lateral Whirl."""

    def __init__(self):
        pass

    def classify(self, data: VibrationDataPacket) -> VibrationClassificationReport:
        """Analyze vibrational data and identify dysfunction severity."""
        rpm_mean = data.surface_rpm_mean
        rpm_delta = data.rpm_max - data.rpm_min

        # 1. Stick-Slip Severity Index (SSSI %)
        # SSSI = (RPM_max - RPM_min) / (2 * RPM_mean) * 100
        sssi = round((rpm_delta / (2.0 * rpm_mean)) * 100.0, 1)

        # 2. Axial Bit Bounce
        # High axial g-RMS (> 2.5 g) or WOB variance > 35% of mean indicates bit bounce
        wob_ratio = data.wob_variance_klb / data.wob_mean_klb if data.wob_mean_klb > 0 else 0
        if data.axial_vibration_g_rms > 2.8 or wob_ratio > 0.40:
            bit_bounce = "SEVERE"
        elif data.axial_vibration_g_rms > 1.5 or wob_ratio > 0.25:
            bit_bounce = "MODERATE"
        else:
            bit_bounce = "LOW"

        # 3. Lateral & Backward Whirl
        # Lateral g-RMS > 4.0 g indicates severe backward whirl destroying MWD electronics
        if data.lateral_vibration_g_rms > 4.0:
            whirl = "CRITICAL_BACKWARD_WHIRL"
        elif data.lateral_vibration_g_rms > 2.0:
            whirl = "MODERATE_WHIRL"
        else:
            whirl = "NEGLIGIBLE"

        # 4. Primary Dysfunction & Threat Level
        threatened_assets = []
        mitigations = []
        rec_params = {}

        if sssi >= 85.0:
            primary = "SEVERE_TORSIONAL_STICK_SLIP"
            risk = "CRITICAL_TOOL_FAILURE"
            threatened_assets.extend(["Mud Motor MM-775-A1 (Rotor/Stator Debonding)", "Tool Joints (FAIL-TWISTOFF)"])
            mitigations.extend([
                "Engage automated Soft-Torque / Dynamic Damping rotary drive system immediately.",
                "Increase surface RPM by 15-20 RPM to exit the torsional resonance harmonic envelope.",
                "Slightly decrease WOB by 3-5 klb to reduce cutter depth of cut and reactive torque.",
            ])
            rec_params = {"target_rpm": f"{int(rpm_mean + 15)} RPM", "target_wob": f"{int(data.wob_mean_klb - 4)} klb"}

        elif whirl == "CRITICAL_BACKWARD_WHIRL":
            primary = "HIGH_ENERGY_LATERAL_WHIRL"
            risk = "CRITICAL_TOOL_FAILURE"
            threatened_assets.extend(["MWD-TENSOR-882 (Telemetry Failure)", "PDC Bit Gauge Cutters"])
            mitigations.extend([
                "Pick up 2 feet off bottom and eliminate rotation to disrupt backward precession.",
                "Re-engage bottom with 10% lower RPM and higher WOB to stabilize bit in center.",
            ])
            rec_params = {"target_rpm": f"{int(rpm_mean - 15)} RPM", "target_wob": f"{int(data.wob_mean_klb + 3)} klb"}

        elif bit_bounce == "SEVERE":
            primary = "AXIAL_BIT_BOUNCE"
            risk = "MODERATE_FATIGUE"
            threatened_assets.extend(["PDC Bit Face Impact Damage", "Mud Motor Thrust Bearings"])
            mitigations.extend([
                "Increase WOB to stabilize cutter engagement; reduce rotary table RPM.",
            ])
            rec_params = {"target_rpm": f"{int(rpm_mean - 10)} RPM", "target_wob": f"{int(data.wob_mean_klb + 4)} klb"}

        else:
            primary = "BENIGN_VIBRATION_REGIME"
            risk = "SAFE"
            threatened_assets.append("Normal Wear Rate")
            mitigations.append("Maintain current optimal drilling parameters.")
            rec_params = {"target_rpm": f"{int(rpm_mean)} RPM", "target_wob": f"{int(data.wob_mean_klb)} klb"}

        return VibrationClassificationReport(
            well_name=data.well_name,
            depth_ft=data.depth_ft,
            stick_slip_severity_index_pct=sssi,
            bit_bounce_severity=bit_bounce,
            lateral_whirl_severity=whirl,
            primary_dysfunction=primary,
            risk_level=risk,
            threatened_bha_assets=threatened_assets,
            mitigating_actions=mitigations,
            recommended_parameters=rec_params,
        )
