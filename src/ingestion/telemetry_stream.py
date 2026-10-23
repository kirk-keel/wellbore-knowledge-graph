"""Real-time drilling telemetry stream ingestion and Knowledge Graph precursor watchdog."""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.ingestion.pipeline import IngestionPipeline, ingestion_pipeline

logger = logging.getLogger(__name__)


class TelemetryFrame(BaseModel):
    """Single discrete timestamped drilling sensor frame."""
    well_name: str = Field(default="Well 104-B")
    timestamp_utc: str = Field(..., description="ISO timestamp")
    depth_ft: float = Field(..., ge=0.0)
    rate_of_penetration_fph: float = Field(default=35.0)
    weight_on_bit_klb: float = Field(default=22.0)
    surface_rpm: float = Field(default=110.0)
    torque_kft_lb: float = Field(default=14.5)
    standpipe_pressure_psi: float = Field(default=2450.0)
    flow_rate_gpm: float = Field(default=650.0)
    pit_volume_bbl: float = Field(default=520.0)
    delta_pit_volume_bbl: float = Field(default=0.0)
    total_gas_units: float = Field(default=15.0)


class PrecursorAlert(BaseModel):
    """Early-warning advisory triggered by matching live telemetry to graph failure precursors."""
    alert_id: str
    severity: str  # "CRITICAL", "WARNING", "INFO"
    precursor_type: str  # "PACKOFF_PRECURSOR", "KICK_PRECURSOR", "DRILLING_BREAK"
    depth_ft: float
    trigger_metric: str
    observed_value: float
    threshold_value: float
    predicted_failure_mode: str
    recommended_sop: str
    immediate_actions: List[str]
    graph_citations: List[str]


class TelemetryWatchdog:
    """Monitors live sensor streams and queries Knowledge Graph failure patterns for real-time risk mitigation."""

    def __init__(self, pipeline: IngestionPipeline = ingestion_pipeline):
        self.pipeline = pipeline
        self.history: List[TelemetryFrame] = []
        self.active_alerts: List[PrecursorAlert] = []

    def ingest_frame(self, frame: TelemetryFrame) -> Optional[PrecursorAlert]:
        """Ingest single frame, evaluate sliding window thresholds, and trigger graph alerts."""
        self.history.append(frame)
        if len(self.history) > 500:
            self.history.pop(0)

        alert = None

        # 1. Pack-off Precursor: Standpipe Pressure Spike + Torque Volatility
        if frame.standpipe_pressure_psi > 3400.0 or (frame.standpipe_pressure_psi > 3100.0 and frame.torque_kft_lb > 25.0):
            alert = PrecursorAlert(
                alert_id=f"ALT-PACKOFF-{int(frame.depth_ft)}",
                severity="CRITICAL",
                precursor_type="PACKOFF_PRECURSOR",
                depth_ft=frame.depth_ft,
                trigger_metric="Standpipe Circulating Pressure (SPP)",
                observed_value=frame.standpipe_pressure_psi,
                threshold_value=3400.0,
                predicted_failure_mode="FAIL-PACKOFF (Drillstring Pack-off / Annular Restriction)",
                recommended_sop="SOP-SP-774 (Drillstring Packoff Remediation)",
                immediate_actions=[
                    "Immediately reduce mud circulating rate by 50% to prevent breaching formation fracture gradient.",
                    "Slack off drillstring weight; maintain moderate right-hand rotation (do NOT pull up with high overpull).",
                    "Reciprocate string in gentle 10-20 ft strokes.",
                    "Pump high-density low-viscosity pill per SOP-SP-774 Chapter 2.",
                ],
                graph_citations=["Well 104-B:DDR_14", "SOP-SP-774", "FAIL-PACKOFF", "MM-775-A1"],
            )

        # 2. Gas Kick Precursor: Pit Volume Gain
        elif frame.delta_pit_volume_bbl >= 4.0 or (len(self.history) >= 3 and (frame.pit_volume_bbl - self.history[-3].pit_volume_bbl) >= 4.0):
            gain = frame.delta_pit_volume_bbl or (frame.pit_volume_bbl - self.history[-3].pit_volume_bbl)
            alert = PrecursorAlert(
                alert_id=f"ALT-KICK-{int(frame.depth_ft)}",
                severity="CRITICAL",
                precursor_type="KICK_PRECURSOR",
                depth_ft=frame.depth_ft,
                trigger_metric="Pit Volume Gain Delta",
                observed_value=round(gain, 1),
                threshold_value=4.0,
                predicted_failure_mode="FAIL-KICK (Formation Fluid Influx)",
                recommended_sop="SOP-SHUTIN-01 (Soft Shut-in Method)",
                immediate_actions=[
                    "Space out drillstring so no tool joint is across BOP preventer rams.",
                    "Stop rotary table and shut down circulating pumps (PUMP-NOV-14P-01).",
                    "Open hydraulic choke line valve on choke manifold.",
                    "Close annular preventer or upper pipe rams on BOP-HYD-5000-08.",
                    "Slowly close remote hydraulic choke; record SIDPP and SICP at 3-minute intervals.",
                ],
                graph_citations=["IADC-WCC-04", "SOP-SHUTIN-01", "BOP-HYD-5000-08", "FAIL-KICK"],
            )

        # 3. Drilling Break: Sudden ROP surge indicating high permeability/overpressure entry
        elif frame.rate_of_penetration_fph > 90.0 and frame.weight_on_bit_klb <= 25.0:
            alert = PrecursorAlert(
                alert_id=f"ALT-BREAK-{int(frame.depth_ft)}",
                severity="WARNING",
                precursor_type="DRILLING_BREAK",
                depth_ft=frame.depth_ft,
                trigger_metric="Rate of Penetration (ROP)",
                observed_value=frame.rate_of_penetration_fph,
                threshold_value=90.0,
                predicted_failure_mode="Overpressured Permeable Sand Entry Hazard",
                recommended_sop="IADC-WCC-04 Section 2 (Flow Check Protocol)",
                immediate_actions=[
                    "Pick up off bottom 5 feet.",
                    "Shut down mud pumps and perform mandatory 10-minute flow check.",
                    "Observe flowline for active wellbore returns with pumps disengaged.",
                ],
                graph_citations=["IADC-WCC-04", "Norphlet Formation", "Eagle Ford Shale"],
            )

        if alert:
            self.active_alerts.append(alert)
            logger.warning("TELEMETRY WATCHDOG TRIGGERED: %s at %d ft", alert.precursor_type, alert.depth_ft)

        return alert

    def ingest_batch(self, frames: List[TelemetryFrame]) -> List[PrecursorAlert]:
        """Ingest batch of streaming frames and collect triggered alerts."""
        alerts = []
        for f in frames:
            alt = self.ingest_frame(f)
            if alt:
                alerts.append(alt)
        return alerts
