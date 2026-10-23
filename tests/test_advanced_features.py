"""Comprehensive tests for Casing Mechanics, RCFA 5-Whys, Telemetry Watchdog, Multi-Agent Panel, and Wellbore Schematics."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.retrieval.casing_calculator import CasingIntegrityCalculator, CasingIntegrityRequest
from src.retrieval.rcfa_generator import RCFAGenerator, RCFARequest
from src.ingestion.telemetry_stream import TelemetryWatchdog, TelemetryFrame
from src.retrieval.multi_agent import MultiAgentDrillingPanel
from src.retrieval.wellbore_schematic import WellboreSchematicGenerator
from scripts.seed_database import seed

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def init_db():
    seed()


def test_casing_integrity_calculator():
    calc = CasingIntegrityCalculator()
    req = CasingIntegrityRequest(
        casing_serial_or_tag="CSG-958-L80",
        shoe_depth_tvd_ft=12500.0,
        target_formation_ppg=14.8,
        formation_fracture_gradient_ppg=17.2,
        mud_weight_ppg=15.2,
    )
    report = calc.evaluate_casing_integrity(req)

    assert report.casing_tag == "CSG-958-L80"
    assert report.maximum_anticipated_surface_pressure_masp_psi > 0
    assert report.burst_safety_factor > 0
    assert report.collapse_safety_factor > 0
    assert report.kick_tolerance_bbl > 0
    assert report.overall_integrity_status in ["APPROVED_BARRIER", "INTEGRITY_RISK"]


def test_rcfa_generator():
    generator = RCFAGenerator()
    req = RCFARequest(incident_id="INC-2024-008", well_name="Deepwater Titan Well")
    report = generator.generate_rcfa(req)

    assert report.incident_id == "INC-2024-008"
    assert len(report.five_whys_causal_tree) == 5
    assert report.five_whys_causal_tree[0].level == 1
    assert report.five_whys_causal_tree[4].level == 5
    assert len(report.preventative_barrier_actions) >= 3
    assert "# ROOT CAUSE FAILURE ANALYSIS" in report.report_markdown


def test_telemetry_watchdog_precursors():
    watchdog = TelemetryWatchdog()

    # Normal frame
    frame_norm = TelemetryFrame(
        timestamp_utc="2024-08-14T10:00:00Z",
        depth_ft=10450.0,
        standpipe_pressure_psi=2450.0,
        pit_volume_bbl=520.0,
    )
    alert1 = watchdog.ingest_frame(frame_norm)
    assert alert1 is None

    # SPP Spike Frame -> Pack-off Precursor
    frame_packoff = TelemetryFrame(
        timestamp_utc="2024-08-14T10:05:00Z",
        depth_ft=10480.0,
        standpipe_pressure_psi=3550.0,
        torque_kft_lb=28.0,
    )
    alert2 = watchdog.ingest_frame(frame_packoff)
    assert alert2 is not None
    assert alert2.precursor_type == "PACKOFF_PRECURSOR"
    assert "SOP-SP-774" in alert2.recommended_sop
    assert len(alert2.immediate_actions) >= 3

    # Pit Gain Frame -> Kick Precursor
    frame_kick = TelemetryFrame(
        timestamp_utc="2024-08-14T10:10:00Z",
        depth_ft=10500.0,
        standpipe_pressure_psi=2500.0,
        pit_volume_bbl=535.0,
        delta_pit_volume_bbl=15.0,
    )
    alert3 = watchdog.ingest_frame(frame_kick)
    assert alert3 is not None
    assert alert3.precursor_type == "KICK_PRECURSOR"
    assert "SOP-SHUTIN-01" in alert3.recommended_sop


def test_multi_agent_panel():
    panel = MultiAgentDrillingPanel()
    report = panel.consult("What procedures mitigate drillstring pack-off and gas influx?", well_name="Well 104-B")

    assert report.operational_classification == "URGENT_OPERATIONAL_REMEDIATION"
    assert len(report.committee_opinions) == 3
    roles = [o.role for o in report.committee_opinions]
    assert "Geomechanics Specialist" in roles
    assert "Equipment Reliability Specialist" in roles
    assert "Well Control Auditor" in roles
    assert len(report.consensus_operational_sequence) >= 3
    assert "Consensus Drilling Operational Directive" in report.final_advisory


def test_wellbore_schematic_generator():
    gen = WellboreSchematicGenerator()
    data = gen.generate_schematic("Well 104-B")

    assert data.well_name == "Well 104-B"
    assert len(data.casing_strings) >= 3
    assert len(data.formations) >= 2
    assert "<svg" in data.svg_schematic
    assert "</svg>" in data.svg_schematic


def test_advanced_api_endpoints():
    # 1. Casing Check Endpoint
    resp1 = client.post(
        "/api/v1/engineering/casing-check",
        json={"casing_serial_or_tag": "CSG-958-L80", "shoe_depth_tvd_ft": 10000.0, "target_formation_ppg": 14.0},
    )
    assert resp1.status_code == 200
    assert "burst_safety_factor" in resp1.json()

    # 2. RCFA Report Endpoint
    resp2 = client.post(
        "/api/v1/reports/rcfa",
        json={"incident_id": "INC-2024-008", "well_name": "Deepwater Titan Well"},
    )
    assert resp2.status_code == 200
    assert len(resp2.json()["five_whys_causal_tree"]) == 5

    # 3. Telemetry Stream Endpoint
    resp3 = client.post(
        "/api/v1/telemetry/stream",
        json={"timestamp_utc": "2024-08-14T12:00:00Z", "depth_ft": 10480.0, "standpipe_pressure_psi": 3600.0},
    )
    assert resp3.status_code == 200
    assert resp3.json()["status"] == "ALERT_TRIGGERED"

    # 4. Multi-Agent Consultation Endpoint
    resp4 = client.post(
        "/api/v1/query/multi-agent",
        json={"inquiry": "How to proceed when encountering Draupne reactive shale?", "well_name": "Well 104-B"},
    )
    assert resp4.status_code == 200
    assert "committee_opinions" in resp4.json()

    # 5. Wellbore Schematic Endpoints
    resp5 = client.get("/api/v1/wellbore/Well%20104-B/schematic")
    assert resp5.status_code == 200
    assert "svg_schematic" in resp5.json()

    resp6 = client.get("/api/v1/wellbore/Well%20104-B/schematic.svg")
    assert resp6.status_code == 200
    assert "image/svg+xml" in resp6.headers["content-type"]
