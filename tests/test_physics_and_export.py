"""Unit and integration tests for Downhole Hydraulics, 3D Anti-Collision, Vibrations, and WITSML Export."""

import pytest
from xml.etree import ElementTree as ET
from fastapi.testclient import TestClient

from src.api.main import app
from src.retrieval.hydraulics_engine import HydraulicsEngine, HydraulicsRequest
from src.retrieval.anti_collision import AntiCollisionScanner, SurveyStation
from src.retrieval.vibration_classifier import VibrationClassifier, VibrationDataPacket
from src.ingestion.witsml_exporter import WITSML141Exporter

client = TestClient(app)


def test_hydraulics_simulation():
    engine = HydraulicsEngine()
    req = HydraulicsRequest(
        well_name="Well 104-B",
        depth_tvd_ft=10500.0,
        mud_weight_ppg=12.2,
        flow_rate_gpm=650.0,
        hole_diameter_in=8.5,
        drillpipe_od_in=5.0,
        plastic_viscosity_cp=22.0,
        yield_point_lb_per_100sqft=16.0,
        tripping_speed_fpm=90.0,
        formation_pore_pressure_ppg=11.5,
        formation_fracture_gradient_ppg=14.8,
    )
    report = engine.simulate(req)

    assert report.well_name == "Well 104-B"
    assert report.annular_velocity_fpm > 0
    assert report.annular_pressure_loss_psi > 0
    assert report.equivalent_circulating_density_ecd_ppg > req.mud_weight_ppg
    assert report.swab_margin_safe is True
    assert report.surge_margin_safe is True
    assert len(report.recommendations) >= 1


def test_anti_collision_scanning():
    scanner = AntiCollisionScanner()
    stns = [
        SurveyStation(measured_depth_md_ft=0.0, inclination_deg=0.0, azimuth_deg=45.0),
        SurveyStation(measured_depth_md_ft=2000.0, inclination_deg=4.0, azimuth_deg=45.0),
        SurveyStation(measured_depth_md_ft=5000.0, inclination_deg=18.0, azimuth_deg=48.0),
        SurveyStation(measured_depth_md_ft=8000.0, inclination_deg=22.0, azimuth_deg=50.0),
    ]
    report = scanner.scan_anti_collision(active_well="Planned Well 104-C", active_stations=stns)

    assert report.active_well_name == "Planned Well 104-C"
    assert len(report.scanned_offset_wells) >= 1
    assert report.minimum_separation_factor > 0
    assert report.overall_status in ["APPROVED", "WARNING", "STOP_DRILLING"]
    assert len(report.scans) >= 1


def test_vibration_dysfunction_classifier():
    classifier = VibrationClassifier()

    # Severe Stick-Slip case
    packet_ss = VibrationDataPacket(
        surface_rpm_mean=100.0,
        rpm_max=220.0,
        rpm_min=0.0,
        wob_mean_klb=25.0,
        lateral_vibration_g_rms=1.5,
    )
    report_ss = classifier.classify(packet_ss)
    assert report_ss.stick_slip_severity_index_pct >= 100.0
    assert report_ss.primary_dysfunction == "SEVERE_TORSIONAL_STICK_SLIP"
    assert "Mud Motor MM-775-A1 (Rotor/Stator Debonding)" in report_ss.threatened_bha_assets
    assert len(report_ss.mitigating_actions) >= 2

    # Lateral Whirl case
    packet_whirl = VibrationDataPacket(
        surface_rpm_mean=120.0,
        rpm_max=130.0,
        rpm_min=110.0,
        lateral_vibration_g_rms=4.5,
    )
    report_whirl = classifier.classify(packet_whirl)
    assert report_whirl.lateral_whirl_severity == "CRITICAL_BACKWARD_WHIRL"
    assert "MWD-TENSOR-882 (Telemetry Failure)" in report_whirl.threatened_bha_assets


def test_witsml_xml_exporter():
    exporter = WITSML141Exporter()
    xml_str = exporter.export_trajectory_witsml(
        well_name="Well 104-B",
        wellbore_name="Wellbore-1",
        survey_name="Definitive_MWD",
    )

    assert "<?xml" in xml_str
    assert "<trajectorys" in xml_str
    assert "xmlns=\"http://www.witsml.org/schemas/1series\"" in xml_str
    assert "<nameWell>Well 104-B</nameWell>" in xml_str
    assert "<trajectoryStation" in xml_str

    # Validate XML parsing
    root = ET.fromstring(xml_str)
    assert "trajectorys" in root.tag


def test_physics_and_export_api_endpoints():
    # 1. Hydraulics & ECD
    res1 = client.post(
        "/api/v1/engineering/hydraulics-ecd",
        json={"well_name": "Well 104-B", "depth_tvd_ft": 9000.0, "mud_weight_ppg": 11.5},
    )
    assert res1.status_code == 200
    assert "equivalent_circulating_density_ecd_ppg" in res1.json()

    # 2. Anti-Collision Scan
    res2 = client.post(
        "/api/v1/engineering/anti-collision",
        json={"active_well_name": "Well 104-B"},
    )
    assert res2.status_code == 200
    assert "minimum_separation_factor" in res2.json()

    # 3. Vibration Classification
    res3 = client.post(
        "/api/v1/engineering/vibration-classify",
        json={"surface_rpm_mean": 110.0, "rpm_max": 230.0, "rpm_min": 5.0},
    )
    assert res3.status_code == 200
    assert "stick_slip_severity_index_pct" in res3.json()

    # 4. WITSML Export
    res4 = client.get("/api/v1/export/witsml/Well%20104-B")
    assert res4.status_code == 200
    assert "application/xml" in res4.headers["content-type"]
    assert "trajectoryStation" in res4.text
