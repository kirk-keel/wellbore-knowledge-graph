"""Tests for LAS log parsing and DDR timeline sequence generation."""

from pathlib import Path
from src.ingestion.las_parser import LASDocumentParser
from src.ingestion.ddr_timeline import DDRTimelineParser


def test_las_log_parser():
    las_path = Path("data/sample_docs/well_104b_lwd.las")
    assert las_path.exists()

    parser = LASDocumentParser()
    well, curves, anomalies = parser.parse_las_file(las_path)

    assert well.name == "Well 104-B"
    assert len(curves) >= 5
    # Anomaly detection: SPP spike over 3,400 psi
    assert len(anomalies) >= 1
    spp_anomalies = [a for a in anomalies if a["anomaly_type"] == "SPP_SPIKE"]
    assert len(spp_anomalies) >= 1
    assert spp_anomalies[0]["failure_mode"] == "FAIL-PACKOFF"


def test_ddr_timeline_sequence():
    ddr_path = Path("data/sample_docs/well_104b_daily_drilling_reports.txt")
    assert ddr_path.exists()

    parser = DDRTimelineParser()
    text = ddr_path.read_text(encoding="utf-8")
    entries, relationships = parser.parse_ddr_text(text)

    assert len(entries) >= 3
    # Verify report numbers order
    r_nums = [e.report_number for e in entries]
    assert r_nums == sorted(r_nums)

    # Verify NEXT_REPORT edges
    next_report_edges = [r for r in relationships if r.relation_type == "NEXT_REPORT"]
    assert len(next_report_edges) == len(entries) - 1
    assert next_report_edges[0].source_id == "Well 104-B:DDR_13"
    assert next_report_edges[0].target_id == "Well 104-B:DDR_14"
