"""Tests for domain and API models."""

import pytest
from src.models.domain import (
    EquipmentEntity,
    FailureModeEntity,
    FormationEntity,
    IncidentEntity,
    Relationship,
    SOPEntity,
    WellEntity,
    DocumentChunk,
)
from src.models.api import QueryRequest, GraphQueryRequest, IngestTextRequest


def test_well_entity():
    w = WellEntity(name="Titan-101", field="Alvheim", total_depth_ft=12500.0)
    assert w.name == "Titan-101"
    assert w.total_depth_ft == 12500.0


def test_equipment_entity():
    eq = EquipmentEntity(
        serial_number="BOP-HYD-5000-08",
        equipment_type="Blowout Preventer",
        manufacturer="Cameron",
        working_pressure_psi=15000.0,
    )
    assert eq.serial_number == "BOP-HYD-5000-08"
    assert eq.working_pressure_psi == 15000.0


def test_formation_entity():
    f = FormationEntity(
        name="Eagle Ford Shale",
        lithology="Shale",
        top_depth_ft=8500.0,
        hazards=["Overpressured Gas", "Sloughing Shale"],
    )
    assert f.name == "Eagle Ford Shale"
    assert len(f.hazards) == 2


def test_failure_mode_entity():
    fm = FailureModeEntity(
        code="FAIL-PACKOFF",
        failure_type="Pack-off",
        severity="Critical",
        description="Cuttings accumulation causing stuck pipe",
    )
    assert fm.code == "FAIL-PACKOFF"
    assert fm.severity == "Critical"


def test_sop_entity():
    sop = SOPEntity(
        sop_id="IADC-WCC-04",
        title="Soft Shut-In Well Control Procedure",
        category="Well Control",
        required_barriers=["Hydrostatic Head", "BOP Annular"],
    )
    assert sop.sop_id == "IADC-WCC-04"
    assert len(sop.required_barriers) == 2


def test_relationship_model():
    rel = Relationship(
        source_type="FailureMode",
        source_id="FAIL-PACKOFF",
        relation_type="MITIGATED_BY",
        target_type="SOP",
        target_id="SOP-SP-774",
    )
    assert rel.relation_type == "MITIGATED_BY"
    assert rel.target_id == "SOP-SP-774"


def test_api_query_request():
    req = QueryRequest(question="What SOP mitigates ram leaks?")
    assert req.question == "What SOP mitigates ram leaks?"
    assert req.top_k_chunks == 5
    assert req.graph_hops == 2
