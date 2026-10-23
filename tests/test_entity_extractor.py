"""Tests for domain entity and relationship extraction."""

from src.ingestion.entity_extractor import WellboreEntityExtractor


def test_entity_extraction_drilling_incident():
    extractor = WellboreEntityExtractor()
    sample_text = """
    Incident on Well 104-B: BOP stack serial number BOP-HYD-5000-08 suffered a ram leak (FAIL-RAMLEAK)
    while drilling Norphlet Formation sandstone at 16,100 ft MD.
    A total of 48.5 hours NPT was logged. Immediate response followed standard SOP-SHUTIN-01 and API-RP-53.
    """

    res = extractor.extract_from_text(sample_text)

    # Check equipment
    serials = [e.serial_number for e in res.equipment]
    assert "BOP-HYD-5000-08" in serials

    # Check failure modes
    failures = [f.code for f in res.failure_modes]
    assert "FAIL-RAMLEAK" in failures

    # Check SOPs
    sop_ids = [s.sop_id for s in res.sops]
    assert "SOP-SHUTIN-01" in sop_ids or any("SHUTIN" in s for s in sop_ids)
    assert any("API-RP-53" in s for s in sop_ids)

    # Check formation
    formations = [f.name for f in res.formations]
    assert any("Norphlet" in f for f in formations)

    # Check relationships
    assert len(res.relationships) > 0
