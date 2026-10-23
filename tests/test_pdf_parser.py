"""Tests for technical document and PDF parser."""

from pathlib import Path
from src.ingestion.pdf_parser import TechnicalDocumentParser


def test_parse_text_file(tmp_path: Path):
    parser = TechnicalDocumentParser()
    sample_file = tmp_path / "sample_sop.txt"
    sample_file.write_text("SOP-SP-774: Drillstring Packoff Remediation in shale.", encoding="utf-8")

    docs = parser.parse_file(sample_file, doc_id="sop_doc")
    assert len(docs) == 1
    assert "SOP-SP-774" in docs[0].page_content
    assert docs[0].metadata["doc_id"] == "sop_doc"


def test_parse_sample_pdf():
    parser = TechnicalDocumentParser()
    pdf_path = Path("data/sample_docs/sample_bop_incident.pdf")
    if pdf_path.exists():
        docs = parser.parse_file(pdf_path)
        assert len(docs) >= 1
        assert "BOP-HYD-5000-08" in docs[0].page_content
        assert docs[0].metadata["page"] == 1
