"""Tests for domain-aware chunking."""

from langchain_core.documents import Document
from src.ingestion.chunker import WellboreDomainChunker


def test_chunker_section_detection():
    chunker = WellboreDomainChunker(chunk_size=120, chunk_overlap=20)
    raw_text = """
## SECTION 1: DRILLING SUMMARY
Well 104-B is drilling ahead at 10,500 ft with mud motor MM-775-A1.

## SECTION 2: FORMATION TOPS
Eagle Ford Shale encountered at 8,500 ft MD with 13.5 ppg pore pressure.
"""
    doc = Document(page_content=raw_text, metadata={"doc_id": "test_doc", "page": 1})
    chunks = chunker.chunk_documents([doc])

    assert len(chunks) >= 2
    assert chunks[0].doc_id == "test_doc"
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata["page"] == 1
