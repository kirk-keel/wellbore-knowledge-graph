"""Document ingestion parsers for engineering PDFs, IADC standards, and technical reports."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class TechnicalDocumentParser:
    """Parses technical engineering documents (PDF, TXT, Markdown) into structured LangChain Documents."""

    def __init__(self):
        pass

    def parse_file(self, file_path: str | Path, doc_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Parse a document file based on extension and return LangChain Documents."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        meta = metadata.copy() if metadata else {}
        meta["filename"] = path.name
        meta["extension"] = path.suffix.lower()
        meta["doc_id"] = doc_id or path.stem

        if path.suffix.lower() == ".pdf":
            return self._parse_pdf(path, meta)
        elif path.suffix.lower() in [".txt", ".md", ".log", ".csv"]:
            return self._parse_text(path, meta)
        else:
            # Fallback to plain text reading
            return self._parse_text(path, meta)

    def parse_raw_text(self, text: str, title: str, doc_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Convert in-memory text string into LangChain Document with rich metadata."""
        meta = metadata.copy() if metadata else {}
        meta["title"] = title
        meta["doc_id"] = doc_id or f"doc_{abs(hash(title))}"
        meta["source"] = meta.get("source", "raw_text_stream")
        meta["page"] = 1
        return [Document(page_content=text.strip(), metadata=meta)]

    def _parse_pdf(self, path: Path, base_metadata: Dict[str, Any]) -> List[Document]:
        """Parse PDF document using PyPDF, preserving page numbers and document structural sections."""
        docs: List[Document] = []
        try:
            reader = PdfReader(str(path))
            total_pages = len(reader.pages)
            base_metadata["total_pages"] = total_pages

            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue

                page_meta = base_metadata.copy()
                page_meta["page"] = page_idx + 1
                page_meta["source"] = f"{path.name} (Page {page_idx + 1})"

                docs.append(Document(page_content=page_text.strip(), metadata=page_meta))

            logger.info("Successfully parsed PDF %s (%d pages with text)", path.name, len(docs))
            if not docs:
                # If no text was extracted (e.g. image-only PDF), create a placeholder
                docs.append(Document(
                    page_content=f"Document {path.name} contains no machine-readable text.",
                    metadata=base_metadata
                ))
            return docs

        except Exception as e:
            logger.error("Failed to parse PDF %s: %s", path, e)
            raise RuntimeError(f"Error parsing PDF file {path.name}: {e}") from e

    def _parse_text(self, path: Path, base_metadata: Dict[str, Any]) -> List[Document]:
        """Parse plain text or markdown file."""
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            meta = base_metadata.copy()
            meta["page"] = 1
            meta["source"] = path.name
            return [Document(page_content=content.strip(), metadata=meta)]
        except Exception as e:
            logger.error("Failed to read text file %s: %s", path, e)
            raise RuntimeError(f"Error reading file {path.name}: {e}") from e
