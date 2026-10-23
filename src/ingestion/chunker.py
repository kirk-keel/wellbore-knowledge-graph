"""Domain-aware chunker for wellbore engineering documents."""

import hashlib
import logging
import re
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.models.domain import DocumentChunk

logger = logging.getLogger(__name__)


class WellboreDomainChunker:
    """
    Splits engineering reports, IADC procedures, and well programs
    while respecting operational boundaries (sections, SOP steps, equipment lists).
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Domain separators prioritized by semantic coherence
        self.separators = [
            "\n## ",
            "\n### ",
            "\nSECTION ",
            "\nSection ",
            "\nAPPENDIX ",
            "\n---",
            "\n\n",
            "\nStep ",
            "\n1.",
            "\n2.",
            "\n",
            " ",
            "",
        ]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_documents(self, documents: List[Document], doc_id_override: Optional[str] = None) -> List[DocumentChunk]:
        """Split a list of LangChain documents into domain DocumentChunk instances."""
        chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for doc in documents:
            doc_id = doc_id_override or doc.metadata.get("doc_id", "unknown_doc")
            page = doc.metadata.get("page", 1)

            # Split content using recursive splitter
            raw_splits = self.splitter.split_text(doc.page_content)

            for split_idx, split_text in enumerate(raw_splits):
                clean_text = split_text.strip()
                if not clean_text:
                    continue

                # Generate deterministic chunk identifier
                chunk_hash = hashlib.sha256(f"{doc_id}_{page}_{global_chunk_idx}_{clean_text[:50]}".encode()).hexdigest()[:8]
                chunk_id = f"{doc_id}:c{global_chunk_idx}_{chunk_hash}"

                # Extract potential section title or context if present
                section_title = self._detect_section_title(clean_text)

                chunk_meta = doc.metadata.copy()
                chunk_meta["chunk_index"] = global_chunk_idx
                chunk_meta["split_index"] = split_idx
                chunk_meta["section_title"] = section_title

                chunk = DocumentChunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    chunk_index=global_chunk_idx,
                    page=page,
                    text=clean_text,
                    embedding=None,
                    metadata=chunk_meta,
                )
                chunks.append(chunk)
                global_chunk_idx += 1

        logger.info("Generated %d chunks across %d input documents", len(chunks), len(documents))
        return chunks

    def _detect_section_title(self, text: str) -> Optional[str]:
        """Detect top-level section header in the text block."""
        match = re.search(r"^(?:#{1,3}\s+|SECTION\s+\d+:?\s*|Section\s+\d+:?\s*)([^\n]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        first_line = text.split("\n", 1)[0].strip()
        if len(first_line) < 60 and first_line.isupper() and len(first_line) > 3:
            return first_line
        return None
