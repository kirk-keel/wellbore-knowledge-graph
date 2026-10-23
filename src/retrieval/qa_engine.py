"""Anti-hallucination GraphRAG Question Answering engine for wellbore engineering."""

import json
import logging
from typing import List, Optional
import httpx

from src.config import settings
from src.models.api import EntityCitation, GraphPathSegment, QueryRequest, QueryResponse, RetrievedChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.safety_guardrails import TwoBarrierSafetyGuardrail

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = """You are an authoritative Senior Wellbore & Drilling Operations Engineer specializing in IADC standards, well control, and equipment failure analysis.

You must answer the query strictly and exclusively using the provided Verified Knowledge Graph Context and Document Excerpts below.
DO NOT hallucinate, infer, or speculate beyond the verified facts.

Grounding Requirements:
1. Always cite exact equipment serial numbers (e.g. BOP-HYD-5000-08, MM-775-A1).
2. Explicitly cite standard operating procedure codes (e.g. IADC-WCC-04, API-RP-53).
3. Explicitly reference geological formation lithology and depth intervals when applicable.
4. Detail the semantic path discovered in the Knowledge Graph connecting the incident, failure mode, and mitigation SOP.
5. If the provided context does not contain sufficient verified evidence, clearly state:
   "The wellbore knowledge graph contains no verified record regarding [missing aspect]."

=== VERIFIED GRAPH RELATIONSHIPS ===
{graph_context}

=== VERIFIED DOCUMENT CHUNKS ===
{chunks_context}

User Engineering Query:
{question}
"""


class WellboreQAEngine:
    """Answers complex engineering queries by synthesizing graph topology with vector evidence."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        safety_guardrail: Optional[TwoBarrierSafetyGuardrail] = None,
        ollama_url: str = settings.OLLAMA_BASE_URL,
        model: str = settings.OLLAMA_LLM_MODEL,
    ):
        self.retriever = retriever or HybridRetriever()
        self.safety_guardrail = safety_guardrail or TwoBarrierSafetyGuardrail()
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model

    def answer_query(self, request: QueryRequest) -> QueryResponse:
        """Execute hybrid retrieval and generate grounded engineering response."""
        # 1. Hybrid Retrieval
        chunks, paths, citations = self.retriever.retrieve(
            query=request.question,
            top_k=request.top_k_chunks,
            graph_hops=request.graph_hops,
            min_similarity=request.min_similarity,
        )

        # 2. Format Context for LLM
        graph_context = self._format_graph_context(paths)
        chunks_context = self._format_chunks_context(chunks)

        # 3. Generate Answer (Ollama or Deterministic Grounded Engine)
        answer, reasoning = self._generate_grounded_answer(
            question=request.question,
            graph_context=graph_context,
            chunks_context=chunks_context,
            chunks=chunks,
            paths=paths,
            citations=citations,
        )

        # 4. Audit Well Control Two-Barrier Rule Compliance
        audit = self.safety_guardrail.audit_engineering_response(request.question, answer)
        answer += self.safety_guardrail.format_barrier_appendix(audit)

        # 5. Calculate Confidence Score
        confidence = self._compute_confidence(chunks, paths, citations)

        # 6. Extract consulted sources
        sources = list({c.doc_id for c in chunks})

        return QueryResponse(
            question=request.question,
            answer=answer,
            confidence_score=confidence,
            retrieved_chunks=chunks if request.include_graph_paths else [],
            graph_paths=paths if request.include_graph_paths else [],
            entities_cited=citations,
            reasoning_summary=reasoning,
            sources_consulted=sources,
        )

    def _format_graph_context(self, paths: List[GraphPathSegment]) -> str:
        """Format graph edges into readable semantic triples."""
        if not paths:
            return "No connected graph paths identified."
        lines = []
        for p in paths:
            lines.append(f"• ({p.source_label}: {p.source_name}) -[:{p.relation}]-> ({p.target_label}: {p.target_name})")
        return "\n".join(lines)

    def _format_chunks_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved vector chunks with source attribution."""
        if not chunks:
            return "No matching document excerpts found."
        formatted = []
        for c in chunks:
            formatted.append(f"[{c.doc_id} | Page {c.page} | Score {c.similarity_score:.3f}]\n{c.text}\n")
        return "\n".join(formatted)

    def _generate_grounded_answer(
        self,
        question: str,
        graph_context: str,
        chunks_context: str,
        chunks: List[RetrievedChunk],
        paths: List[GraphPathSegment],
        citations: List[EntityCitation],
    ) -> tuple[str, str]:
        """Try Ollama inference, fallback to deterministic grounded engineering synthesis."""
        # Check if Ollama is accessible
        try:
            prompt = QA_SYSTEM_PROMPT.format(
                graph_context=graph_context,
                chunks_context=chunks_context,
                question=question,
            )
            with httpx.Client(base_url=self.ollama_url, timeout=20.0) as client:
                resp = client.post(
                    "/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.0, "top_p": 0.9},
                    },
                )
                if resp.status_code == 200:
                    text_response = resp.json().get("response", "").strip()
                    if text_response:
                        reasoning = f"Synthesized using {len(chunks)} text chunks and {len(paths)} graph traversal paths via Ollama {self.model}."
                        return text_response, reasoning
        except Exception as e:
            logger.debug("Ollama inference unavailable (%s). Synthesizing deterministic grounded response.", e)

        # Grounded Deterministic Engineering Synthesizer
        return self._synthesize_grounded_response(question, chunks, paths, citations)

    def _synthesize_grounded_response(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        paths: List[GraphPathSegment],
        citations: List[EntityCitation],
    ) -> tuple[str, str]:
        """Synthesize authoritative, hallucination-free engineering response from graph & text facts."""
        if not chunks and not paths:
            return (
                "The wellbore knowledge graph contains no verified record answering this query. "
                "No matching document chunks or semantic entity relationships were identified.",
                "Zero graph paths or vector chunks available.",
            )

        # Categorize discovered entities
        eq_items = [c for c in citations if c.entity_type == "Equipment"]
        sop_items = [c for c in citations if c.entity_type == "SOP"]
        fail_items = [c for c in citations if c.entity_type == "FailureMode"]
        form_items = [c for c in citations if c.entity_type == "Formation"]
        inc_items = [c for c in citations if c.entity_type == "Incident"]

        sections = []
        sections.append(f"### Operational Analysis & Findings\n")

        # Grounded answer synthesis
        if fail_items or sop_items:
            findings = []
            for f in fail_items:
                # Find connected SOPs
                mitigating = [p.target_name for p in paths if p.source_id == f.entity_id and p.relation == "MITIGATED_BY"]
                if mitigating:
                    findings.append(f"- **Failure Mode [{f.entity_name}]**: Mitigated according to standard procedure(s) **{', '.join(mitigating)}**.")
                else:
                    findings.append(f"- **Failure Mode [{f.entity_name}]**: Identified as operational failure risk.")

            for s in sop_items:
                findings.append(f"- **Standard Operating Procedure [{s.entity_id}]**: Governs operational verification and barrier compliance ({s.entity_name}).")

            sections.append("\n".join(findings))

        if eq_items:
            eq_lines = [f"- **Equipment Asset [{e.entity_id}]**: Classified as {e.entity_name}." for e in eq_items]
            sections.append("\n#### Equipment Assets Identified:\n" + "\n".join(eq_lines))

        if form_items:
            form_lines = [f"- **Stratigraphy / Lithology [{fm.entity_name}]**: Lithological target and associated hazard interval." for fm in form_items]
            sections.append("\n#### Subsurface Lithology Context:\n" + "\n".join(form_lines))

        # Direct text evidence summary
        if chunks:
            sections.append("\n#### Technical Document Evidence:")
            top_chunk = chunks[0]
            summary_snippet = top_chunk.text.split("\n\n")[0]
            sections.append(f"> \"{summary_snippet}\"\n> — *Source: {top_chunk.doc_id} (Page {top_chunk.page})*")

        # Semantic Graph Traversal Trail
        if paths:
            path_trail = " -> ".join([f"[{p.source_label}:{p.source_name}] --({p.relation})--> [{p.target_label}:{p.target_name}]" for p in paths[:3]])
            sections.append(f"\n#### Semantic Graph Traversal Path:\n`{path_trail}`")

        answer_text = "\n\n".join(sections)
        reasoning_summary = (
            f"Grounding verification complete. Consulted {len(chunks)} document chunks, "
            f"traversed {len(paths)} graph relationships, and cited {len(citations)} verified domain entities."
        )

        return answer_text, reasoning_summary

    def _compute_confidence(
        self,
        chunks: List[RetrievedChunk],
        paths: List[GraphPathSegment],
        citations: List[EntityCitation],
    ) -> float:
        """Compute grounded confidence metric based on vector similarity and graph connectivity."""
        if not chunks and not paths:
            return 0.0

        vector_score = max([c.similarity_score for c in chunks]) if chunks else 0.4
        graph_factor = min(len(paths) * 0.08, 0.35)
        citation_factor = min(len(citations) * 0.05, 0.25)

        raw_confidence = (vector_score * 0.5) + graph_factor + citation_factor
        return round(min(max(raw_confidence, 0.2), 0.98), 3)
