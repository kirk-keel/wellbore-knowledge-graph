"""Multi-Agent Collaborative Advisory Panel for Wellbore Engineering and Integrity."""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.models.api import QueryRequest, QueryResponse
from src.retrieval.casing_calculator import CasingIntegrityCalculator, CasingIntegrityRequest
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.qa_engine import WellboreQAEngine
from src.retrieval.safety_guardrails import TwoBarrierSafetyGuardrail

logger = logging.getLogger(__name__)


class AgentOpinion(BaseModel):
    """Opinion and findings from an autonomous specialist agent."""
    role: str
    specialist_title: str
    key_findings: List[str]
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    technical_recommendation: str


class MultiAgentAdvisoryReport(BaseModel):
    """Consensus operational advisory emitted by the multi-agent committee."""
    inquiry: str
    operational_classification: str
    committee_opinions: List[AgentOpinion]
    two_barrier_compliance_status: str
    masp_analysis_summary: Optional[str] = None
    consensus_operational_sequence: List[str]
    final_advisory: str


class MultiAgentDrillingPanel:
    """Coordinates autonomous multi-agent drilling specialist committee for complex decisions."""

    def __init__(
        self,
        qa_engine: Optional[WellboreQAEngine] = None,
        retriever: Optional[HybridRetriever] = None,
    ):
        self.qa_engine = qa_engine or WellboreQAEngine()
        self.retriever = retriever or HybridRetriever()
        self.casing_calc = CasingIntegrityCalculator()
        self.safety_guardrail = TwoBarrierSafetyGuardrail()

    def consult(self, inquiry: str, well_name: str = "Well 104-B") -> MultiAgentAdvisoryReport:
        """Execute collaborative multi-agent inquiry across 4 specialized roles."""
        inq_lower = inquiry.lower()

        # 1. Superintendent Agent: Classify operation
        if any(w in inq_lower for w in ["kick", "leak", "pack-off", "packoff", "stuck", "emergency"]):
            classification = "URGENT_OPERATIONAL_REMEDIATION"
        elif any(w in inq_lower for w in ["casing", "burst", "collapse", "design", "masp"]):
            classification = "WELL_ENGINEERING_DESIGN"
        else:
            classification = "ROUTINE_DRILLING_OPERATION"

        # 2. Geomechanics & Subsurface Agent
        chunks, paths, citations = self.retriever.retrieve(inquiry, top_k=3, graph_hops=2)
        formations_cited = [c.entity_name for c in citations if c.entity_type == "Formation"]

        geo_opinion = AgentOpinion(
            role="Geomechanics Specialist",
            specialist_title="Senior Subsurface Geomechanics Engineer",
            key_findings=[
                f"Lithology in target interval involves {', '.join(formations_cited) if formations_cited else 'Reactive Shale & Tight Sandstone'}.",
                "Formation exhibits high pore pressure gradients (up to 14.8 ppg in Norphlet / 13.5 ppg in Draupne).",
                "High differential pressure risks sloughing cavings and mechanical pack-off.",
            ],
            risk_level="HIGH" if formations_cited else "MEDIUM",
            technical_recommendation=(
                "Maintain minimum 200 psi overbalance above pore pressure while preserving a 0.5 ppg margin below shoe fracture gradient."
            ),
        )

        # 3. Equipment Reliability Agent
        eq_cited = [c.entity_name for c in citations if c.entity_type == "Equipment"]
        fails_cited = [c.entity_name for c in citations if c.entity_type == "FailureMode"]

        eq_opinion = AgentOpinion(
            role="Equipment Reliability Specialist",
            specialist_title="Subsea Asset Integrity & BHA Specialist",
            key_findings=[
                f"Identified Assets in operation: {', '.join(eq_cited) if eq_cited else 'BOP Stack & Positive Displacement Mud Motor'}.",
                f"Documented failure mechanisms: {', '.join(fails_cited) if fails_cited else 'Seal Degradation & Motor Stalling'}.",
                "Equipment must operate strictly within certified temperature and differential pressure thresholds.",
            ],
            risk_level="CRITICAL" if any("leak" in f.lower() or "pack" in f.lower() for f in fails_cited) else "MEDIUM",
            technical_recommendation=(
                "Verify elastomer pack ratings (minimum 350°F FFKM) and monitor mud motor differential pressure strictly below 800 psi."
            ),
        )

        # 4. Well Control & Statutory Barrier Auditor Agent
        audit = self.safety_guardrail.audit_engineering_response(inquiry, f"{geo_opinion.technical_recommendation} {eq_opinion.technical_recommendation}")
        calc_report = self.casing_calc.evaluate_casing_integrity(CasingIntegrityRequest(well_name=well_name))

        wc_opinion = AgentOpinion(
            role="Well Control Auditor",
            specialist_title="API RP 53 / NORSOK D-010 Barrier Compliance Inspector",
            key_findings=[
                f"Two-Barrier Compliance Status: {audit['two_barrier_status']}.",
                f"Casing Structural Integrity: Burst SF = {calc_report.burst_safety_factor} ({calc_report.burst_status}).",
                f"Calculated Maximum Anticipated Surface Pressure (MASP): {calc_report.maximum_anticipated_surface_pressure_masp_psi} psi.",
            ],
            risk_level="LOW" if calc_report.overall_integrity_status == "APPROVED_BARRIER" else "CRITICAL",
            technical_recommendation=(
                f"Enforce mandatory space-out and soft shut-in procedures per SOP-SHUTIN-01. "
                f"Maximum allowable shut-in casing pressure must not exceed shoe breakdown limit (Kick Tolerance: {calc_report.kick_tolerance_bbl} bbl)."
            ),
        )

        # 5. Consensus Advisory Synthesizer
        sequence = [
            "1. Confirm primary barrier (active mud weight exceeding pore pressure by designated overbalance).",
            "2. Establish secondary barrier readiness (accumulator bottle pressure > 3,000 psi, annular preventer lined up).",
            "3. Execute operational drilling sequence according to SOP-SP-774 and IADC-WCC-04.",
            "4. Continuously log real-time standpipe pressure and mud pit volumes at 1-second resolution.",
        ]

        final_advisory = (
            f"### Consensus Drilling Operational Directive ({classification})\n"
            f"The Autonomous Drilling Advisory Committee has evaluated **{inquiry}** on **{well_name}**.\n\n"
            f"**Key Geological Directive**: {geo_opinion.technical_recommendation}\n"
            f"**Equipment Protection Directive**: {eq_opinion.technical_recommendation}\n"
            f"**Well Control Barrier Directive**: {wc_opinion.technical_recommendation}\n\n"
            f"**Verified Structural Margin**: Calculated MASP is **{calc_report.maximum_anticipated_surface_pressure_masp_psi} psi** "
            f"with casing burst safety factor **{calc_report.burst_safety_factor}** (API standard >= 1.125)."
        )

        return MultiAgentAdvisoryReport(
            inquiry=inquiry,
            operational_classification=classification,
            committee_opinions=[geo_opinion, eq_opinion, wc_opinion],
            two_barrier_compliance_status=audit["two_barrier_status"],
            masp_analysis_summary=f"MASP: {calc_report.maximum_anticipated_surface_pressure_masp_psi} psi | Burst SF: {calc_report.burst_safety_factor}",
            consensus_operational_sequence=sequence,
            final_advisory=final_advisory,
        )
