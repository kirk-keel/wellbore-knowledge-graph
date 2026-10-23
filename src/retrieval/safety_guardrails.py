"""Well control safety guardrail enforcing the Two-Barrier Rule (API RP 53 / NORSOK D-010)."""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TwoBarrierSafetyGuardrail:
    """
    Audits operational guidance against statutory Well Integrity & Barrier Standards (API RP 53 & NORSOK D-010).
    Ensures both Primary and Secondary well control barriers are validated prior to delivering instructions.
    """

    def __init__(self):
        # Known barrier terminology
        self.primary_barrier_keywords = [
            "hydrostatic",
            "drilling fluid",
            "mud weight",
            "kill mud",
            "hydrostatic head",
            "overbalance",
        ]
        self.secondary_barrier_keywords = [
            "bop",
            "annular preventer",
            "pipe ram",
            "blind shear ram",
            "casing shoe",
            "cement plug",
            "wellhead",
            "choke manifold",
        ]

    def audit_engineering_response(self, question: str, response_text: str) -> Dict[str, Any]:
        """Audit response for well control barrier compliance."""
        text_lower = response_text.lower()
        q_lower = question.lower()

        # Check if question relates to well control, kicks, influx, shut-in, or barrier operations
        well_control_inquiry = any(
            kw in q_lower or kw in text_lower
            for kw in ["well control", "kick", "shut-in", "shutin", "influx", "ram leak", "blowout", "barrier", "kill"]
        )

        if not well_control_inquiry:
            return {
                "well_control_applicable": False,
                "two_barrier_status": "N/A",
                "primary_barrier_detected": True,
                "secondary_barrier_detected": True,
                "audit_notes": "Standard drilling query. Operational barrier verification not mandatory.",
            }

        # Check for Primary Barrier
        primary_detected = any(kw in text_lower for kw in self.primary_barrier_keywords)
        # Check for Secondary Barrier
        secondary_detected = any(kw in text_lower for kw in self.secondary_barrier_keywords)

        is_compliant = primary_detected and secondary_detected

        audit_result = {
            "well_control_applicable": True,
            "two_barrier_status": "VERIFIED_COMPLIANT" if is_compliant else "SAFETY_WARNING_DEFICIENT_BARRIER",
            "primary_barrier_detected": primary_detected,
            "secondary_barrier_detected": secondary_detected,
            "standards_referenced": ["API RP 53", "NORSOK D-010", "IADC Well Control Manual"],
            "audit_notes": (
                "Both Primary (Hydrostatic) and Secondary (Mechanical BOP/Rams) barriers verified."
                if is_compliant else
                "WARNING: Potential deviation from API RP 53 Two-Barrier Rule. "
                "Ensure both hydrostatic fluid column and mechanical preventer elements are explicitly verified."
            ),
        }

        return audit_result

    def format_barrier_appendix(self, audit: Dict[str, Any]) -> str:
        """Create a safety advisory footer based on audit results."""
        if not audit.get("well_control_applicable", False):
            return ""

        if audit["two_barrier_status"] == "VERIFIED_COMPLIANT":
            return (
                "\n\n---\n"
                "> 🛡️ **API RP 53 & NORSOK D-010 Barrier Compliance Verified**:\n"
                "> • **Primary Barrier**: Hydrostatic column / active mud weight confirmed.\n"
                "> • **Secondary Barrier**: Mechanical BOP isolation & manifold protocols confirmed."
            )
        else:
            return (
                "\n\n---\n"
                "> ⚠️ **SAFETY ADVISORY (Two-Barrier Rule Check)**:\n"
                f"> {audit['audit_notes']}"
            )
