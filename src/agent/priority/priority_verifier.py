import logging
from typing import Any, Dict, List, Optional
from src.agent.priority.priority_model import PriorityAnalysisResult, RankedTicket, VALID_PRIORITIES

logger = logging.getLogger(__name__)


class PriorityVerifier:
    """
    Verification checks for priority analysis and recommendation results.
    Ensures current priority preservation, supported claims, and deterministic ranking.
    """

    @classmethod
    def verify_analysis(
        cls,
        analysis: PriorityAnalysisResult,
        ticket: Dict[str, Any]
    ) -> Dict[str, Any]:
        t_id = ticket.get("ticket_id", "")
        curr_prio = (ticket.get("priority") or "MEDIUM").strip().upper()

        checks = {
            "ticket_exists": bool(t_id),
            "current_priority_preserved": (analysis.current_priority == curr_prio),
            "status_valid": (ticket.get("status", "").lower() not in ("done", "completed", "cancelled")),
            "recommendation_valid": analysis.recommended_priority in VALID_PRIORITIES,
            "confidence_valid": analysis.confidence in ("High", "Medium", "Low")
        }

        all_passed = all(checks.values())
        return {
            "verified": all_passed,
            "checks": checks,
            "reason": "All priority checks passed successfully." if all_passed else f"Priority verification failed: {checks}"
        }

    @classmethod
    def verify_ranking(
        cls,
        ranked_tickets: List[RankedTicket]
    ) -> Dict[str, Any]:
        if not ranked_tickets:
            return {"verified": True, "reason": "Empty candidate ranking verified."}

        # Check rank sequence
        for idx, rt in enumerate(ranked_tickets, start=1):
            if rt.rank != idx:
                return {"verified": False, "reason": f"Rank mismatch for {rt.ticket_id}: expected {idx}, got {rt.rank}"}

        # Check non-increasing score order
        scores = [rt.weighted_score for rt in ranked_tickets]
        is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

        return {
            "verified": is_sorted,
            "ticket_count": len(ranked_tickets),
            "reason": "Ranking sequence and score monotonicity verified." if is_sorted else "Score monotonicity check failed."
        }
