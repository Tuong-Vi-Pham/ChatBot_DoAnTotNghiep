import logging
from typing import Any, Dict, List, Optional
from src.agent.priority.priority_model import PriorityAnalysisResult, RankedTicket
from src.agent.priority.priority_rules import PriorityRuleEvaluator

logger = logging.getLogger(__name__)


class PriorityRanker:
    """
    Candidate ticket filter and multi-ticket priority ranker.
    Applies a 7-tier deterministic tie-breaking policy to rank tickets reproducibly.
    """

    @classmethod
    def filter_active_candidates(cls, tickets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes completed ('Done') and 'Cancelled' tickets from active priority ranking.
        """
        active: List[Dict[str, Any]] = []
        for t in tickets:
            status = str(t.get("status", "")).strip().lower()
            if status not in ("done", "completed", "cancelled", "canceled"):
                active.append(t)
        return active

    @classmethod
    def rank_tickets(
        cls,
        tickets: List[Dict[str, Any]],
        crm_evidence: Optional[List[Dict[str, Any]]] = None,
        tech_evidence: Optional[List[Dict[str, Any]]] = None
    ) -> List[RankedTicket]:
        """
        Ranks candidate tickets using multi-factor evaluation and deterministic tie-breaking.
        """
        active_tickets = cls.filter_active_candidates(tickets)
        if not active_tickets:
            logger.info("[PriorityRanker] No active candidate tickets found for ranking.")
            return []

        analyzed: List[tuple[Dict[str, Any], PriorityAnalysisResult]] = []
        for t in active_tickets:
            analysis = PriorityRuleEvaluator.evaluate_ticket(
                ticket=t,
                all_tickets=active_tickets,
                crm_evidence=crm_evidence,
                tech_evidence=tech_evidence
            )
            analyzed.append((t, analysis))

        # Sort using 7-Tier Deterministic Tie-Breaking Policy
        def tie_break_key(item: tuple[Dict[str, Any], PriorityAnalysisResult]):
            t, a = item
            dep = a.factors["DEPENDENCY"].score
            rel = a.factors["RELEASE"].score
            biz = a.factors["BUSINESS"].score
            sch = a.factors["SCHEDULE"].score
            tec = a.factors["TECHNICAL"].score
            cur = a.factors["CURRENT_PRIORITY"].score
            
            # Story Point effort: smaller effort gets slight preference when scores are tied
            sp = t.get("story_points")
            effort_penalty = float(sp) if sp is not None else 5.0

            return (
                -a.weighted_score,    # 0. Primary Weighted Score (descending)
                -dep,                 # 1. Blocking / Dependency Impact (descending)
                -rel,                 # 2. Release Impact (descending)
                -biz,                 # 3. Business Impact (descending)
                -sch,                 # 4. Schedule Urgency (descending)
                -tec,                 # 5. Technical Impact (descending)
                -cur,                 # 6. Current Priority (descending)
                effort_penalty,       # 7. Smaller Story Points effort (ascending)
                t.get("ticket_id", "")# 8. Ticket ID tie-breaker (alphabetical)
            )

        sorted_items = sorted(analyzed, key=tie_break_key)

        ranked: List[RankedTicket] = []
        for idx, (t, a) in enumerate(sorted_items, start=1):
            rt = RankedTicket(
                rank=idx,
                ticket_id=a.ticket_id,
                ticket_name=a.ticket_name,
                current_priority=a.current_priority,
                recommended_priority=a.recommended_priority,
                weighted_score=a.weighted_score,
                confidence=a.confidence,
                summary_reason=a.summary_reason,
                differs_from_current=a.differs_from_current,
                analysis=a
            )
            ranked.append(rt)

        logger.info(f"[PriorityRanker] Successfully ranked {len(ranked)} candidate tickets.")
        return ranked
