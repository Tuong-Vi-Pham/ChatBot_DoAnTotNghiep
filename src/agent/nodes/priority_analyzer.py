import logging
from typing import Any, Dict, List
from src.agent.state import AgentState
from src.agent.priority import PriorityRuleEvaluator, PriorityRanker, PriorityVerifier

logger = logging.getLogger(__name__)


def _get_evidence_source(e: Any) -> str:
    if hasattr(e, "source"):
        return getattr(e, "source") or ""
    elif isinstance(e, dict):
        return e.get("source") or ""
    return ""


class PriorityAnalyzerNode:
    """
    LangGraph Node executing Ticket Priority Analysis and Candidate Ranking.
    Computes recommended priority and ranking without replacing original Lark Daily_Task priority.
    """

    @classmethod
    def execute(cls, state: AgentState) -> AgentState:
        intent = state.get("intent", "")
        state["priority_ranking"] = state.get("priority_ranking") or []

        if intent not in ("PRIORITY_ANALYSIS", "TICKET_ANALYSIS", "PRIORITY_APPROVAL_REQUEST"):
            logger.info(f"[PriorityAnalyzerNode] Intent '{intent}' does not require priority analysis. Skipping.")
            return state

        ticket_context: List[Dict[str, Any]] = state.get("ticket_context") or []
        project_context: List[Any] = state.get("project_context") or []

        # Separate CRM and Tech_Team evidence
        crm_evidence = [e for e in project_context if _get_evidence_source(e) in ("CRM", "FAQ")]
        tech_evidence = [e for e in project_context if _get_evidence_source(e) == "Tech_Team"]

        # Single Ticket Analysis vs Multi-ticket Ranking
        if len(ticket_context) == 1:
            single_t = ticket_context[0]
            analysis = PriorityRuleEvaluator.evaluate_ticket(
                ticket=single_t,
                all_tickets=ticket_context,
                crm_evidence=crm_evidence,
                tech_evidence=tech_evidence
            )
            state["priority_analysis"] = analysis.to_dict()

            ranked = PriorityRanker.rank_tickets(
                tickets=ticket_context,
                crm_evidence=crm_evidence,
                tech_evidence=tech_evidence
            )
            state["priority_ranking"] = [r.to_dict() for r in ranked]
        elif len(ticket_context) > 1:
            ranked = PriorityRanker.rank_tickets(
                tickets=ticket_context,
                crm_evidence=crm_evidence,
                tech_evidence=tech_evidence
            )
            state["priority_ranking"] = [r.to_dict() for r in ranked]
            if ranked:
                state["priority_analysis"] = ranked[0].analysis.to_dict()

        logger.info(
            f"[PriorityAnalyzerNode] Priority analysis completed: "
            f"{len(state.get('priority_ranking', []))} candidate tickets ranked."
        )
        return state
