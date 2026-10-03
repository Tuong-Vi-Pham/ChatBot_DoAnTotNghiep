import logging
from typing import Any, Dict, List
from src.agent.state import AgentState
from src.agent.config import DOCUMENT_SIMILARITY_THRESHOLD, MAX_RETRIEVAL_ITERATIONS

logger = logging.getLogger(__name__)


class EvidenceAssessorNode:
    """
    Evaluates evidence quality and determines whether sufficient evidence is present to fulfill query.
    """

    @classmethod
    def execute(cls, state: AgentState) -> AgentState:
        intent = state.get("intent", "GENERAL_RAG")
        retrieval_results = state.get("retrieval_results", [])
        ticket_context = state.get("ticket_context", [])

        max_score = 0.0
        for ev in retrieval_results:
            if hasattr(ev, "relevance_score"):
                score = float(getattr(ev, "relevance_score", 0.0))
            elif isinstance(ev, dict):
                score = float(ev.get("relevance_score", 0.0))
            else:
                score = 0.0
            if score > max_score:
                max_score = score

        # If ticket query and ticket data was found, quality is high
        if intent in ("TICKET_QUERY", "TICKET_ANALYSIS") and ticket_context:
            evidence_quality = max(max_score, 0.85)
        elif intent == "PRIORITY_ANALYSIS" and ticket_context:
            evidence_quality = max(max_score, 0.85 if max_score >= 0.30 else 0.75)
        else:
            evidence_quality = max_score

        state["evidence_quality"] = round(evidence_quality, 4)
        state["confidence"] = round(evidence_quality, 4)

        logger.info(
            f"[EvidenceAssessorNode] Evidence Quality: {state['evidence_quality']} "
            f"(Retrieval items: {len(retrieval_results)}, Ticket items: {len(ticket_context)})"
        )
        return state

    @classmethod
    def is_evidence_sufficient(cls, state: AgentState) -> bool:
        """Conditional edge evaluation logic."""
        intent = state.get("intent", "")
        quality = state.get("evidence_quality", 0.0)
        iter_count = state.get("iteration_count", 0)

        if intent in ("CLARIFICATION_REQUIRED", "FAQ"):
            return True

        if intent in ("TICKET_QUERY", "TICKET_ANALYSIS") and state.get("ticket_context"):
            return True

        if intent == "PRIORITY_ANALYSIS":
            if state.get("ticket_context"):
                return True
            if iter_count >= MAX_RETRIEVAL_ITERATIONS:
                logger.warning(f"[EvidenceAssessorNode] Bounded retrieval limit reached ({iter_count}/{MAX_RETRIEVAL_ITERATIONS}) for PRIORITY_ANALYSIS without tickets.")
                return True
            return False

        if quality >= DOCUMENT_SIMILARITY_THRESHOLD:
            return True

        if iter_count >= MAX_RETRIEVAL_ITERATIONS:
            logger.warning(f"[EvidenceAssessorNode] Bounded retrieval limit reached ({iter_count}/{MAX_RETRIEVAL_ITERATIONS}). Proceeding to verification.")
            return True

        return False


class QueryRewriterNode:
    """
    Rewrites user retrieval query for query expansion during retrieval retry loops.
    """

    @classmethod
    def execute(cls, state: AgentState) -> AgentState:
        orig_query = state.get("retrieval_query") or state.get("user_query", "")
        
        # Simple domain query expansion
        rewritten = f"{orig_query} overview specification documentation architecture"
        state["retrieval_query"] = rewritten
        
        logger.info(f"[QueryRewriterNode] Rewrote query for retry: '{orig_query}' -> '{rewritten}'")
        return state
