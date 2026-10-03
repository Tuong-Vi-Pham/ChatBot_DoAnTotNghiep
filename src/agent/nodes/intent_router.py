import re
import logging
from typing import Any, Dict
from src.agent.state import AgentState

logger = logging.getLogger(__name__)


class IntentRouterNode:
    """
    Categorizes incoming user queries into explicit operational intents and determines source scope.
    Deterministic rule routing with pattern matching.
    """

    @classmethod
    def execute(cls, state: AgentState) -> AgentState:
        query = state.get("user_query", "").strip()
        query_lower = query.lower()

        intent = "GENERAL_RAG"
        scope = "BOTH"

        # 0a. Check for Priority Update Execution intent (e.g. "apply APR-001", "execute APR-001", "apply approved recommendation APR-001")
        if re.search(r'\b(apply|execute)\s+(approved\s+)?apr-\d+\b', query_lower) or any(k in query_lower for k in ["apply apr-", "execute apr-", "apply approved apr-"]):
            intent = "PRIORITY_UPDATE_EXECUTION"
            scope = "NONE"
        # 0b. Check for explicit Approval Decision intent with APR-xxx (e.g. "approve APR-001", "reject APR-001")
        elif re.search(r'\b(approve|reject|cancel)\s+apr-\d+\b', query_lower):
            intent = "PRIORITY_APPROVAL_DECISION"
            scope = "NONE"
        # 0c. Check for Approval Request / Conversational Approval Decision intent
        elif (
            # Explicit creation phrases
            any(k in query_lower for k in [
                "apply recommended priority", "apply priority for task-", "request approval",
                "create approval request", "update the priority of", "update priority of"
            ])
            # Natural approval phrases
            or re.search(r'\b(approve(\s+this|\s+the)?\s+recommendation(\s+for\s+[a-z]+-\d+)?)\b', query_lower)
            or re.search(r'\b(i\s+approve|approved)\b', query_lower)
            or re.search(r'\b(proceed\s+with\s+(the\s+)?update|go\s+ahead\s+with\s+(the\s+)?update)\b', query_lower)
            or re.search(r'\byes,?\s*(update|proceed)\b', query_lower)
            or re.search(r'\b(confirm(\s+(the\s+)?update)?)\b', query_lower)
            or re.search(r'\bapprove\s+[a-z]+-\d+\b', query_lower)
            or query_lower in ["approve", "approved", "i approve", "yes", "yes, update", "proceed with the update", "go ahead with the update", "yes, proceed", "confirm", "confirm update"]
            # Natural rejection phrases
            or re.search(r'\b(reject(\s+this|\s+the)?\s+recommendation|i\s+reject(\s+this)?\s+recommendation)\b', query_lower)
            or re.search(r'\b(no,?\s*don\'?t\s+update( it)?|keep\s+(the\s+)?current\s+priority)\b', query_lower)
            or re.search(r'\bdo\s+not\s+proceed\b', query_lower)
            or query_lower in ["reject", "no", "do not proceed", "no don't update it", "no dont update it", "no, don't update it"]
            # Natural cancellation phrases
            or re.search(r'\bcancel(\s+(the\s+)?update|\s+approval)?\b', query_lower)
            or query_lower in ["cancel"]
        ):
            intent = "PRIORITY_APPROVAL_REQUEST"
            scope = "BOTH"
            if any(k in query_lower for k in ["reject", "no", "do not proceed", "don't update", "keep the current"]):
                state["approval_action"] = "REJECT"
            elif "cancel" in query_lower:
                state["approval_action"] = "CANCEL"
            elif any(k in query_lower for k in ["approve", "yes", "proceed", "confirm", "go ahead", "approved"]):
                state["approval_action"] = "APPROVE"
            else:
                state["approval_action"] = "REQUEST"
        # A context-free request cannot identify a ticket set or a prioritization goal.
        # Keep it in the clarification flow instead of treating every "work first"
        # phrase as a ranking request.
        elif query_lower in ["what should we work on first?", "what should we work on first"]:
            intent = "CLARIFICATION_REQUIRED"
            scope = "NONE"
        # 2. Check for Priority Analysis intent (safe word boundary on recommend)
        elif any(k in query_lower for k in [
            "prioritize", "prioritise", "prioritized", "priority first", "should we do first",
            "work on first", "which unfinished ticket", "what should we work on", "which ticket should",
            "rank", "ranking", "highest priority"
        ]) or re.search(r'\brecommends?\b', query_lower):
            intent = "PRIORITY_ANALYSIS"
            scope = "BOTH"
        # 1. Check for vague/ambiguous queries requiring clarification
        elif query_lower in ["prioritize tickets", "prioritize", "ticket status", "show tickets"]:
            intent = "CLARIFICATION_REQUIRED"
            scope = "NONE"

        # 3. Check for explicit Ticket queries (TASK-xxx, EPIC-xxx, CRM-xxx, BUG-xxx, etc.)
        elif re.search(r'\b(task|epic|bug|sub|crm|story)-\d+\b', query_lower) or any(k in query_lower for k in ["status of task", "status of epic", "status of ticket", "status of crm", "sprint ", "unfinished tickets", "tickets for"]):
            if any(k in query_lower for k in ["analyze", "detail", "context", "explain", "brd", "srs", "architecture"]):
                intent = "TICKET_ANALYSIS"
                scope = "BOTH"
            else:
                intent = "TICKET_QUERY"
                scope = "NONE"
        # 4. Check for FAQ queries
        elif any(k in query_lower for k in ["smartlogi", "faq", "what is smartlogi"]):
            intent = "FAQ"
            scope = "BOTH"
        # 5. General RAG Scope Resolution
        else:
            intent = "GENERAL_RAG"
            if any(k in query_lower for k in ["brd", "srs", "business", "crm", "requirement", "customer"]):
                scope = "CRM"
            elif any(k in query_lower for k in ["tech", "team", "architecture", "deployment", "test", "operation", "api"]):
                scope = "Tech_Team"
            else:
                scope = "BOTH"

        state["intent"] = intent
        state["scope"] = scope
        state["retrieval_query"] = query
        state["tool_calls"] = state.get("tool_calls") or []
        state["iteration_count"] = state.get("iteration_count", 0)

        logger.info(f"[IntentRouterNode] Query: '{query}' -> Intent: {intent}, Scope: {scope}")
        return state
