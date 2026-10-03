import logging
from typing import Any, Dict
from src.agent.state import AgentState

logger = logging.getLogger(__name__)


class ClarificationNode:
    """
    Generates structured clarification options when user queries lack required scope or parameters.
    """

    @classmethod
    def execute(cls, state: AgentState) -> AgentState:
        query = state.get("user_query", "")

        response_lines = [
            f"Your request '{query}' requires additional scope parameters to execute accurately.\n",
            "Please select or specify one of the following options:\n",
            "1. **Analyze Unfinished Tickets for Sprint 12**",
            "2. **Analyze Backend Team Tickets**",
            "3. **Specify a single Ticket ID (e.g. TASK-101)**\n",
            "Reply with your chosen option or refine your query."
        ]

        state["final_response"] = "\n".join(response_lines)
        state["verification_result"] = {
            "verified": True,
            "is_grounded": True,
            "reason": "Clarification options rendered."
        }

        logger.info(f"[ClarificationNode] Formulated clarification options for query: '{query}'")
        return state
