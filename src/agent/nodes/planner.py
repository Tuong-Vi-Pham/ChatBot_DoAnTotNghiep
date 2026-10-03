import logging
from typing import List
from src.agent.state import AgentState

logger = logging.getLogger(__name__)


class PlannerNode:
    """
    Generates targeted multi-step execution plans based on intent and scope.
    Ensures the agent retrieves only necessary ticket/document evidence without dumping full KB.
    """

    @classmethod
    def execute(cls, state: AgentState) -> AgentState:
        intent = state.get("intent", "GENERAL_RAG")
        scope = state.get("scope", "BOTH")
        query = state.get("user_query", "")

        plan: List[str] = []

        if intent == "FAQ":
            plan = [
                "1. Search FAQ collection for exact/high-confidence Q&A match.",
                "2. Verify answer grounding."
            ]
        elif intent == "TICKET_QUERY":
            plan = [
                "1. Query Lark Base Daily_Task table for target ticket or ticket list.",
                "2. Extract ticket attributes (status, assignee, story points, sprint).",
                "3. Format ticket response."
            ]
        elif intent == "TICKET_ANALYSIS":
            plan = [
                "1. Retrieve target ticket details from Daily_Task.",
                "2. Fetch related sub-tickets and parent epic links.",
                "3. Cross-reference technical specs if needed.",
                "4. Synthesize ticket analysis."
            ]
        elif intent == "PRIORITY_ANALYSIS":
            plan = [
                "1. Fetch unfinished tickets from Daily_Task (status != Done).",
                "2. Retrieve relevant CRM business requirements and Tech_Team specs.",
                "3. Evaluate urgency, business impact, and technical dependencies.",
                "4. Formulate priority recommendations."
            ]
        elif intent == "CLARIFICATION_REQUIRED":
            plan = [
                "1. Identify missing scope/filters in user query.",
                "2. Generate structured clarification options."
            ]
        else:  # GENERAL_RAG
            plan = [
                f"1. Perform source-aware vector search against {scope} knowledge base.",
                "2. Rerank retrieved candidate chunks using BGE CrossEncoder.",
                "3. Verify evidence adequacy and generate grounded response."
            ]

        state["plan"] = plan
        logger.info(f"[PlannerNode] Generated plan for intent '{intent}': {plan}")
        return state
