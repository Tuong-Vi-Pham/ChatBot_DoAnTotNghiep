import logging
from typing import Any, Dict, Optional
from langgraph.graph import StateGraph, START, END

from src.agent.state import AgentState
from src.agent.nodes.intent_router import IntentRouterNode
from src.agent.nodes.planner import PlannerNode
from src.agent.nodes.tool_executor import ToolExecutionNode
from src.agent.nodes.evidence_assessor import EvidenceAssessorNode, QueryRewriterNode
from src.agent.nodes.priority_analyzer import PriorityAnalyzerNode
from src.agent.nodes.approval_node import ApprovalNode
from src.agent.nodes.update_node import UpdateExecutionNode
from src.agent.nodes.verifier import VerifierNode
from src.agent.nodes.response_generator import ResponseGeneratorNode
from src.agent.nodes.clarifier import ClarificationNode
from src.agent.tools.ticket_tool import TicketToolAdapter
from src.agent.tools.rag_tool import RAGToolAdapter
from src.agent.approval.approval_service import ApprovalService
from src.agent.update.update_service import ControlledUpdateService

logger = logging.getLogger(__name__)


def route_intent_edge(state: AgentState) -> str:
    """Conditional edge routing from intent_router node."""
    intent = state.get("intent")
    action = state.get("approval_action")
    if intent == "CLARIFICATION_REQUIRED":
        return "clarifier"
    elif intent == "PRIORITY_APPROVAL_DECISION":
        return "approval_node"
    elif intent == "PRIORITY_UPDATE_EXECUTION":
        return "update_node"
    elif intent == "PRIORITY_APPROVAL_REQUEST":
        if action in ("APPROVE", "REJECT", "CANCEL"):
            return "approval_node"
        return "planner"
    return "planner"


def evaluate_evidence_edge(state: AgentState) -> str:
    """Conditional edge routing from evidence_assessor node for bounded retrieval & priority/approval analysis."""
    if not EvidenceAssessorNode.is_evidence_sufficient(state):
        return "query_rewriter"

    intent = state.get("intent", "")
    if intent == "PRIORITY_APPROVAL_REQUEST":
        return "priority_analyzer"
    elif intent in ("PRIORITY_ANALYSIS", "TICKET_ANALYSIS"):
        return "priority_analyzer"
    return "verifier"


def build_agent_graph(
    ticket_tool: Optional[TicketToolAdapter] = None,
    rag_tool: Optional[RAGToolAdapter] = None,
    approval_service: Optional[ApprovalService] = None,
    update_service: Optional[ControlledUpdateService] = None
) -> Any:
    """
    Constructs and compiles the LangGraph StateGraph orchestrator with Priority Analyzer, Human Approval, and Controlled Update integration.
    """
    shared_approval_service = approval_service or ApprovalService()
    shared_update_service = update_service or ControlledUpdateService(approval_service=shared_approval_service)
    tool_executor_node = ToolExecutionNode(ticket_tool=ticket_tool, rag_tool=rag_tool)
    approval_node = ApprovalNode(approval_service=shared_approval_service)
    update_node = UpdateExecutionNode(update_service=shared_update_service)
    verifier_node = VerifierNode()
    response_generator_node = ResponseGeneratorNode()

    # Define Node Wrappers
    def intent_router_step(state: AgentState) -> AgentState:
        return IntentRouterNode.execute(state)

    def planner_step(state: AgentState) -> AgentState:
        return PlannerNode.execute(state)

    def tool_executor_step(state: AgentState) -> AgentState:
        return tool_executor_node.execute(state)

    def evidence_assessor_step(state: AgentState) -> AgentState:
        return EvidenceAssessorNode.execute(state)

    def query_rewriter_step(state: AgentState) -> AgentState:
        return QueryRewriterNode.execute(state)

    def priority_analyzer_step(state: AgentState) -> AgentState:
        res = PriorityAnalyzerNode.execute(state)
        # If approval request intent OR recommendation differs from current priority, create pending approval request
        priority_analysis = res.get("priority_analysis") or {}
        if state.get("intent") == "PRIORITY_APPROVAL_REQUEST" or priority_analysis.get("differs_from_current"):
            res = approval_node.execute(res)
        return res

    def approval_step(state: AgentState) -> AgentState:
        return approval_node.execute(state)

    def update_step(state: AgentState) -> AgentState:
        return update_node.execute(state)

    def verifier_step(state: AgentState) -> AgentState:
        return verifier_node.execute(state)

    def response_generator_step(state: AgentState) -> AgentState:
        return response_generator_node.execute(state)

    def clarifier_step(state: AgentState) -> AgentState:
        return ClarificationNode.execute(state)

    # Build Graph
    builder = StateGraph(AgentState)

    builder.add_node("intent_router", intent_router_step)
    builder.add_node("planner", planner_step)
    builder.add_node("tool_executor", tool_executor_step)
    builder.add_node("evidence_assessor", evidence_assessor_step)
    builder.add_node("query_rewriter", query_rewriter_step)
    builder.add_node("priority_analyzer", priority_analyzer_step)
    builder.add_node("approval_node", approval_step)
    builder.add_node("update_node", update_step)
    builder.add_node("verifier", verifier_step)
    builder.add_node("response_generator", response_generator_step)
    builder.add_node("clarifier", clarifier_step)

    # Wire Edges
    builder.add_edge(START, "intent_router")

    builder.add_conditional_edges(
        "intent_router",
        route_intent_edge,
        {
            "clarifier": "clarifier",
            "approval_node": "approval_node",
            "update_node": "update_node",
            "planner": "planner"
        }
    )

    builder.add_edge("planner", "tool_executor")
    builder.add_edge("tool_executor", "evidence_assessor")

    builder.add_conditional_edges(
        "evidence_assessor",
        evaluate_evidence_edge,
        {
            "verifier": "verifier",
            "query_rewriter": "query_rewriter",
            "priority_analyzer": "priority_analyzer"
        }
    )

    def route_approval_edge(state: AgentState) -> str:
        if state.get("approval_status") == "APPROVED" and state.get("intent") == "PRIORITY_APPROVAL_REQUEST":
            return "update_node"
        return "verifier"

    builder.add_edge("query_rewriter", "tool_executor")
    builder.add_edge("priority_analyzer", "verifier")
    builder.add_conditional_edges(
        "approval_node",
        route_approval_edge,
        {
            "update_node": "update_node",
            "verifier": "verifier"
        }
    )
    builder.add_edge("update_node", "verifier")
    builder.add_edge("verifier", "response_generator")

    builder.add_edge("clarifier", END)
    builder.add_edge("response_generator", END)

    return builder.compile()


class AgentOrchestrator:
    """
    High-level orchestrator interface running the compiled LangGraph StateGraph.
    """

    def __init__(
        self,
        ticket_tool: Optional[TicketToolAdapter] = None,
        rag_tool: Optional[RAGToolAdapter] = None,
        approval_service: Optional[ApprovalService] = None,
        update_service: Optional[ControlledUpdateService] = None
    ):
        self.graph = build_agent_graph(
            ticket_tool=ticket_tool,
            rag_tool=rag_tool,
            approval_service=approval_service,
            update_service=update_service
        )

    def run(self, user_query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Executes full agentic workflow for given query."""
        initial_state: AgentState = {
            "user_query": user_query,
            "session_id": session_id,
            "approval_action": None,
            "intent": "GENERAL_RAG",
            "scope": "BOTH",
            "plan": [],
            "ticket_data": None,
            "retrieval_query": user_query,
            "retrieval_results": [],
            "ticket_context": [],
            "project_context": [],
            "evidence_quality": 0.0,
            "verification_result": {},
            "confidence": 0.0,
            "iteration_count": 0,
            "tool_calls": [],
            "priority_analysis": None,
            "priority_ranking": [],
            "approval_request": None,
            "approval_status": None,
            "approval_id": None,
            "approval_error": None,
            "update_result": None,
            "update_status": None,
            "update_error": None,
            "final_response": "",
            "error": None
        }

        try:
            final_state = self.graph.invoke(initial_state)
            return dict(final_state)
        except Exception as e:
            logger.error(f"[AgentOrchestrator] Execution failed: {e}", exc_info=True)
            initial_state["error"] = str(e)
            initial_state["final_response"] = (
                "An unexpected error occurred during query execution. "
                "The system failed safely without fabricating data."
            )
            return dict(initial_state)
