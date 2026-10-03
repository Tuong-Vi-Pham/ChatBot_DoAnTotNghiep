import re
import logging
from typing import Any, Dict, List, Optional
from src.agent.state import AgentState
from src.agent.tools.ticket_tool import TicketToolAdapter
from src.agent.tools.rag_tool import RAGToolAdapter

logger = logging.getLogger(__name__)


class ToolExecutionNode:
    """
    Executes appropriate ticket tools and/or RAG tools based on state intent, plan, and scope.
    Populates ticket_context, project_context, and retrieval_results.
    """

    def __init__(
        self,
        ticket_tool: Optional[TicketToolAdapter] = None,
        rag_tool: Optional[RAGToolAdapter] = None
    ):
        self.ticket_tool = ticket_tool or TicketToolAdapter()
        self.rag_tool = rag_tool or RAGToolAdapter()

    @staticmethod
    def extract_project(query: str) -> Optional[str]:
        """Extracts project name or identifier from user query."""
        m_proj = re.search(r'project\s*[:=]?\s*([A-Za-z0-9\s]+?)(?:\s*[-–—]\s*sprint|\s+sprint|\s*\?|\s*$)', query, re.IGNORECASE)
        if m_proj:
            cand = m_proj.group(1).strip().rstrip('?.!;')
            if cand and not cand.lower().startswith('sprint'):
                return cand

        m_in = re.search(r'(?:in|for)\s+(?:the\s+)?([A-Za-z0-9\s]+?)(?:\s*[-–—]\s*sprint|\s+sprint|\s*\?|\s*$)', query, re.IGNORECASE)
        if m_in:
            cand = m_in.group(1).strip().rstrip('?.!;')
            if cand and not cand.lower().startswith('sprint'):
                return cand
        return None

    @staticmethod
    def extract_sprint(query: str) -> Optional[str]:
        """Extracts full sprint name from user query."""
        sprint_match = re.search(r'sprint\s*[^?.!;,\n]+', query, re.IGNORECASE)
        if sprint_match:
            return sprint_match.group(0).strip().rstrip('?.!;')
        return None

    def execute(self, state: AgentState) -> AgentState:
        intent = state.get("intent", "GENERAL_RAG")
        scope = state.get("scope", "BOTH")
        query = state.get("retrieval_query") or state.get("user_query", "")

        tool_calls = state.get("tool_calls", [])
        ticket_context: List[Dict[str, Any]] = state.get("ticket_context") or []
        project_context: List[Dict[str, Any]] = state.get("project_context") or []
        retrieval_results: List[Dict[str, Any]] = state.get("retrieval_results") or []

        extracted_project = self.extract_project(query)
        extracted_sprint = self.extract_sprint(query)

        # 1. Execute Ticket Tools for ticket queries, analysis, & approval requests
        if intent in ("TICKET_QUERY", "TICKET_ANALYSIS", "PRIORITY_ANALYSIS", "PRIORITY_APPROVAL_REQUEST"):
            # Check if query contains explicit Ticket ID
            match = re.search(r'\b(TASK|EPIC|BUG|SUB|CRM|STORY)-\d+\b', query, re.IGNORECASE)
            if match:
                t_id = match.group(0).upper()
                tool_calls.append({"tool": "ticket_tool.get_ticket_by_id", "args": {"ticket_id": t_id, "project": extracted_project}})
                res = self.ticket_tool.get_ticket_by_id(t_id)
                if res.get("success") and res.get("ticket"):
                    ticket_data = res["ticket"]
                    state["ticket_data"] = ticket_data
                    ticket_context.append(ticket_data)

                    # Fetch related tickets if analysis or approval request
                    if intent in ("TICKET_ANALYSIS", "PRIORITY_APPROVAL_REQUEST"):
                        tool_calls.append({"tool": "ticket_tool.get_related_tickets", "args": {"ticket_id": t_id}})
                        rel_res = self.ticket_tool.get_related_tickets(t_id)
                        if rel_res.get("success"):
                            ticket_context.extend(rel_res.get("related_tickets", []))
            # Check for Sprint filter
            elif extracted_sprint:
                tool_calls.append({"tool": "ticket_tool.get_tickets_by_sprint", "args": {"sprint": extracted_sprint, "project": extracted_project}})
                res = self.ticket_tool.get_tickets_by_sprint(extracted_sprint)
                if res.get("success"):
                    tickets = res.get("tickets", [])
                    ticket_context.extend(tickets)
                    if tickets:
                        state["ticket_data"] = tickets[0]
            # Check for Project filter without sprint
            elif extracted_project:
                tool_calls.append({"tool": "ticket_tool.get_tickets_by_project", "args": {"project": extracted_project}})
                res = self.ticket_tool.get_tickets_by_project(extracted_project)
                if res.get("success"):
                    tickets = res.get("tickets", [])
                    if intent in ("PRIORITY_ANALYSIS", "PRIORITY_APPROVAL_REQUEST"):
                        unfinished = [t for t in tickets if str(t.get("status", "")).strip().lower() not in ("done", "completed", "cancelled", "canceled")]
                        ticket_context.extend(unfinished if unfinished else tickets)
                    else:
                        ticket_context.extend(tickets)
                    if ticket_context:
                        state["ticket_data"] = ticket_context[0]
            # Default / Priority Analysis fetch
            else:
                tool_calls.append({"tool": "ticket_tool.get_all_tickets", "args": {"max_pages": 5}})
                res = self.ticket_tool.get_all_tickets(max_pages=5)
                if res.get("success"):
                    tickets = res.get("tickets", [])
                    # Filter unfinished for priority analysis
                    if intent in ("PRIORITY_ANALYSIS", "PRIORITY_APPROVAL_REQUEST"):
                        unfinished = [t for t in tickets if str(t.get("status", "")).strip().lower() not in ("done", "completed", "cancelled", "canceled")]
                        ticket_context.extend(unfinished if unfinished else tickets)
                    else:
                        ticket_context.extend(tickets)
                    if ticket_context:
                        state["ticket_data"] = ticket_context[0]

            logger.info(
                f"[ToolExecutionNode] LARK RETRIEVAL: project='{extracted_project}', "
                f"sprint='{extracted_sprint}', normalized_tickets={len(ticket_context)}, "
                f"ticket_ids={[t.get('ticket_id') for t in ticket_context]}"
            )

        # 2. Execute RAG Tools for General RAG, Priority Analysis, or document context
        if intent in ("GENERAL_RAG", "FAQ", "PRIORITY_ANALYSIS", "TICKET_ANALYSIS", "PRIORITY_APPROVAL_REQUEST"):
            rag_scope = "BOTH" if intent in ("PRIORITY_ANALYSIS", "PRIORITY_APPROVAL_REQUEST") else scope
            tool_calls.append({"tool": "rag_tool.retrieve", "args": {"query": query, "scope": rag_scope}})
            
            rag_res = self.rag_tool.retrieve(query=query, scope=rag_scope, top_k=5)
            if rag_res.get("success"):
                ev_items = rag_res.get("evidence", [])
                retrieval_results.extend(ev_items)
                project_context.extend(ev_items)

        state["tool_calls"] = tool_calls
        state["ticket_context"] = ticket_context
        state["project_context"] = project_context
        state["retrieval_results"] = retrieval_results
        state["iteration_count"] = state.get("iteration_count", 0) + 1

        logger.info(
            f"[ToolExecutionNode] Iteration {state['iteration_count']}: "
            f"{len(ticket_context)} ticket items, {len(project_context)} RAG evidence items retrieved."
        )
        return state
