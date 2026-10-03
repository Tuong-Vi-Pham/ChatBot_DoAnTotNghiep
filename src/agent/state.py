from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, TypedDict


@dataclass
class Evidence:
    """Structured evidence item representing a single piece of retrieved information."""
    source: str                 # "Lark" | "CRM" | "Tech_Team" | "FAQ"
    document_id: str
    chunk_id: str
    source_path: str
    content: str
    relevance_score: float
    ticket_id: Optional[str] = None
    evidence_type: str = "TECHNICAL"  # "BUSINESS" | "TECHNICAL" | "TICKET" | "DEPENDENCY" | "RELEASE" | "SPRINT"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentState(TypedDict, total=False):
    """
    Strongly structured LangGraph AgentState representing the orchestrator execution state.
    Exposes explicit state without storing hidden chain-of-thought.
    """
    user_query: str
    intent: str                 # "FAQ" | "GENERAL_RAG" | "TICKET_QUERY" | "TICKET_ANALYSIS" | "PRIORITY_ANALYSIS" | "CLARIFICATION_REQUIRED" | "PRIORITY_APPROVAL_REQUEST" | "PRIORITY_APPROVAL_DECISION" | "PRIORITY_UPDATE_EXECUTION"
    scope: str                  # "CRM" | "Tech_Team" | "BOTH" | "NONE"
    plan: List[str]
    ticket_data: Optional[Dict[str, Any]]
    retrieval_query: str
    retrieval_results: List[Dict[str, Any]]
    ticket_context: List[Dict[str, Any]]
    project_context: List[Dict[str, Any]]
    evidence_quality: float
    verification_result: Dict[str, Any]
    confidence: float
    iteration_count: int
    tool_calls: List[Dict[str, Any]]
    priority_analysis: Optional[Dict[str, Any]]
    priority_ranking: List[Dict[str, Any]]
    approval_request: Optional[Dict[str, Any]]
    approval_status: Optional[str]
    approval_id: Optional[str]
    approval_error: Optional[str]
    update_result: Optional[Dict[str, Any]]
    update_status: Optional[str]
    update_error: Optional[str]
    session_id: Optional[str]
    approval_action: Optional[str]
    final_response: str
    error: Optional[str]

