import re
import logging
from typing import Any, Dict, Optional
from src.agent.state import AgentState
from src.agent.approval.approval_model import ActorType
from src.agent.update import ControlledUpdateService, UpdateResultStatus

logger = logging.getLogger(__name__)


class UpdateExecutionNode:
    """
    LangGraph Node executing controlled priority updates against Lark Base for APPROVED recommendations.
    """

    def __init__(self, update_service: Optional[ControlledUpdateService] = None):
        self.update_service = update_service or ControlledUpdateService()

    def execute(self, state: AgentState) -> AgentState:
        intent = state.get("intent", "")
        query = state.get("user_query", "")

        if intent not in ("PRIORITY_UPDATE_EXECUTION", "PRIORITY_APPROVAL_REQUEST"):
            return state

        approval_id = state.get("approval_id")
        if not approval_id:
            match_id = re.search(r'\bapr-\d+\b', query, re.IGNORECASE)
            if match_id:
                approval_id = match_id.group(0).upper()

        if not approval_id:
            state["update_error"] = "Update execution query must specify a valid Approval ID (e.g. APR-001)."
            state["update_status"] = UpdateResultStatus.UPDATE_REJECTED.value
            return state

        approval_id = approval_id.upper()

        reviewer_match = re.search(r'by\s+([a-zA-Z0-9_\-]+)', query, re.IGNORECASE)
        reviewer_id = reviewer_match.group(1) if reviewer_match else "user_pm"

        if "actor=agent" in query.lower() or reviewer_id.lower() == "agent":
            actor_type = ActorType.AGENT
        else:
            actor_type = ActorType.APPROVER

        logger.info(f"[UpdateExecutionNode] Invoking update pipeline for '{approval_id}' by reviewer '{reviewer_id}'.")
        result = self.update_service.apply_approved_priority_update(
            approval_id=approval_id,
            actor_id=reviewer_id,
            actor_type=actor_type
        )

        state["update_result"] = result.to_dict()
        state["update_status"] = result.status.value

        session_id = state.get("session_id")
        if session_id and result.status in (UpdateResultStatus.UPDATE_VERIFIED, UpdateResultStatus.ALREADY_APPLIED):
            self.update_service.store.clear_session_pending_approval(session_id)

        # Only set update_error for hard unhandled errors, not structured operational statuses
        handled_statuses = (
            UpdateResultStatus.UPDATE_VERIFIED,
            UpdateResultStatus.ALREADY_APPLIED,
            UpdateResultStatus.UPDATE_STALE,
            UpdateResultStatus.UPDATE_REJECTED
        )
        if not result.verified and result.status not in handled_statuses:
            state["update_error"] = result.reason

        return state
