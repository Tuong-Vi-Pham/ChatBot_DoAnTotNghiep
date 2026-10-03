import re
import logging
from typing import Any, Dict, Optional
from src.agent.state import AgentState
from src.agent.approval import (
    ApprovalService,
    ApprovalStatus,
    ActorType,
    ApprovalError,
    InvalidStateTransitionError,
    UnauthorizedApproverError,
)

logger = logging.getLogger(__name__)


class ApprovalNode:
    """
    LangGraph Node executing Human Approval requests and decision handling.
    STRICTLY READ-ONLY with respect to Lark Base (performs NO write operations against Lark).
    """

    def __init__(self, approval_service: Optional[ApprovalService] = None):
        self.approval_service = approval_service or ApprovalService()

    def execute(self, state: AgentState) -> AgentState:
        intent = state.get("intent", "")
        query = state.get("user_query", "")

        priority_analysis = state.get("priority_analysis") or {}
        differs = priority_analysis.get("differs_from_current", False)

        if intent not in ("PRIORITY_APPROVAL_REQUEST", "PRIORITY_APPROVAL_DECISION") and not differs:
            return state

        try:
            if intent == "PRIORITY_APPROVAL_DECISION":
                self._handle_approval_decision(state, query)
            elif intent == "PRIORITY_APPROVAL_REQUEST":
                action = state.get("approval_action")
                if not action:
                    query_lower = query.lower()
                    if any(k in query_lower for k in ["reject", "no", "do not proceed", "don't update", "keep the current"]):
                        action = "REJECT"
                    elif "cancel" in query_lower:
                        action = "CANCEL"
                    elif any(k in query_lower for k in ["approve", "yes", "proceed", "confirm", "approved", "go ahead"]):
                        action = "APPROVE"
                    else:
                        action = "REQUEST"

                if action in ("APPROVE", "REJECT", "CANCEL"):
                    self._handle_conversational_decision(state, query, action)
                else:
                    self._handle_approval_request(state)
            elif differs:
                self._handle_approval_request(state)
        except ApprovalError as e:
            logger.error(f"[ApprovalNode] Approval operation error: {e}")
            state["approval_error"] = str(e)
            state["error"] = str(e)

        return state

    def _handle_approval_request(self, state: AgentState) -> None:
        ticket = state.get("ticket_data") or (state.get("ticket_context")[0] if state.get("ticket_context") else {})
        priority_analysis = state.get("priority_analysis") or {}

        if not ticket:
            state["approval_error"] = "Cannot create approval request: No target ticket found."
            return

        req = self.approval_service.create_approval_request(
            ticket=ticket,
            priority_analysis=priority_analysis,
            requested_by="AGENT"
        )

        session_id = state.get("session_id")
        if session_id:
            self.approval_service.store.set_session_pending_approval(session_id, req.approval_id)

        state["approval_request"] = req.to_dict()
        state["approval_id"] = req.approval_id
        state["approval_status"] = req.status.value
        logger.info(f"[ApprovalNode] Created approval request {req.approval_id} for ticket {req.ticket_id}.")

    def _handle_conversational_decision(self, state: AgentState, query: str, action: str) -> None:
        query_lower = query.lower()
        session_id = state.get("session_id")

        target_req: Optional[Any] = None

        # Check explicit APR-xxx
        match_id = re.search(r'\bapr-\d+\b', query_lower)
        if match_id:
            apr_id = match_id.group(0).upper()
            target_req = self.approval_service.store.get_request(apr_id)
            if not target_req:
                state["approval_error"] = f"Approval request '{apr_id}' not found."
                state["error"] = state["approval_error"]
                return

        # Check ticket ID in query (e.g. CRM-044)
        ticket_match = re.search(r'\b(task|epic|bug|sub|crm|story)-\d+\b', query_lower)
        ticket_id = ticket_match.group(0).upper() if ticket_match else None

        # Enforce session isolation if session_id is provided
        if session_id:
            session_apr_id = self.approval_service.store.get_session_pending_approval(session_id)
            if not session_apr_id:
                state["approval_error"] = "No pending approval found in current session."
                state["error"] = state["approval_error"]
                return
            if not target_req:
                target_req = self.approval_service.store.get_request(session_apr_id)
            elif target_req.approval_id != session_apr_id:
                state["approval_error"] = f"Approval request '{target_req.approval_id}' does not belong to the current session."
                state["error"] = state["approval_error"]
                return
            if ticket_id and target_req and target_req.ticket_id.upper() != ticket_id:
                state["approval_error"] = f"Specified ticket '{ticket_id}' does not match pending approval ticket '{target_req.ticket_id}'."
                state["error"] = state["approval_error"]
                return
        elif not target_req:
            # Fallback when no session_id: check ticket ID
            if ticket_id:
                target_req = self.approval_service.store.find_pending_by_ticket(ticket_id)
            else:
                # Find most recent pending request
                pending = [r for r in self.approval_service.store.list_requests() if r.status == ApprovalStatus.PENDING_APPROVAL]
                if pending:
                    target_req = pending[-1]

        if not target_req:
            state["approval_error"] = "No pending recommendation found to approve."
            state["error"] = state["approval_error"]
            return

        if target_req.status != ApprovalStatus.PENDING_APPROVAL:
            state["approval_error"] = f"Approval request '{target_req.approval_id}' has already been processed (Status: {target_req.status.value})."
            state["error"] = state["approval_error"]
            return

        reviewer_match = re.search(r'by\s+([a-zA-Z0-9_\-]+)', query, re.IGNORECASE)
        reviewer_id = reviewer_match.group(1) if reviewer_match else "user_pm"

        if "actor=agent" in query_lower or reviewer_id.lower() == "agent":
            actor_type = ActorType.AGENT
        else:
            actor_type = ActorType.APPROVER

        apr_id = target_req.approval_id
        if action == "APPROVE":
            req = self.approval_service.approve(approval_id=apr_id, reviewer_id=reviewer_id, actor_type=actor_type)
        elif action == "REJECT":
            reason_match = re.search(r'reason:\s*(.+)', query, re.IGNORECASE)
            reason = reason_match.group(1).strip() if reason_match else "Rejected by user."
            req = self.approval_service.reject(approval_id=apr_id, reviewer_id=reviewer_id, rejection_reason=reason, actor_type=actor_type)
            if session_id:
                self.approval_service.store.clear_session_pending_approval(session_id)
        elif action == "CANCEL":
            req = self.approval_service.cancel(approval_id=apr_id, actor_id=reviewer_id, actor_type=ActorType.USER)
            if session_id:
                self.approval_service.store.clear_session_pending_approval(session_id)

        state["approval_request"] = req.to_dict()
        state["approval_id"] = req.approval_id
        state["approval_status"] = req.status.value
        logger.info(f"[ApprovalNode] Conversational decision '{action}' executed for {apr_id}. Resulting status: {req.status.value}.")

    def _handle_approval_decision(self, state: AgentState, query: str) -> None:
        query_lower = query.lower()

        match_id = re.search(r'\bapr-\d+\b', query_lower)
        if not match_id:
            state["approval_error"] = "Approval decision query must specify a valid Approval ID (e.g. APR-001)."
            return

        apr_id = match_id.group(0).upper()

        # Parse action
        if "approve" in query_lower:
            action = "APPROVE"
        elif "reject" in query_lower:
            action = "REJECT"
        elif "cancel" in query_lower:
            action = "CANCEL"
        else:
            state["approval_error"] = f"Unknown approval decision action in query: '{query}'."
            return

        # Parse reviewer/actor
        reviewer_match = re.search(r'by\s+([a-zA-Z0-9_\-]+)', query, re.IGNORECASE)
        reviewer_id = reviewer_match.group(1) if reviewer_match else "approver_pm"

        # Check actor type
        if "actor=agent" in query_lower or reviewer_id.lower() == "agent":
            actor_type = ActorType.AGENT
        else:
            actor_type = ActorType.APPROVER

        if action == "APPROVE":
            req = self.approval_service.approve(approval_id=apr_id, reviewer_id=reviewer_id, actor_type=actor_type)
        elif action == "REJECT":
            reason_match = re.search(r'reason:\s*(.+)', query, re.IGNORECASE)
            reason = reason_match.group(1).strip() if reason_match else "Rejected by approver."
            req = self.approval_service.reject(approval_id=apr_id, reviewer_id=reviewer_id, rejection_reason=reason, actor_type=actor_type)
        elif action == "CANCEL":
            req = self.approval_service.cancel(approval_id=apr_id, actor_id=reviewer_id, actor_type=ActorType.USER)

        state["approval_request"] = req.to_dict()
        state["approval_id"] = req.approval_id
        state["approval_status"] = req.status.value
        logger.info(f"[ApprovalNode] Decision '{action}' executed for {apr_id}. Resulting status: {req.status.value}.")

