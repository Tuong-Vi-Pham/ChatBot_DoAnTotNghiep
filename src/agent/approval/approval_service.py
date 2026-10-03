import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

from src.agent.approval.approval_model import (
    ApprovalRequest,
    ApprovalStatus,
    ActorType,
    AuditEvent,
    RecommendationSnapshot,
)
from src.agent.approval.approval_store import ApprovalStore, get_shared_approval_store

logger = logging.getLogger(__name__)


class ApprovalError(Exception):
    """Base exception for approval subsystem errors."""
    pass


class InvalidStateTransitionError(ApprovalError):
    """Raised when an illegal approval state transition is attempted."""
    pass


class UnauthorizedApproverError(ApprovalError):
    """Raised when an unauthorized actor attempts to perform an approver action."""
    pass


class StaleRecommendationError(ApprovalError):
    """Raised when an approval request snapshot is stale due to ticket priority changes."""
    pass


class DuplicateApprovalError(ApprovalError):
    """Raised when a duplicate approval request is created for a pending ticket."""
    pass


class ApprovalService:
    """
    Service managing the human-in-the-loop approval lifecycle and append-only audit trail.
    STRICTLY READ-ONLY with respect to Lark Base (performs NO write operations against Lark).
    """

    def __init__(self, store: Optional[ApprovalStore] = None):
        self.store = store if store is not None else get_shared_approval_store()

    def create_approval_request(
        self,
        ticket: Dict[str, Any],
        priority_analysis: Dict[str, Any],
        requested_by: str = "AGENT"
    ) -> ApprovalRequest:
        """
        Creates a new PENDING_APPROVAL request capturing a RecommendationSnapshot.
        Detects pending duplicates to prevent redundant requests.
        """
        t_id = ticket.get("ticket_id", "")
        if not t_id:
            raise ApprovalError("Cannot create approval request for ticket without ticket_id.")

        # Duplicate pending request check
        existing_pending = self.store.find_pending_by_ticket(t_id)
        if existing_pending:
            logger.info(f"[ApprovalService] Existing pending request {existing_pending.approval_id} found for ticket {t_id}.")
            return existing_pending

        approval_id = self.store.generate_approval_id()
        snapshot_id = f"SNP-{approval_id}"

        # Capture snapshot
        snapshot = RecommendationSnapshot(
            snapshot_id=snapshot_id,
            ticket_id=t_id,
            original_priority=priority_analysis.get("current_priority") or ticket.get("priority", "MEDIUM"),
            recommended_priority=priority_analysis.get("recommended_priority", "HIGH"),
            weighted_score=float(priority_analysis.get("weighted_score", 0.0)),
            confidence=str(priority_analysis.get("confidence", "Medium")),
            summary_reason=str(priority_analysis.get("summary_reason", "")),
            factors=priority_analysis.get("factors", {}),
            evidence_references=[
                f"{ev.get('source')}: {ev.get('source_path')}"
                for ev in priority_analysis.get("project_context", [])
            ]
        )

        request = ApprovalRequest(
            approval_id=approval_id,
            ticket_id=t_id,
            snapshot=snapshot,
            status=ApprovalStatus.PENDING_APPROVAL,
            requested_by=requested_by
        )

        self.store.save_request(request)

        # Audit event
        audit_event = AuditEvent(
            audit_id=self.store.generate_audit_id(),
            approval_id=approval_id,
            ticket_id=t_id,
            action="APPROVAL_REQUESTED",
            previous_state="NONE",
            new_state=ApprovalStatus.PENDING_APPROVAL.value,
            actor_type=ActorType.AGENT if requested_by == "AGENT" else ActorType.USER,
            actor_id=requested_by,
            reason=f"Priority change recommendation requested for {t_id} ({snapshot.original_priority} -> {snapshot.recommended_priority})",
            snapshot_id=snapshot_id
        )
        self.store.append_audit_event(audit_event)

        return request

    def approve(
        self,
        approval_id: str,
        reviewer_id: str,
        actor_type: ActorType = ActorType.APPROVER
    ) -> ApprovalRequest:
        """
        Approves a PENDING_APPROVAL request.
        Requires explicit human reviewer identity (actor_type != AGENT).
        STRICTLY READ-ONLY (does NOT execute Lark write APIs in Session 06A).
        """
        if actor_type == ActorType.AGENT or reviewer_id.upper() == "AGENT":
            raise UnauthorizedApproverError("Agent/System is forbidden from acting as human approver.")

        if not reviewer_id or not reviewer_id.strip():
            raise UnauthorizedApproverError("Reviewer identity required to approve request.")

        request = self.store.get_request(approval_id)
        if not request:
            raise ApprovalError(f"Approval request '{approval_id}' not found.")

        # Idempotency check
        if request.status == ApprovalStatus.APPROVED:
            logger.info(f"[ApprovalService] Request {approval_id} is already APPROVED.")
            return request

        # Valid transition check
        if request.status != ApprovalStatus.PENDING_APPROVAL:
            raise InvalidStateTransitionError(
                f"Cannot approve request {approval_id} in state '{request.status.value}'. Only PENDING_APPROVAL can be approved."
            )

        prev_state = request.status.value
        request.status = ApprovalStatus.APPROVED
        request.reviewed_by = reviewer_id
        request.reviewed_at = datetime.now(timezone.utc).isoformat()

        self.store.save_request(request)

        # Audit event
        audit_event = AuditEvent(
            audit_id=self.store.generate_audit_id(),
            approval_id=approval_id,
            ticket_id=request.ticket_id,
            action="APPROVED",
            previous_state=prev_state,
            new_state=ApprovalStatus.APPROVED.value,
            actor_type=actor_type,
            actor_id=reviewer_id,
            reason=f"Priority change approved by human reviewer {reviewer_id}. (Session 06A READ-ONLY)",
            snapshot_id=request.snapshot.snapshot_id
        )
        self.store.append_audit_event(audit_event)

        return request

    def reject(
        self,
        approval_id: str,
        reviewer_id: str,
        rejection_reason: str,
        actor_type: ActorType = ActorType.APPROVER
    ) -> ApprovalRequest:
        """
        Rejects a PENDING_APPROVAL request with explicit rejection rationale.
        """
        if actor_type == ActorType.AGENT or reviewer_id.upper() == "AGENT":
            raise UnauthorizedApproverError("Agent/System is forbidden from acting as human approver.")

        if not reviewer_id or not reviewer_id.strip():
            raise UnauthorizedApproverError("Reviewer identity required to reject request.")

        request = self.store.get_request(approval_id)
        if not request:
            raise ApprovalError(f"Approval request '{approval_id}' not found.")

        # Idempotency check
        if request.status == ApprovalStatus.REJECTED:
            logger.info(f"[ApprovalService] Request {approval_id} is already REJECTED.")
            return request

        # Valid transition check
        if request.status != ApprovalStatus.PENDING_APPROVAL:
            raise InvalidStateTransitionError(
                f"Cannot reject request {approval_id} in state '{request.status.value}'. Only PENDING_APPROVAL can be rejected."
            )

        prev_state = request.status.value
        request.status = ApprovalStatus.REJECTED
        request.reviewed_by = reviewer_id
        request.reviewed_at = datetime.now(timezone.utc).isoformat()
        request.rejection_reason = rejection_reason or "No rejection reason provided."

        self.store.save_request(request)

        # Audit event
        audit_event = AuditEvent(
            audit_id=self.store.generate_audit_id(),
            approval_id=approval_id,
            ticket_id=request.ticket_id,
            action="REJECTED",
            previous_state=prev_state,
            new_state=ApprovalStatus.REJECTED.value,
            actor_type=actor_type,
            actor_id=reviewer_id,
            reason=f"Priority change rejected by reviewer {reviewer_id}: {request.rejection_reason}",
            snapshot_id=request.snapshot.snapshot_id
        )
        self.store.append_audit_event(audit_event)

        return request

    def cancel(self, approval_id: str, actor_id: str, actor_type: ActorType = ActorType.USER) -> ApprovalRequest:
        """
        Cancels a PENDING_APPROVAL request.
        """
        request = self.store.get_request(approval_id)
        if not request:
            raise ApprovalError(f"Approval request '{approval_id}' not found.")

        if request.status == ApprovalStatus.CANCELLED:
            return request

        if request.status != ApprovalStatus.PENDING_APPROVAL:
            raise InvalidStateTransitionError(f"Cannot cancel request {approval_id} in state '{request.status.value}'.")

        prev_state = request.status.value
        request.status = ApprovalStatus.CANCELLED
        self.store.save_request(request)

        audit_event = AuditEvent(
            audit_id=self.store.generate_audit_id(),
            approval_id=approval_id,
            ticket_id=request.ticket_id,
            action="CANCELLED",
            previous_state=prev_state,
            new_state=ApprovalStatus.CANCELLED.value,
            actor_type=actor_type,
            actor_id=actor_id,
            reason="Approval request cancelled by user.",
            snapshot_id=request.snapshot.snapshot_id
        )
        self.store.append_audit_event(audit_event)
        return request

    def expire(self, approval_id: str) -> ApprovalRequest:
        """
        Expires a PENDING_APPROVAL request.
        """
        request = self.store.get_request(approval_id)
        if not request:
            raise ApprovalError(f"Approval request '{approval_id}' not found.")

        if request.status == ApprovalStatus.EXPIRED:
            return request

        if request.status != ApprovalStatus.PENDING_APPROVAL:
            raise InvalidStateTransitionError(f"Cannot expire request {approval_id} in state '{request.status.value}'.")

        prev_state = request.status.value
        request.status = ApprovalStatus.EXPIRED
        self.store.save_request(request)

        audit_event = AuditEvent(
            audit_id=self.store.generate_audit_id(),
            approval_id=approval_id,
            ticket_id=request.ticket_id,
            action="EXPIRED",
            previous_state=prev_state,
            new_state=ApprovalStatus.EXPIRED.value,
            actor_type=ActorType.SYSTEM,
            actor_id="SYSTEM",
            reason="Approval request expired due to age limit.",
            snapshot_id=request.snapshot.snapshot_id
        )
        self.store.append_audit_event(audit_event)
        return request

    def validate_approval(self, approval_id: str, current_ticket: Dict[str, Any]) -> Tuple[str, str]:
        """
        Validates whether recommendation snapshot matches live ticket priority baseline.
        Returns ("APPROVAL_VALID", reason) or ("APPROVAL_STALE", reason).
        """
        request = self.store.get_request(approval_id)
        if not request:
            return "INVALID_REQUEST", f"Approval request '{approval_id}' not found."

        live_priority = (current_ticket.get("priority") or "").upper()
        snapshot_priority = request.snapshot.original_priority.upper()

        if live_priority != snapshot_priority:
            msg = (
                f"Ticket priority changed in Lark Daily_Task (Live: {live_priority}, "
                f"Snapshot: {snapshot_priority}) since recommendation snapshot creation."
            )
            logger.warning(f"[ApprovalService] Stale detection for {approval_id}: {msg}")
            return "APPROVAL_STALE", msg

        return "APPROVAL_VALID", "Recommendation snapshot matches live ticket priority baseline."
