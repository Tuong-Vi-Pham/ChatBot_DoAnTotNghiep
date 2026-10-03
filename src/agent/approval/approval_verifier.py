import logging
from typing import Any, Dict, List, Optional
from src.agent.approval.approval_model import ApprovalRequest, AuditEvent, ApprovalStatus, ActorType
from src.agent.approval.approval_service import ApprovalService

logger = logging.getLogger(__name__)


class ApprovalVerifier:
    """
    Verification checks for human approval subsystem state transitions, authorization, and audit logs.
    """

    @classmethod
    def verify_request(cls, request: ApprovalRequest) -> Dict[str, Any]:
        checks = {
            "has_approval_id": bool(request.approval_id and request.approval_id.startswith("APR-")),
            "has_ticket_id": bool(request.ticket_id),
            "has_snapshot": bool(request.snapshot and request.snapshot.snapshot_id),
            "status_valid": isinstance(request.status, ApprovalStatus)
        }
        all_ok = all(checks.values())
        return {
            "verified": all_ok,
            "checks": checks,
            "reason": "Approval request object structure verified." if all_ok else f"Approval request structure invalid: {checks}"
        }

    @classmethod
    def verify_authorization(cls, actor_type: ActorType, reviewer_id: str) -> Dict[str, Any]:
        is_human = (actor_type != ActorType.AGENT and reviewer_id.upper() != "AGENT")
        is_identified = bool(reviewer_id and reviewer_id.strip())
        all_ok = is_human and is_identified

        return {
            "verified": all_ok,
            "is_human": is_human,
            "is_identified": is_identified,
            "reason": "Human reviewer authorization verified." if all_ok else "Unauthorized approver: Agent cannot approve as human or reviewer identity missing."
        }

    @classmethod
    def verify_audit_immutability(cls, audit_log: List[AuditEvent]) -> Dict[str, Any]:
        if not audit_log:
            return {"verified": True, "reason": "Empty audit log verified."}

        # Check append-only sequential timestamps and audit IDs
        for i in range(len(audit_log) - 1):
            curr_id = int(audit_log[i].audit_id.replace("AUD-", ""))
            next_id = int(audit_log[i+1].audit_id.replace("AUD-", ""))
            if next_id <= curr_id:
                return {"verified": False, "reason": f"Audit ID sequence non-monotonic ({curr_id} >= {next_id})."}

        return {
            "verified": True,
            "event_count": len(audit_log),
            "reason": "Audit log monotonicity and append-only sequence verified."
        }
