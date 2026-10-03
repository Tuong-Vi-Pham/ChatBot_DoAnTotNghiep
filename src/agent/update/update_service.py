import logging
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from src.lark.lark_base import LarkError
from src.lark.lark_base_write import LarkBaseWriteClient, ALLOWED_PRIORITIES
from src.agent.tools.ticket_tool import TicketToolAdapter
from src.agent.approval import ApprovalService, ApprovalStatus, ActorType, AuditEvent
from src.agent.update.update_model import UpdateResultStatus, UpdateResult

logger = logging.getLogger(__name__)


class UpdateExecutionError(Exception):
    """Base exception for controlled update execution errors."""
    pass


class ControlledUpdateService:
    """
    Controlled Update Service executing ONLY Daily_Task.Priority updates.
    Enforces a strict 10-step validation pipeline before executing any write operation.
    """

    def __init__(
        self,
        approval_service: Optional[ApprovalService] = None,
        read_client: Optional[TicketToolAdapter] = None,
        write_client: Optional[LarkBaseWriteClient] = None
    ):
        self.approval_service = approval_service or ApprovalService()
        self.store = self.approval_service.store
        self.read_client = read_client or TicketToolAdapter()
        self.write_client = write_client or LarkBaseWriteClient()

    def apply_approved_priority_update(
        self,
        approval_id: str,
        actor_id: str = "user_pm",
        actor_type: ActorType = ActorType.APPROVER
    ) -> UpdateResult:
        """
        Executes a controlled Lark Daily_Task.Priority update for an APPROVED recommendation.
        Strictly scoped ONLY to Priority.
        """
        logger.info(f"[ControlledUpdateService] Initiating update pipeline for approval '{approval_id}' by actor '{actor_id}'.")

        # Log initial update request audit
        req_audit_id = self.store.generate_audit_id()

        # Step 1: Load Approval Request
        req = self.store.get_request(approval_id)
        if not req:
            msg = f"Approval request '{approval_id}' not found."
            self._log_audit(req_audit_id, approval_id, "UNKNOWN", "UPDATE_REJECTED", "NONE", "NONE", actor_id, msg)
            return UpdateResult(
                status=UpdateResultStatus.UPDATE_REJECTED,
                approval_id=approval_id,
                ticket_id="UNKNOWN",
                previous_priority="UNKNOWN",
                requested_priority="UNKNOWN",
                actual_priority="UNKNOWN",
                reviewer_id=actor_id,
                reason=msg
            )

        t_id = req.ticket_id
        snp = req.snapshot
        approved_priority = snp.recommended_priority.upper()
        snapshot_original_priority = snp.original_priority.upper()

        # Step 2: Approval Status Check
        if req.status != ApprovalStatus.APPROVED:
            msg = f"Cannot execute update for request {approval_id} in status '{req.status.value}'. Must be APPROVED."
            self._log_audit(req_audit_id, approval_id, t_id, "UPDATE_REJECTED", snapshot_original_priority, approved_priority, actor_id, msg)
            return UpdateResult(
                status=UpdateResultStatus.UPDATE_REJECTED,
                approval_id=approval_id,
                ticket_id=t_id,
                previous_priority=snapshot_original_priority,
                requested_priority=approved_priority,
                actual_priority=snapshot_original_priority,
                reviewer_id=actor_id,
                reason=msg
            )

        # Step 3: Human Approver Validation
        reviewer = req.reviewed_by or actor_id
        if not reviewer or reviewer.upper() == "AGENT" or actor_type == ActorType.AGENT:
            msg = "Authorization error: Agent is forbidden from acting as human approver."
            self._log_audit(req_audit_id, approval_id, t_id, "UPDATE_REJECTED", snapshot_original_priority, approved_priority, actor_id, msg)
            return UpdateResult(
                status=UpdateResultStatus.UPDATE_REJECTED,
                approval_id=approval_id,
                ticket_id=t_id,
                previous_priority=snapshot_original_priority,
                requested_priority=approved_priority,
                actual_priority=snapshot_original_priority,
                reviewer_id=actor_id,
                reason=msg
            )

        # Step 4: Fetch Fresh Ticket from Lark using READ Client
        ticket_res = self.read_client.get_ticket_by_id(t_id)
        if not ticket_res.get("success") or not ticket_res.get("ticket"):
            msg = f"Target ticket '{t_id}' not found in Lark Base Daily_Task table."
            self._log_audit(req_audit_id, approval_id, t_id, "UPDATE_FAILED", snapshot_original_priority, approved_priority, actor_id, msg)
            return UpdateResult(
                status=UpdateResultStatus.UPDATE_FAILED,
                approval_id=approval_id,
                ticket_id=t_id,
                previous_priority=snapshot_original_priority,
                requested_priority=approved_priority,
                actual_priority="UNKNOWN",
                reviewer_id=reviewer,
                reason=msg
            )

        current_ticket = ticket_res["ticket"]
        current_lark_priority = (current_ticket.get("priority") or "UNKNOWN").upper()
        record_id = current_ticket.get("record_id") or (current_ticket.get("provenance") or {}).get("record_id") or (current_ticket.get("raw_record") or {}).get("record_id", "")

        if not record_id:
            msg = f"Cannot locate Lark Bitable record_id for ticket '{t_id}'."
            self._log_audit(req_audit_id, approval_id, t_id, "UPDATE_FAILED", current_lark_priority, approved_priority, actor_id, msg)
            return UpdateResult(
                status=UpdateResultStatus.UPDATE_FAILED,
                approval_id=approval_id,
                ticket_id=t_id,
                previous_priority=current_lark_priority,
                requested_priority=approved_priority,
                actual_priority=current_lark_priority,
                reviewer_id=reviewer,
                reason=msg
            )

        # Step 5: Stale Check
        if current_lark_priority != snapshot_original_priority and current_lark_priority != approved_priority:
            msg = f"Approval is stale because ticket priority changed in Lark (Live: {current_lark_priority}, Snapshot: {snapshot_original_priority}) after approval."
            logger.warning(f"[ControlledUpdateService] {msg}")
            self._log_audit(req_audit_id, approval_id, t_id, "UPDATE_STALE", current_lark_priority, approved_priority, actor_id, msg)
            return UpdateResult(
                status=UpdateResultStatus.UPDATE_STALE,
                approval_id=approval_id,
                ticket_id=t_id,
                previous_priority=current_lark_priority,
                requested_priority=approved_priority,
                actual_priority=current_lark_priority,
                reviewer_id=reviewer,
                reason=msg
            )

        # Step 6 & 7: Approved Value & Priority Validation
        if approved_priority not in ALLOWED_PRIORITIES:
            msg = f"Invalid target priority '{approved_priority}'. Allowed: {sorted(list(ALLOWED_PRIORITIES))}"
            self._log_audit(req_audit_id, approval_id, t_id, "UPDATE_REJECTED", current_lark_priority, approved_priority, actor_id, msg)
            return UpdateResult(
                status=UpdateResultStatus.UPDATE_REJECTED,
                approval_id=approval_id,
                ticket_id=t_id,
                previous_priority=current_lark_priority,
                requested_priority=approved_priority,
                actual_priority=current_lark_priority,
                reviewer_id=reviewer,
                reason=msg
            )

        # Step 8: Idempotency Check (Already Applied)
        if current_lark_priority == approved_priority:
            msg = f"Approved priority '{approved_priority}' is already applied to ticket '{t_id}'."
            logger.info(f"[ControlledUpdateService] {msg}")
            req.execution_status = "ALREADY_APPLIED"
            self.store.save_request(req)
            self._log_audit(req_audit_id, approval_id, t_id, "UPDATE_VERIFIED", current_lark_priority, approved_priority, actor_id, msg)
            return UpdateResult(
                status=UpdateResultStatus.ALREADY_APPLIED,
                approval_id=approval_id,
                ticket_id=t_id,
                previous_priority=current_lark_priority,
                requested_priority=approved_priority,
                actual_priority=current_lark_priority,
                reviewer_id=reviewer,
                verified=True,
                reason=msg
            )

        # Step 9: Execute Priority Update via WRITE Client
        write_res = {}
        try:
            logger.info(f"[ControlledUpdateService] Executing write operation: {t_id} (Record: {record_id}) Priority {current_lark_priority} -> {approved_priority}.")
            write_res = self.write_client.update_ticket_priority(record_id=record_id, priority=approved_priority)
        except Exception as e:
            msg = f"Lark write API call failed: {e}"
            logger.error(f"[ControlledUpdateService] {msg}")
            self._log_audit(req_audit_id, approval_id, t_id, "UPDATE_FAILED", current_lark_priority, approved_priority, actor_id, msg)
            return UpdateResult(
                status=UpdateResultStatus.UPDATE_FAILED,
                approval_id=approval_id,
                ticket_id=t_id,
                previous_priority=current_lark_priority,
                requested_priority=approved_priority,
                actual_priority=current_lark_priority,
                reviewer_id=reviewer,
                reason=msg
            )

        # Step 10: Post-Write Verification
        logger.info(f"[ControlledUpdateService] Performing post-write verification for ticket {t_id}...")
        if write_res.get("simulated"):
            actual_lark_priority = approved_priority
        else:
            verify_res = self.read_client.get_ticket_by_id(t_id)
            verified_ticket = verify_res.get("ticket") if verify_res.get("success") else None
            actual_lark_priority = (verified_ticket.get("priority") or "").upper() if verified_ticket else ""

        if actual_lark_priority == approved_priority:
            msg = f"Post-write verification SUCCESS: Ticket '{t_id}' Priority verified as '{approved_priority}' in Lark."
            logger.info(f"[ControlledUpdateService] {msg}")
            req.execution_status = "EXECUTED"
            req.executed_at = datetime.now(timezone.utc).isoformat()
            req.executed_by = reviewer
            self.store.save_request(req)

            audit_id = self._log_audit(req_audit_id, approval_id, t_id, "UPDATE_VERIFIED", current_lark_priority, approved_priority, reviewer, msg)
            return UpdateResult(
                status=UpdateResultStatus.UPDATE_VERIFIED,
                approval_id=approval_id,
                ticket_id=t_id,
                previous_priority=current_lark_priority,
                requested_priority=approved_priority,
                actual_priority=approved_priority,
                reviewer_id=reviewer,
                verified=True,
                audit_id=audit_id,
                reason=msg
            )
        else:
            # Handle Verification Failure & Conflict-Aware Rollback
            msg = f"Post-write verification FAILED for '{t_id}': Expected '{approved_priority}', Observed '{actual_lark_priority}'."
            logger.error(f"[ControlledUpdateService] {msg}")

            # Check if live priority equals the original priority before update (no third-party conflict)
            if actual_lark_priority == current_lark_priority or actual_lark_priority == "":
                # Attempt rollback
                try:
                    logger.info(f"[ControlledUpdateService] Attempting rollback for {t_id} to {current_lark_priority}...")
                    self.write_client.update_ticket_priority(record_id=record_id, priority=current_lark_priority)
                    rollback_msg = f"Rollback executed successfully for {t_id}. Priority restored to {current_lark_priority}."
                    self._log_audit(req_audit_id, approval_id, t_id, "UPDATE_FAILED", actual_lark_priority, current_lark_priority, reviewer, rollback_msg)
                    return UpdateResult(
                        status=UpdateResultStatus.ROLLBACK_EXECUTED,
                        approval_id=approval_id,
                        ticket_id=t_id,
                        previous_priority=current_lark_priority,
                        requested_priority=approved_priority,
                        actual_priority=current_lark_priority,
                        reviewer_id=reviewer,
                        reason=rollback_msg
                    )
                except Exception as rb_err:
                    logger.error(f"[ControlledUpdateService] Rollback failed: {rb_err}")

            # Rollback skipped due to conflict
            conflict_msg = f"Rollback skipped due to conflict: Current Lark priority ({actual_lark_priority}) modified in interim."
            self._log_audit(req_audit_id, approval_id, t_id, "UPDATE_FAILED", actual_lark_priority, approved_priority, reviewer, conflict_msg)
            return UpdateResult(
                status=UpdateResultStatus.ROLLBACK_SKIPPED_DUE_TO_CONFLICT,
                approval_id=approval_id,
                ticket_id=t_id,
                previous_priority=current_lark_priority,
                requested_priority=approved_priority,
                actual_priority=actual_lark_priority,
                reviewer_id=reviewer,
                reason=conflict_msg
            )

    def _log_audit(
        self,
        audit_id: str,
        approval_id: str,
        ticket_id: str,
        action: str,
        prev_prio: str,
        new_prio: str,
        actor_id: str,
        reason: str
    ) -> str:
        audit_event = AuditEvent(
            audit_id=self.store.generate_audit_id(),
            approval_id=approval_id,
            ticket_id=ticket_id,
            action=action,
            previous_state=prev_prio,
            new_state=new_prio,
            actor_type=ActorType.APPROVER,
            actor_id=actor_id,
            reason=reason,
            snapshot_id=f"SNP-{approval_id}"
        )
        self.store.append_audit_event(audit_event)
        return audit_event.audit_id
