import logging
import threading
from typing import Any, Dict, List, Optional
from src.agent.approval.approval_model import ApprovalRequest, AuditEvent, ApprovalStatus

logger = logging.getLogger(__name__)


class ApprovalStore:
    """
    Thread-safe repository for approval requests and append-only audit log events.
    Supports in-memory dictionary storage with thread-safe RLock protection.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._requests: Dict[str, ApprovalRequest] = {}
        self._audit_log: List[AuditEvent] = []
        self._approval_counter: int = 0
        self._audit_counter: int = 0
        self._session_pending: Dict[str, str] = {}

    def set_session_pending_approval(self, session_id: str, approval_id: str) -> None:
        with self._lock:
            self._session_pending[session_id] = approval_id
            logger.info(f"[ApprovalStore] Session '{session_id}' mapped to pending approval '{approval_id}'.")

    def get_session_pending_approval(self, session_id: str) -> Optional[str]:
        with self._lock:
            return self._session_pending.get(session_id)

    def clear_session_pending_approval(self, session_id: str) -> None:
        with self._lock:
            self._session_pending.pop(session_id, None)

    def generate_approval_id(self) -> str:
        with self._lock:
            self._approval_counter += 1
            return f"APR-{self._approval_counter:03d}"

    def generate_audit_id(self) -> str:
        with self._lock:
            self._audit_counter += 1
            return f"AUD-{self._audit_counter:03d}"

    def save_request(self, request: ApprovalRequest) -> None:
        with self._lock:
            self._requests[request.approval_id] = request
            logger.info(f"[ApprovalStore] Saved request {request.approval_id} for ticket {request.ticket_id} (Status: {request.status.value}).")

    def get_request(self, approval_id: str) -> Optional[ApprovalRequest]:
        with self._lock:
            return self._requests.get(approval_id)

    def find_pending_by_ticket(self, ticket_id: str) -> Optional[ApprovalRequest]:
        with self._lock:
            for req in self._requests.values():
                if req.ticket_id == ticket_id and req.status == ApprovalStatus.PENDING_APPROVAL:
                    return req
            return None

    def list_requests(self) -> List[ApprovalRequest]:
        with self._lock:
            return list(self._requests.values())

    def append_audit_event(self, event: AuditEvent) -> None:
        with self._lock:
            self._audit_log.append(event)
            logger.info(
                f"[ApprovalStore] Append Audit Event [{event.audit_id}]: "
                f"Action={event.action}, ApprovalID={event.approval_id}, "
                f"Transition=({event.previous_state} -> {event.new_state}), Actor={event.actor_id}"
            )

    def get_audit_log(self, approval_id: Optional[str] = None) -> List[AuditEvent]:
        with self._lock:
            if approval_id:
                return [e for e in self._audit_log if e.approval_id == approval_id]
            return list(self._audit_log)


_SHARED_APPROVAL_STORE: Optional[ApprovalStore] = None
_STORE_INIT_LOCK = threading.RLock()


def get_shared_approval_store() -> ApprovalStore:
    """Returns the shared module-level ApprovalStore singleton."""
    global _SHARED_APPROVAL_STORE
    with _STORE_INIT_LOCK:
        if _SHARED_APPROVAL_STORE is None:
            _SHARED_APPROVAL_STORE = ApprovalStore()
        return _SHARED_APPROVAL_STORE


def reset_shared_approval_store() -> ApprovalStore:
    """Resets the shared module-level ApprovalStore singleton (for testing)."""
    global _SHARED_APPROVAL_STORE
    with _STORE_INIT_LOCK:
        _SHARED_APPROVAL_STORE = ApprovalStore()
        return _SHARED_APPROVAL_STORE

