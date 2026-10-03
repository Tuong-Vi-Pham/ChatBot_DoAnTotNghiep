from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime, timezone


class UpdateResultStatus(str, Enum):
    UPDATE_VERIFIED = "UPDATE_VERIFIED"
    UPDATE_STALE = "UPDATE_STALE"
    UPDATE_REJECTED = "UPDATE_REJECTED"
    UPDATE_FAILED = "UPDATE_FAILED"
    ALREADY_APPLIED = "ALREADY_APPLIED"
    ROLLBACK_EXECUTED = "ROLLBACK_EXECUTED"
    ROLLBACK_SKIPPED_DUE_TO_CONFLICT = "ROLLBACK_SKIPPED_DUE_TO_CONFLICT"


@dataclass
class UpdateResult:
    """
    Structured execution result object for controlled priority updates.
    """
    status: UpdateResultStatus
    approval_id: str
    ticket_id: str
    previous_priority: str
    requested_priority: str
    actual_priority: str
    reviewer_id: str
    verified: bool = False
    audit_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reason: str = ""
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["status"] = self.status.value if isinstance(self.status, UpdateResultStatus) else self.status
        return res
