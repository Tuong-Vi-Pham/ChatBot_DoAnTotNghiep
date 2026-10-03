from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


class ApprovalStatus(str, Enum):
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class ActorType(str, Enum):
    USER = "USER"
    APPROVER = "APPROVER"
    AGENT = "AGENT"
    SYSTEM = "SYSTEM"


@dataclass
class RecommendationSnapshot:
    """
    Immutable snapshot of priority recommendation captured at approval request creation.
    Prevents silent recalculation during approval decision.
    """
    snapshot_id: str
    ticket_id: str
    original_priority: str     # Baseline Priority from Lark Daily_Task (e.g. "CRITICAL", "HIGH", "MEDIUM")
    recommended_priority: str  # Agent Recommended Priority (e.g. "CRITICAL", "HIGH", "MEDIUM")
    weighted_score: float
    confidence: str            # "High" | "Medium" | "Low"
    summary_reason: str
    factors: Dict[str, Any] = field(default_factory=dict)
    evidence_references: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ApprovalRequest:
    """
    Human-in-the-loop priority change approval request object.
    """
    approval_id: str
    ticket_id: str
    snapshot: RecommendationSnapshot
    status: ApprovalStatus = ApprovalStatus.PENDING_APPROVAL
    requested_by: str = "AGENT"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    expires_at: Optional[str] = None
    execution_status: Optional[str] = None   # "EXECUTED" | "ALREADY_APPLIED" | None
    executed_at: Optional[str] = None
    executed_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["snapshot"] = self.snapshot.to_dict() if isinstance(self.snapshot, RecommendationSnapshot) else self.snapshot
        res["status"] = self.status.value if isinstance(self.status, ApprovalStatus) else self.status
        return res


@dataclass
class AuditEvent:
    """
    Append-only immutable audit log record for tracking approval lifecycle & update events.
    """
    audit_id: str
    approval_id: str
    ticket_id: str
    action: str                # "APPROVAL_REQUESTED" | "APPROVED" | "REJECTED" | "CANCELLED" | "EXPIRED" | "UPDATE_REQUESTED" | "UPDATE_VALIDATED" | "UPDATE_EXECUTED" | "UPDATE_VERIFIED" | "UPDATE_FAILED" | "UPDATE_REJECTED" | "UPDATE_STALE"
    previous_state: str
    new_state: str
    actor_type: ActorType
    actor_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reason: str = ""
    snapshot_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["actor_type"] = self.actor_type.value if isinstance(self.actor_type, ActorType) else self.actor_type
        return res
