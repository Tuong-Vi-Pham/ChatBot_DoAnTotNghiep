from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


VALID_PRIORITIES = {"CRITICAL", "HIGH", "MEDIUM"}


@dataclass
class FactorScore:
    """
    Evaluation score and rationale for a single priority factor.
    """
    factor_name: str          # "BUSINESS" | "DEPENDENCY" | "SCHEDULE" | "RELEASE" | "TECHNICAL" | "CURRENT_PRIORITY" | "EFFORT"
    score: float              # Normalized score 0.0 - 1.0
    explanation: str          # Rationale for score
    evidence_sources: List[str] = field(default_factory=list)
    evidence_quality: str = "HIGH"  # "HIGH" | "MEDIUM" | "LOW" | "INSUFFICIENT_EVIDENCE"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PriorityAnalysisResult:
    """
    Structured priority evaluation result for a single ticket.
    Preserves current_priority from Lark Daily_Task and computes recommended_priority separately.
    """
    ticket_id: str
    ticket_name: str
    current_priority: str     # "CRITICAL" | "HIGH" | "MEDIUM" (From Lark Daily_Task - NEVER OVERWRITTEN)
    recommended_priority: str # "CRITICAL" | "HIGH" | "MEDIUM" (Computed Agent Recommendation)
    weighted_score: float     # Total aggregated score 0.0 - 1.0
    confidence: str           # "High" | "Medium" | "Low"
    factors: Dict[str, FactorScore] = field(default_factory=dict)
    summary_reason: str = ""
    differs_from_current: bool = False

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["factors"] = {k: v.to_dict() if isinstance(v, FactorScore) else v for k, v in self.factors.items()}
        return res


@dataclass
class RankedTicket:
    """
    Ranked candidate ticket item in multi-ticket priority ranking.
    """
    rank: int
    ticket_id: str
    ticket_name: str
    current_priority: str
    recommended_priority: str
    weighted_score: float
    confidence: str
    summary_reason: str
    differs_from_current: bool
    analysis: PriorityAnalysisResult

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["analysis"] = self.analysis.to_dict() if isinstance(self.analysis, PriorityAnalysisResult) else self.analysis
        return res
