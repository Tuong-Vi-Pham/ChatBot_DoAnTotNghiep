import logging
from typing import Any, Dict, List, Optional
from src.agent.priority.priority_model import FactorScore, PriorityAnalysisResult, VALID_PRIORITIES

logger = logging.getLogger(__name__)


def _get_val(obj: Any, key: str, default: Any = "") -> Any:
    if hasattr(obj, key):
        val = getattr(obj, key)
        return val if val is not None else default
    elif isinstance(obj, dict):
        return obj.get(key, default)
    return default


class PriorityRuleEvaluator:
    """
    Evaluates seven operational priority dimensions for a Daily_Task ticket.
    Separates CRM business/release evidence from Tech_Team architecture evidence.
    """

    @classmethod
    def evaluate_ticket(
        cls,
        ticket: Dict[str, Any],
        all_tickets: List[Dict[str, Any]],
        crm_evidence: Optional[List[Any]] = None,
        tech_evidence: Optional[List[Any]] = None
    ) -> PriorityAnalysisResult:
        crm_ev = crm_evidence or []
        tech_ev = tech_evidence or []

        t_id = ticket.get("ticket_id", "UNKNOWN")
        curr_prio = (ticket.get("priority") or "MEDIUM").strip().upper()

        dep_factor = cls._eval_dependency_impact(ticket, all_tickets)
        biz_factor = cls._eval_business_impact(ticket, crm_ev)
        sched_factor = cls._eval_schedule_urgency(ticket)
        rel_factor = cls._eval_release_impact(ticket, crm_ev)
        tech_factor = cls._eval_technical_impact(ticket, tech_ev)
        curr_factor = cls._eval_current_priority(curr_prio)
        eff_factor = cls._eval_effort_context(ticket)

        factors = {
            "DEPENDENCY": dep_factor,
            "BUSINESS": biz_factor,
            "SCHEDULE": sched_factor,
            "RELEASE": rel_factor,
            "TECHNICAL": tech_factor,
            "CURRENT_PRIORITY": curr_factor,
            "EFFORT": eff_factor
        }

        # Weights: DEP 0.25, BIZ 0.20, SCHED 0.20, REL 0.15, TECH 0.10, CURR 0.05, EFF 0.05
        weights = {
            "DEPENDENCY": 0.25,
            "BUSINESS": 0.20,
            "SCHEDULE": 0.20,
            "RELEASE": 0.15,
            "TECHNICAL": 0.10,
            "CURRENT_PRIORITY": 0.05,
            "EFFORT": 0.05
        }

        weighted_score = sum(f.score * weights.get(k, 0.0) for k, f in factors.items())
        weighted_score = round(weighted_score, 4)

        recommended_priority = cls.map_score_to_priority(weighted_score)
        differs = recommended_priority != curr_prio

        if crm_ev and tech_ev:
            confidence = "High"
        elif crm_ev or tech_ev:
            confidence = "Medium"
        else:
            confidence = "Low"

        key_reasons = []
        if dep_factor.score >= 0.70:
            key_reasons.append(dep_factor.explanation)
        if biz_factor.score >= 0.70:
            key_reasons.append(biz_factor.explanation)
        if sched_factor.score >= 0.70:
            key_reasons.append(sched_factor.explanation)
        if rel_factor.score >= 0.70:
            key_reasons.append(rel_factor.explanation)

        if not key_reasons:
            key_reasons.append(f"Multi-factor weighted evaluation score: {weighted_score:.2f}")

        summary_reason = "; ".join(key_reasons)

        return PriorityAnalysisResult(
            ticket_id=t_id,
            ticket_name=ticket.get("name", "Untitled"),
            current_priority=curr_prio,
            recommended_priority=recommended_priority,
            weighted_score=weighted_score,
            confidence=confidence,
            factors=factors,
            differs_from_current=differs,
            summary_reason=summary_reason
        )

    @classmethod
    def map_score_to_priority(cls, score: float) -> str:
        """
        Boundary Threshold Mapping:
        S >= 0.65 -> CRITICAL
        0.45 <= S < 0.65 -> HIGH
        S <  0.45 -> MEDIUM
        """
        if score >= 0.65:
            return "CRITICAL"
        elif score >= 0.45:
            return "HIGH"
        else:
            return "MEDIUM"

    @classmethod
    def _eval_dependency_impact(cls, ticket: Dict[str, Any], all_tickets: List[Dict[str, Any]]) -> FactorScore:
        t_id = ticket.get("ticket_id", "")
        sub_ids = ticket.get("sub_ticket_ids") or []
        parent_id = ticket.get("parent_ticket_id")

        # Check if other tickets list this ticket as parent
        blocker_count = len(sub_ids)
        for ot in all_tickets:
            if ot.get("parent_ticket_id") == t_id and ot.get("ticket_id") != t_id:
                blocker_count += 1

        if blocker_count >= 2:
            expl = f"Critical blocker for {blocker_count} downstream tasks"
            score = 1.0
        elif blocker_count == 1:
            expl = f"Blocks 1 downstream task ({sub_ids[0] if sub_ids else 'child task'})"
            score = 0.75
        elif parent_id:
            expl = f"Sub-task belonging to parent epic {parent_id}"
            score = 0.50
        elif sub_ids:
            score = 0.30
            expl = "Has sub-tickets"
        else:
            expl = "Standalone task with no detected blocking dependencies"
            score = 0.20

        return FactorScore(
            factor_name="DEPENDENCY",
            score=score,
            explanation=expl,
            evidence_sources=["Lark Daily_Task Relations"],
            evidence_quality="HIGH"
        )

    @classmethod
    def _eval_business_impact(cls, ticket: Dict[str, Any], crm_evidence: List[Any]) -> FactorScore:
        if not crm_evidence:
            return FactorScore(
                factor_name="BUSINESS",
                score=0.40,
                explanation="No relevant CRM documentation found for business verification",
                evidence_sources=[],
                evidence_quality="INSUFFICIENT_EVIDENCE"
            )

        top_ev = crm_evidence[0]
        content_lower = str(_get_val(top_ev, "content", "")).lower()
        score_raw = float(_get_val(top_ev, "relevance_score", 0.5))

        if any(k in content_lower for k in ["critical", "mandatory", "must have", "core", "srs", "brd"]):
            score = min(1.0, score_raw + 0.2)
            expl = f"High business impact verified in CRM ({_get_val(top_ev, 'source_path', 'BRD/SRS')})"
        else:
            score = round(score_raw, 2)
            expl = f"Moderate business relevance from CRM ({_get_val(top_ev, 'source_path', 'CRM')})"

        return FactorScore(
            factor_name="BUSINESS",
            score=score,
            explanation=expl,
            evidence_sources=[_get_val(top_ev, "source_path", "CRM RAG")],
            evidence_quality="HIGH" if score >= 0.70 else "MEDIUM"
        )

    @classmethod
    def _eval_schedule_urgency(cls, ticket: Dict[str, Any]) -> FactorScore:
        sprint = str(ticket.get("sprint") or "").lower()
        status = str(ticket.get("status") or "").lower()

        if status == "blocked":
            return FactorScore("SCHEDULE", 0.90, "Ticket is currently BLOCKED", ["Lark Status"], "HIGH")

        if "sprint 12" in sprint or "active" in sprint or "current" in sprint:
            score = 0.80
            expl = f"Active sprint commitment ({ticket.get('sprint')})"
        elif sprint:
            score = 0.50
            expl = f"Assigned to sprint {ticket.get('sprint')}"
        else:
            score = 0.30
            expl = "Backlog ticket with no active sprint assignment"

        return FactorScore("SCHEDULE", score, expl, ["Lark Sprint"], "HIGH")

    @classmethod
    def _eval_release_impact(cls, ticket: Dict[str, Any], crm_evidence: List[Any]) -> FactorScore:
        name = str(ticket.get("name") or "").lower()
        desc = str(ticket.get("description") or "").lower()
        rel_note = str(ticket.get("release_note") or "").lower()

        if any(k in name or k in desc or k in rel_note for k in ["release", "launch", "milestone", "go-live", "production", "v1."]):
            return FactorScore("RELEASE", 0.85, "Ticket directly impacts release milestone", ["Lark Title/Desc/ReleaseNote"], "HIGH")

        for ev in crm_evidence:
            ev_content = str(_get_val(ev, "content", "")).lower()
            if "release" in ev_content or "milestone" in ev_content:
                src_path = _get_val(ev, "source_path", "CRM")
                return FactorScore("RELEASE", 0.75, f"Associated with release milestone in {src_path}", [src_path], "MEDIUM")

        return FactorScore("RELEASE", 0.30, "No explicit release impact identified", [], "INSUFFICIENT_EVIDENCE")

    @classmethod
    def _eval_technical_impact(cls, ticket: Dict[str, Any], tech_evidence: List[Any]) -> FactorScore:
        if not tech_evidence:
            return FactorScore("TECHNICAL", 0.40, "No technical specification evidence retrieved", [], "INSUFFICIENT_EVIDENCE")

        top_ev = tech_evidence[0]
        content_lower = str(_get_val(top_ev, "content", "")).lower()
        score_raw = float(_get_val(top_ev, "relevance_score", 0.5))

        if any(k in content_lower for k in ["architecture", "security", "api", "database", "core"]):
            score = min(1.0, score_raw + 0.15)
            src_path = _get_val(top_ev, "source_path", "Tech_Team")
            expl = f"Core technical architectural impact verified in {src_path}"
        else:
            score = round(score_raw, 2)
            src_path = _get_val(top_ev, "source_path", "Tech_Team")
            expl = f"Technical impact verified in {src_path}"

        return FactorScore("TECHNICAL", score, expl, [_get_val(top_ev, "source_path", "Tech_Team")], "HIGH" if score >= 0.70 else "MEDIUM")

    @classmethod
    def _eval_current_priority(cls, current_prio: str) -> FactorScore:
        prio_map = {
            "P0": 1.0, "CRITICAL": 1.0, "URGENT": 1.0,
            "P1": 0.8, "HIGH": 0.8,
            "P2": 0.5, "MEDIUM": 0.5,
            "P3": 0.2, "LOW": 0.2
        }
        score = prio_map.get(current_prio.upper(), 0.5)
        return FactorScore("CURRENT_PRIORITY", score, f"Current human baseline priority is {current_prio}", ["Lark Priority"], "HIGH")

    @classmethod
    def _eval_effort_context(cls, ticket: Dict[str, Any]) -> FactorScore:
        sp = ticket.get("story_points")
        if sp is None:
            return FactorScore("EFFORT", 0.50, "Story points unassigned", [], "INSUFFICIENT_EVIDENCE")

        sp_val = float(sp)
        if sp_val <= 3.0:
            expl = f"Quick win / Low effort ({sp_val} pts)"
            score = 0.80
        elif sp_val <= 8.0:
            expl = f"Medium effort ({sp_val} pts)"
            score = 0.50
        else:
            expl = f"High effort / Major task ({sp_val} pts)"
            score = 0.30

        return FactorScore("EFFORT", score, expl, ["Lark Story Point"], "HIGH")
