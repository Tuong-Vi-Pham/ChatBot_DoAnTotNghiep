import logging
from typing import Any, Dict, List, Optional
from src.agent.state import AgentState

logger = logging.getLogger(__name__)


def _get_val(obj: Any, key: str, default: Any = "") -> Any:
    if hasattr(obj, key):
        val = getattr(obj, key)
        return val if val is not None else default
    elif isinstance(obj, dict):
        return obj.get(key, default)
    return default


class ResponseGeneratorNode:
    """
    LangGraph Node formatting user-facing responses with structured provenance citations,
    priority analysis reports, approval notifications, and controlled update status.
    """

    @classmethod
    def execute(cls, state: AgentState) -> AgentState:
        generator = cls()
        response_text = generator.generate(state)
        state["final_response"] = response_text
        logger.info(f"[ResponseGeneratorNode] Final response generated (Length: {len(response_text)} chars).")
        return state

    def generate(self, state: AgentState) -> str:
        intent = state.get("intent", "GENERAL_RAG")
        verification = state.get("verification_result", {})
        ticket_context = state.get("ticket_context", [])
        project_context = state.get("project_context", [])
        query = state.get("user_query", "")
        priority_analysis = state.get("priority_analysis")
        priority_ranking = state.get("priority_ranking") or []
        approval_req = state.get("approval_request")
        update_res = state.get("update_result")
        err = state.get("update_error") or state.get("approval_error") or state.get("error")

        if err:
            logger.warning(f"[ResponseGeneratorNode] Returning error response: {err}")
            return f"Operation Error: {err}"

        if not verification.get("verified", False) and intent not in ("CLARIFICATION_REQUIRED", "PRIORITY_APPROVAL_REQUEST", "PRIORITY_APPROVAL_DECISION", "PRIORITY_UPDATE_EXECUTION"):
            msg = verification.get("reason", "I could not find sufficient information in the knowledge base.")
            if intent == "PRIORITY_ANALYSIS":
                return (
                    f"### Ticket Priority Analysis\n\n"
                    f"{msg}\n\n"
                    f"*Source: Lark Base (Daily_Task table)*"
                )
            return f"Insufficient Data: {msg}"

        approval_id = state.get("approval_id")

        if intent == "PRIORITY_UPDATE_EXECUTION":
            return self._format_update_execution_response(update_res)
        elif intent == "PRIORITY_APPROVAL_DECISION":
            return self._format_approval_decision_response(approval_req)
        elif intent == "PRIORITY_APPROVAL_REQUEST":
            if update_res:
                return self._format_update_execution_response(update_res)
            elif approval_req and approval_req.get("status") in ("REJECTED", "CANCELLED", "APPROVED"):
                return self._format_approval_decision_response(approval_req)
            return self._format_approval_request_response(approval_req)
        elif intent == "PRIORITY_ANALYSIS":
            return self._format_priority_analysis_response(
                query=query,
                tickets=ticket_context,
                project_context=project_context,
                priority_ranking=priority_ranking,
                priority_analysis=priority_analysis,
                approval_id=approval_id
            )
        elif intent == "TICKET_ANALYSIS":
            return self._format_ticket_analysis_response(
                query=query,
                tickets=ticket_context,
                project_context=project_context,
                priority_analysis=priority_analysis
            )
        elif intent == "TICKET_QUERY":
            return self._format_ticket_query_response(query=query, tickets=ticket_context)
        elif intent == "CLARIFICATION_REQUIRED":
            response = state.get("final_response") or "Please specify which team, sprint, or ticket ID you would like to analyze."
            return response
        else:
            return self._format_rag_response(query=query, project_context=project_context)

    def _format_update_execution_response(self, update_res: Optional[Dict[str, Any]]) -> str:
        if not update_res:
            return "Update execution details unavailable."

        res = update_res
        status = res.get("status")
        t_id = res.get("ticket_id")
        apr_id = res.get("approval_id")
        prev_prio = res.get("previous_priority")
        new_prio = res.get("requested_priority")
        actual_prio = res.get("actual_priority")
        reviewer = res.get("reviewer_id")
        reason = res.get("reason", "")

        if status == "UPDATE_VERIFIED":
            lines = [
                "### Priority Update Successful\n",
                f"**Ticket:** `{t_id}`\n",
                f"- **Previous Priority:** `{prev_prio}`",
                f"- **New Priority:** `{new_prio}`",
                f"- **Approval ID:** `{apr_id}`",
                f"- **Reviewer:** `{reviewer}`",
                "- **Verification:** `SUCCESS` (Verified in Lark Daily_Task Bitable table)\n",
                "*Audit Log Recorded.*"
            ]
        elif status == "UPDATE_STALE":
            lines = [
                "### Priority Update Blocked\n",
                "**Reason:** Approval is stale because the ticket priority changed after approval.\n",
                f"**Expected Priority:** `{prev_prio}`",
                f"**Current Priority:** `{actual_prio}`",
                f"**Approval ID:** `{apr_id}`\n",
                "*No overwrite was performed.*"
            ]
        elif status == "ALREADY_APPLIED":
            lines = [
                "### Priority Update Already Applied\n",
                f"**Ticket:** `{t_id}`\n",
                f"**Approved Priority:** `{new_prio}` (Already set in Lark Daily_Task)",
                f"**Approval ID:** `{apr_id}`\n",
                "*No duplicate write call was performed.*"
            ]
        elif status == "UPDATE_REJECTED":
            lines = [
                "### Update Rejected\n",
                f"**Reason:** {reason}\n",
                "*Lark Daily_Task remains UNCHANGED.*"
            ]
        else:
            lines = [
                f"### Update Status: {status}\n",
                f"**Ticket:** `{t_id}` | **Approval ID:** `{apr_id}`",
                f"**Details:** {reason}"
            ]

        return "\n".join(lines)

    def _format_approval_request_response(self, req: Optional[Dict[str, Any]]) -> str:
        if not req:
            return "Failed to create approval request."

        snp = req.get("snapshot", {})
        lines = [
            "### Priority Change Request\n",
            f"**Ticket:** `{req.get('ticket_id')}`\n",
            f"- **Current Priority:** `{snp.get('original_priority')}`",
            f"- **Recommended Priority:** `{snp.get('recommended_priority')}`",
            f"- **Confidence:** `{snp.get('confidence')}`\n",
            "**Reason:**",
            f"- {snp.get('summary_reason')}\n",
            f"**Approval ID:** `{req.get('approval_id')}`",
            f"**Status:** `{req.get('status')}`\n",
            "*No Lark data has been changed.*"
        ]
        return "\n".join(lines)

    def _format_approval_decision_response(self, req: Optional[Dict[str, Any]]) -> str:
        if not req:
            return "Approval decision details unavailable."

        status = req.get("status")
        snp = req.get("snapshot", {})
        t_id = req.get("ticket_id")

        if status == "APPROVED":
            lines = [
                "### Recommendation Approved\n",
                f"**Ticket:** `{t_id}`\n",
                f"- **Current Priority:** `{snp.get('original_priority')}`",
                f"- **Approved Priority:** `{snp.get('recommended_priority')}`",
                f"- **Decision:** `{status}`",
                f"- **Reviewer:** `{req.get('reviewed_by')}`\n",
                "*Approval recorded. Lark Daily_Task has NOT been modified in SESSION 06A.*"
            ]
        elif status == "REJECTED":
            lines = [
                "### Recommendation Rejected\n",
                f"**Ticket:** `{t_id}`\n",
                f"- **Current Priority:** `{snp.get('original_priority')}`",
                f"- **Decision:** `{status}`",
                f"- **Reviewer:** `{req.get('reviewed_by')}`",
                f"- **Rejection Reason:** {req.get('rejection_reason') or 'No reason provided.'}\n",
                "*Lark Daily_Task remains UNCHANGED.*"
            ]
        elif status == "CANCELLED":
            lines = [
                "### Recommendation Cancelled\n",
                f"**Ticket:** `{t_id}`\n",
                f"- **Current Priority:** `{snp.get('original_priority')}`",
                f"- **Decision:** `{status}`",
                f"- **Cancelled By:** `{req.get('reviewed_by') or req.get('requested_by')}`\n",
                "*Lark Daily_Task remains UNCHANGED.*"
            ]
        else:
            lines = [
                f"### Approval Request Details ({t_id})\n",
                f"**Approval ID:** `{req.get('approval_id')}` | **Status:** `{status}`"
            ]

        return "\n".join(lines)

    def _format_ticket_query_response(self, query: str, tickets: List[Dict[str, Any]]) -> str:
        if not tickets:
            return "No matching tickets found in Lark Daily_Task."

        lines = [f"### Lark Daily_Task Ticket Query Results ({len(tickets)} tickets found)\n"]
        for t in tickets:
            t_id = t.get("ticket_id", "UNKNOWN")
            name = t.get("name", "Untitled")
            status = t.get("status", "UNKNOWN")
            assignee = t.get("assignee") or "Unassigned"
            sp = t.get("story_points")
            sp_str = f"{sp} pts" if sp is not None else "N/A"
            sprint = t.get("sprint") or "Unassigned"
            priority = t.get("priority", "UNKNOWN")

            lines.append(f"- **[{t_id}] {name}**")
            lines.append(f"  - **Status:** `{status}` | **Current Priority:** `{priority}` | **Story Points:** `{sp_str}`")
            lines.append(f"  - **Assignee:** {assignee} | **Sprint:** {sprint}\n")

        lines.append("\n*Source: Lark Base (Daily_Task table)*")
        return "\n".join(lines)

    def _format_ticket_analysis_response(
        self,
        query: str,
        tickets: List[Dict[str, Any]],
        project_context: List[Any],
        priority_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        if not tickets:
            return "Ticket data unavailable for analysis."

        t = tickets[0]
        t_id = t.get("ticket_id", "UNKNOWN")
        name = t.get("name", "Untitled")

        lines = [f"### Detailed Ticket Analysis: [{t_id}] {name}\n"]
        lines.append(f"- **Status:** `{t.get('status')}` | **Current Priority:** `{t.get('priority')}` | **Size:** `{t.get('size') or 'N/A'}`")
        lines.append(f"- **Assignee:** {t.get('assignee') or 'Unassigned'} | **Sprint:** {t.get('sprint') or 'N/A'}")
        lines.append(f"- **Description:** {t.get('description') or 'No description provided.'}")

        if priority_analysis:
            rec_prio = priority_analysis.get("recommended_priority")
            curr_prio = priority_analysis.get("current_priority")
            differs = priority_analysis.get("differs_from_current", False)
            conf = priority_analysis.get("confidence", "Medium")
            reason = priority_analysis.get("summary_reason", "")

            lines.append("\n#### Priority Recommendation:")
            lines.append(f"- **Current Priority (Daily_Task):** `{curr_prio}`")
            lines.append(f"- **Agent Recommended Priority:** `{rec_prio}` ({'Differs from Current' if differs else 'Matches Current'})")
            lines.append(f"- **Recommendation Confidence:** `{conf}`")
            lines.append(f"- **Key Rationale:** {reason}")

        if t.get("sub_ticket_ids"):
            lines.append(f"- **Sub-tickets:** {', '.join(t.get('sub_ticket_ids'))}")
        if t.get("parent_ticket_id"):
            lines.append(f"- **Parent Epic/Ticket:** {t.get('parent_ticket_id')}")

        if project_context:
            lines.append("\n#### Cross-referenced Project Knowledge:")
            for ev in project_context[:2]:
                lines.append(f"  - *[{_get_val(ev, 'source')}: {_get_val(ev, 'source_path')}]* {str(_get_val(ev, 'content'))[:120]}...")

        lines.append("\n*Sources: Lark Base (Daily_Task), CRM & Tech_Team RAG*")
        return "\n".join(lines)

    def _format_priority_analysis_response(
        self,
        query: str,
        tickets: List[Dict[str, Any]],
        project_context: List[Any],
        priority_ranking: Optional[List[Dict[str, Any]]] = None,
        priority_analysis: Optional[Dict[str, Any]] = None,
        approval_id: Optional[str] = None
    ) -> str:
        ranking = priority_ranking or []

        if not ranking and not tickets:
            return (
                "### Ticket Priority Analysis\n\n"
                "Unable to determine the highest-priority ticket because no eligible tickets were retrieved "
                "from Lark Daily_Task for the specified project and sprint.\n\n"
                "*Source: Lark Base (Daily_Task table)*"
            )

        lines = ["### Ticket Priority Recommendation & Ranking\n"]

        # 1. Recommended Ticket Section
        if ranking:
            top = ranking[0]
            top_id = top.get("ticket_id", "UNKNOWN")
            top_name = top.get("ticket_name") or top.get("name", "Untitled")
            top_curr = top.get("current_priority", "UNKNOWN")
            top_rec = top.get("recommended_priority", "UNKNOWN")
            top_score = top.get("weighted_score", 0.0)
            top_conf = top.get("confidence", "Medium")
            top_reason = top.get("summary_reason", "")

            lines.append("### Recommended Ticket\n")
            lines.append(f"**[{top_id}] {top_name}**")
            lines.append(f"- **Recommended Priority:** `{top_rec}`")
            lines.append(f"- **Current Priority (Daily_Task):** `{top_curr}` (Preserved)")
            lines.append(f"- **Priority Score:** `{top_score:.4f}` | **Confidence:** `{top_conf}`\n")

            # 2. Priority Ranking Section
            lines.append("### Priority Ranking\n")
            lines.append("| Rank | Ticket | Title | Current Priority | Recommended Priority | Score | Key Rationale |")
            lines.append("|---|---|---|---|---|---|---|")

            for item in ranking[:10]:
                rank = item.get("rank")
                t_id = item.get("ticket_id")
                t_title = item.get("ticket_name") or item.get("name", "Untitled")
                curr = item.get("current_priority")
                rec = item.get("recommended_priority")
                score = item.get("weighted_score", 0.0)
                reason = item.get("summary_reason", "")
                if len(reason) > 80:
                    reason = reason[:80] + "..."
                lines.append(f"| {rank} | `{t_id}` | {t_title} | `{curr}` | **`{rec}`** | `{score:.4f}` | {reason} |")

            # 3. Recommendation Rationale
            lines.append("\n### Recommendation Rationale\n")
            lines.append(f"- **Primary Justification:** {top_reason}")
            if priority_analysis:
                factors = priority_analysis.get("factors", {})
                if "DEPENDENCY" in factors and factors["DEPENDENCY"].get("score", 0) >= 0.70:
                    lines.append(f"- **Dependency Impact:** {factors['DEPENDENCY'].get('explanation')}")
                if "RELEASE" in factors and factors["RELEASE"].get("score", 0) >= 0.70:
                    lines.append(f"- **Release Milestone Impact:** {factors['RELEASE'].get('explanation')}")
                if "BUSINESS" in factors and factors["BUSINESS"].get("score", 0) >= 0.70:
                    lines.append(f"- **Business Value:** {factors['BUSINESS'].get('explanation')}")
                if "SCHEDULE" in factors and factors["SCHEDULE"].get("score", 0) >= 0.70:
                    lines.append(f"- **Schedule Urgency:** {factors['SCHEDULE'].get('explanation')}")
        else:
            lines.append("### Priority Candidates\n")
            for idx, t in enumerate(tickets[:5], start=1):
                t_id = t.get("ticket_id", "UNKNOWN")
                name = t.get("name", "Untitled")
                prio = t.get("priority", "UNKNOWN")
                status = t.get("status", "UNKNOWN")
                lines.append(f"{idx}. **[{t_id}] {name}** - Current Priority `{prio}`, Status `{status}`")

        # 4. Current vs Recommended Priority Policy
        lines.append("\n### Current vs Recommended Priority")
        lines.append("- **Current Priority (Lark Daily_Task):** Preserved without modification.")
        lines.append("- **Recommended Priority:** Advisory recommendation derived from multi-factor analysis.")
        lines.append("- *No Lark Daily_Task records have been modified.*")

        differs = (priority_analysis and priority_analysis.get("differs_from_current")) or (ranking and ranking[0].get("differs_from_current"))
        if approval_id and differs:
            target_id = ranking[0].get("ticket_id") if ranking else (tickets[0].get("ticket_id") if tickets else "UNKNOWN")
            curr_p = ranking[0].get("current_priority") if ranking else (priority_analysis.get("current_priority") if priority_analysis else "UNKNOWN")
            rec_p = ranking[0].get("recommended_priority") if ranking else (priority_analysis.get("recommended_priority") if priority_analysis else "UNKNOWN")

            lines.append("\n### Human Approval Required\n")
            lines.append("Recommendation requires human approval before modifying Lark Daily_Task.\n")
            lines.append(f"- **Ticket:** `{target_id}`")
            lines.append(f"- **Current Priority:** `{curr_p}`")
            lines.append(f"- **Recommended Priority:** `{rec_p}`")
            lines.append(f"- **Approval ID:** `{approval_id}`")
            lines.append("- **Status:** `PENDING_APPROVAL`\n")
            lines.append("Please respond with **Approve** (to apply this update to Lark) or **Reject** (to keep the current priority).")

        if project_context:
            lines.append("\n#### Supporting Specification Evidence:")
            for ev in project_context[:3]:
                lines.append(f"- **[{_get_val(ev, 'source')}: {_get_val(ev, 'document_id')}]**: {str(_get_val(ev, 'content'))[:120]}...")

        lines.append("\n*Sources: Lark Base (Daily_Task table), CRM & Tech_Team RAG*")
        return "\n".join(lines)

    def _format_rag_response(self, query: str, project_context: List[Any]) -> str:
        if not project_context:
            return "No relevant documentation found."

        lines = [f"### Knowledge Base Answer for: '{query}'\n"]
        top_ev = project_context[0]
        lines.append(str(_get_val(top_ev, "content")))

        lines.append("\n\n#### Sources:")
        seen = set()
        for ev in project_context:
            src_key = f"{_get_val(ev, 'source')}: {_get_val(ev, 'source_path')}"
            if src_key not in seen:
                seen.add(src_key)
                rel_score = float(_get_val(ev, 'relevance_score', 0.0))
                lines.append(f"- [{_get_val(ev, 'source')}] `{_get_val(ev, 'source_path')}` (Relevance: {rel_score:.2f})")

        return "\n".join(lines)
