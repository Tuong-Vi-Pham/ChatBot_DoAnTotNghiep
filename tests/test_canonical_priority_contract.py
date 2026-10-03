import unittest
from unittest.mock import MagicMock, patch

from src.agent.priority.priority_model import VALID_PRIORITIES, PriorityAnalysisResult
from src.agent.priority.priority_rules import PriorityRuleEvaluator
from src.agent.priority.priority_verifier import PriorityVerifier
from src.agent.approval import ApprovalService, ApprovalStore
from src.agent.approval.approval_store import reset_shared_approval_store
from src.agent.approval.approval_model import ApprovalStatus
from src.agent.update.update_service import ControlledUpdateService
from src.agent.update.update_model import UpdateResultStatus
from src.agent.tools.ticket_tool import TicketToolAdapter
from src.agent.tools.rag_tool import RAGToolAdapter
from src.agent.graph import AgentOrchestrator
from src.lark.lark_base_write import LarkBaseWriteClient, ALLOWED_PRIORITIES
from src.lark.lark_base import LarkAPIError


class TestCanonicalPriorityContract(unittest.TestCase):
    """
    Dedicated contract tests verifying the canonical priority contract:
    VALID_PRIORITIES = {"CRITICAL", "HIGH", "MEDIUM"}
    across Lark Read -> Priority Engine -> Recommendation -> Human Approval -> Controlled Update -> Lark Write.
    """

    def setUp(self):
        reset_shared_approval_store()
        self.store = ApprovalStore()
        self.approval_service = ApprovalService(store=self.store)

    # Test 1: Current CRITICAL -> Preserved CRITICAL -> Lark receives CRITICAL
    def test_01_current_critical_preserved_and_written(self):
        ticket = {
            "ticket_id": "CRM-044",
            "name": "Create Product Roadmap",
            "priority": "CRITICAL",
            "record_id": "rec_crm044",
            "sub_ticket_ids": ["CRM-045", "CRM-046"]
        }
        res = PriorityRuleEvaluator.evaluate_ticket(ticket, [ticket], [], [])
        self.assertEqual(res.current_priority, "CRITICAL")
        self.assertEqual(ticket["priority"], "CRITICAL")

        # Snapshot and execute update with approved priority CRITICAL
        mock_read = MagicMock(spec=TicketToolAdapter)
        mock_read.get_ticket_by_id.side_effect = [
            {"success": True, "ticket": dict(ticket, priority="HIGH")}, # was HIGH before update
            {"success": True, "ticket": dict(ticket, priority="CRITICAL")} # post-write
        ]
        mock_write = MagicMock(spec=LarkBaseWriteClient)
        mock_write.update_ticket_priority.return_value = {
            "success": True,
            "record_id": "rec_crm044",
            "updated_priority": "CRITICAL"
        }

        update_svc = ControlledUpdateService(
            approval_service=self.approval_service,
            read_client=mock_read,
            write_client=mock_write
        )

        req = self.approval_service.create_approval_request(
            ticket=dict(ticket, priority="HIGH"),
            priority_analysis={"ticket_id": "CRM-044", "current_priority": "HIGH", "recommended_priority": "CRITICAL", "weighted_score": 0.85}
        )
        self.approval_service.approve(req.approval_id, reviewer_id="user_pm")

        update_res = update_svc.apply_approved_priority_update(req.approval_id, actor_id="user_pm")
        self.assertEqual(update_res.status, UpdateResultStatus.UPDATE_VERIFIED)
        self.assertEqual(update_res.requested_priority, "CRITICAL")
        self.assertEqual(update_res.actual_priority, "CRITICAL")
        mock_write.update_ticket_priority.assert_called_once_with(record_id="rec_crm044", priority="CRITICAL")

    # Test 2: Current HIGH, Recommendation CRITICAL -> Lark receives CRITICAL
    def test_02_current_high_recommendation_critical(self):
        ticket = {
            "ticket_id": "CRM-044",
            "name": "Australian Retirement Fund Core Integration",
            "priority": "HIGH",
            "record_id": "rec_crm044",
            "sub_ticket_ids": ["CRM-045", "CRM-046"],
            "sprint": "Sprint 12",
            "release_note": "Required for production release"
        }
        crm_ev = [{
            "source": "CRM", "document_id": "DOC-1",
            "content": "Mandatory core compliance requirement.",
            "relevance_score": 0.90
        }]
        res = PriorityRuleEvaluator.evaluate_ticket(ticket, [ticket], crm_evidence=crm_ev, tech_evidence=[])
        self.assertEqual(res.current_priority, "HIGH")
        self.assertEqual(res.recommended_priority, "CRITICAL")
        self.assertTrue(res.differs_from_current)
        self.assertTrue(res.weighted_score >= 0.65)

    # Test 3: Current MEDIUM, Recommendation HIGH -> Lark receives HIGH
    def test_03_current_medium_recommendation_high(self):
        # A ticket with moderate score (0.45 <= S < 0.65)
        ticket = {
            "ticket_id": "TASK-202",
            "name": "Audit Log Export Enhancement",
            "priority": "MEDIUM",
            "record_id": "rec_task202",
            "sprint": "Sprint 12",
            "sub_ticket_ids": ["TASK-203"], # 1 blocker -> score 0.75
            "story_points": 5.0
        }
        res = PriorityRuleEvaluator.evaluate_ticket(ticket, [ticket], crm_evidence=[], tech_evidence=[])
        self.assertEqual(res.current_priority, "MEDIUM")
        self.assertEqual(res.recommended_priority, "HIGH")
        self.assertTrue(0.45 <= res.weighted_score < 0.65)
        self.assertTrue(res.differs_from_current)

    # Test 4: Explicit assertion that CRITICAL != P1, HIGH != P2, MEDIUM != P3
    def test_04_no_p_priorities_in_system(self):
        self.assertEqual(VALID_PRIORITIES, {"CRITICAL", "HIGH", "MEDIUM"})
        self.assertEqual(ALLOWED_PRIORITIES, {"CRITICAL", "HIGH", "MEDIUM"})

        self.assertNotEqual("CRITICAL", "P1")
        self.assertNotEqual("CRITICAL", "P0")
        self.assertNotEqual("HIGH", "P2")
        self.assertNotEqual("MEDIUM", "P3")

        # Test all scores map to one of VALID_PRIORITIES
        for s in [0.0, 0.2, 0.44, 0.45, 0.55, 0.64, 0.65, 0.85, 1.0]:
            prio = PriorityRuleEvaluator.map_score_to_priority(s)
            self.assertIn(prio, VALID_PRIORITIES)
            self.assertNotIn(prio, {"P0", "P1", "P2", "P3"})

    # Test 5: Invalid priority P1 rejected with error
    def test_05_invalid_priority_p1_rejected(self):
        write_client = LarkBaseWriteClient(app_token="app_test", table_id="tbl_test")

        for legacy_prio in ["P0", "P1", "P2", "P3", "LOW", "P9"]:
            with self.assertRaises(LarkAPIError) as ctx:
                write_client.update_ticket_priority("rec101", legacy_prio)
            self.assertIn(f"Invalid priority '{legacy_prio}'", str(ctx.exception))

        # Also rejected by ControlledUpdateService
        ticket = {"ticket_id": "CRM-044", "priority": "MEDIUM", "record_id": "rec_crm044"}
        req = self.approval_service.create_approval_request(
            ticket=ticket,
            priority_analysis={"ticket_id": "CRM-044", "current_priority": "MEDIUM", "recommended_priority": "P1"}
        )
        self.approval_service.approve(req.approval_id, reviewer_id="user_pm")

        mock_read = MagicMock(spec=TicketToolAdapter)
        mock_read.get_ticket_by_id.return_value = {"success": True, "ticket": ticket}
        update_svc = ControlledUpdateService(
            approval_service=self.approval_service,
            read_client=mock_read,
            write_client=write_client
        )
        res = update_svc.apply_approved_priority_update(req.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_REJECTED)
        self.assertIn("Invalid target priority 'P1'", res.reason)

    # Test 6: Approval integrity across full pipeline
    def test_06_approval_integrity_pipeline(self):
        ticket = {
            "ticket_id": "CRM-044",
            "name": "Create Product Roadmap",
            "priority": "HIGH",
            "record_id": "rec_crm044"
        }
        analysis = {
            "ticket_id": "CRM-044",
            "current_priority": "HIGH",
            "recommended_priority": "CRITICAL",
            "weighted_score": 0.88,
            "confidence": "High",
            "summary_reason": "Strategic roadmap milestone"
        }
        req = self.approval_service.create_approval_request(ticket=ticket, priority_analysis=analysis)
        self.assertEqual(req.snapshot.original_priority, "HIGH")
        self.assertEqual(req.snapshot.recommended_priority, "CRITICAL")
        self.assertEqual(req.status, ApprovalStatus.PENDING_APPROVAL)

        # Approve
        self.approval_service.approve(req.approval_id, reviewer_id="user_pm")
        approved_req = self.store.get_request(req.approval_id)
        self.assertEqual(approved_req.status, ApprovalStatus.APPROVED)
        self.assertEqual(approved_req.snapshot.recommended_priority, "CRITICAL")

        # Verify audit event
        log = self.store.get_audit_log(req.approval_id)
        self.assertEqual(len(log), 2)
        self.assertEqual(log[1].action, "APPROVED")
        self.assertEqual(log[1].new_state, "APPROVED")


if __name__ == "__main__":
    unittest.main()
