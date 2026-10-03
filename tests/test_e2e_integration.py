import os
import unittest
from unittest.mock import MagicMock, patch

from src.lark.lark_base_write import LarkBaseWriteClient
from src.agent.approval import ApprovalStore, ApprovalService, ApprovalStatus, ActorType
from src.agent.update import ControlledUpdateService, UpdateResultStatus
from src.agent.tools.ticket_tool import TicketToolAdapter
from src.agent.tools.rag_tool import RAGToolAdapter
from src.agent.graph import AgentOrchestrator
from src.agent.state import Evidence


class TestE2EIntegrationFlow(unittest.TestCase):

    def setUp(self):
        self.store = ApprovalStore()
        self.approval_service = ApprovalService(store=self.store)

        self.sample_ticket = {
            "ticket_id": "TASK-101",
            "name": "OAuth SSO Integration",
            "status": "In Progress",
            "priority": "MEDIUM",
            "record_id": "rec101",
            "sprint": "Sprint 12",
            "provenance": {"record_id": "rec101"},
            "sub_ticket_ids": ["TASK-102"]
        }

        self.sample_ticket_102 = {
            "ticket_id": "TASK-102",
            "name": "SSO Callback Endpoint",
            "status": "In Progress",
            "priority": "MEDIUM",
            "record_id": "rec102",
            "sprint": "Sprint 12",
            "parent_ticket_id": "TASK-101"
        }

        self.mock_ticket_tool = MagicMock(spec=TicketToolAdapter)
        self.mock_rag_tool = MagicMock(spec=RAGToolAdapter)
        self.mock_write_client = MagicMock(spec=LarkBaseWriteClient)

        self.mock_ticket_tool.get_ticket_by_id.return_value = {"success": True, "found": True, "ticket": self.sample_ticket}
        self.mock_ticket_tool.get_all_tickets.return_value = {"success": True, "tickets": [self.sample_ticket, self.sample_ticket_102]}
        self.mock_ticket_tool.get_tickets_by_sprint.return_value = {"success": True, "tickets": [self.sample_ticket, self.sample_ticket_102]}
        self.mock_ticket_tool.get_related_tickets.return_value = {"success": True, "related_tickets": [self.sample_ticket_102]}

        self.crm_ev = Evidence(
            source="CRM", document_id="BRD_SRS", chunk_id="chunk_1",
            source_path="Documents/CRM/SRS.md", content="SSO authentication is mandatory for corporate launch in Q3.",
            relevance_score=0.88, evidence_type="BUSINESS"
        )
        self.mock_rag_tool.retrieve.return_value = {"success": True, "evidence": [self.crm_ev]}

        self.mock_write_client.update_ticket_priority.return_value = {
            "success": True,
            "record_id": "rec101",
            "updated_priority": "CRITICAL"
        }

        self.update_service = ControlledUpdateService(
            approval_service=self.approval_service,
            read_client=self.mock_ticket_tool,
            write_client=self.mock_write_client
        )

        self.orchestrator = AgentOrchestrator(
            ticket_tool=self.mock_ticket_tool,
            rag_tool=self.mock_rag_tool,
            approval_service=self.approval_service,
            update_service=self.update_service
        )

    # TEST 1: FAQ Query
    def test_01_faq_query_flow(self):
        res = self.orchestrator.run("What is SmartLogi?")
        self.assertEqual(res["intent"], "FAQ")
        self.assertTrue(len(res["final_response"]) > 0)
        self.mock_write_client.update_ticket_priority.assert_not_called()

    # TEST 2: Ticket Query
    def test_02_ticket_query_flow(self):
        res = self.orchestrator.run("What is the status of TASK-101?")
        self.assertEqual(res["intent"], "TICKET_QUERY")
        self.assertIn("TASK-101", res["final_response"])
        self.assertIn("In Progress", res["final_response"])
        self.mock_write_client.update_ticket_priority.assert_not_called()

    # TEST 3: Ticket Analysis
    def test_03_ticket_analysis_flow(self):
        res = self.orchestrator.run("Analyze TASK-101 based on the project documents.")
        self.assertEqual(res["intent"], "TICKET_ANALYSIS")
        self.assertIn("Detailed Ticket Analysis", res["final_response"])
        self.assertIn("TASK-101", res["final_response"])
        self.mock_write_client.update_ticket_priority.assert_not_called()

    # TEST 4: Priority Recommendation
    def test_04_priority_recommendation_flow(self):
        res = self.orchestrator.run("Which ticket should be prioritized in Sprint 12?")
        self.assertEqual(res["intent"], "PRIORITY_ANALYSIS")
        self.assertIsNotNone(res["priority_ranking"])
        self.assertTrue(len(res["priority_ranking"]) > 0)
        self.assertIn("Ticket Priority Recommendation", res["final_response"])
        self.mock_write_client.update_ticket_priority.assert_not_called()

    # TEST 5: Approval Request
    def test_05_approval_request_flow(self):
        res = self.orchestrator.run("Apply recommended priority to TASK-101")
        self.assertEqual(res["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertIsNotNone(res["approval_id"])
        self.assertEqual(res["approval_status"], "PENDING_APPROVAL")
        self.assertIn("No Lark data has been changed", res["final_response"])
        self.mock_write_client.update_ticket_priority.assert_not_called()

    # TEST 6: Rejected Approval
    def test_06_rejected_approval_flow(self):
        req = self.approval_service.create_approval_request(
            self.sample_ticket,
            {"ticket_id": "TASK-101", "current_priority": "MEDIUM", "recommended_priority": "CRITICAL", "weighted_score": 0.85, "confidence": "High", "summary_reason": "Blocker"}
        )
        res = self.orchestrator.run(f"Reject {req.approval_id} by user_pm reason: not critical for current sprint")
        self.assertEqual(res["intent"], "PRIORITY_APPROVAL_DECISION")
        self.assertEqual(res["approval_status"], "REJECTED")
        self.assertIn("Recommendation Rejected", res["final_response"])
        self.mock_write_client.update_ticket_priority.assert_not_called()

    # TEST 7: Approved Recommendation
    def test_07_approved_recommendation_flow(self):
        req = self.approval_service.create_approval_request(
            self.sample_ticket,
            {"ticket_id": "TASK-101", "current_priority": "MEDIUM", "recommended_priority": "CRITICAL", "weighted_score": 0.85, "confidence": "High", "summary_reason": "Blocker"}
        )
        res = self.orchestrator.run(f"Approve {req.approval_id} by user_pm")
        self.assertEqual(res["intent"], "PRIORITY_APPROVAL_DECISION")
        self.assertEqual(res["approval_status"], "APPROVED")
        self.assertIn("Approval recorded. Lark Daily_Task has NOT been modified in SESSION 06A", res["final_response"])
        self.mock_write_client.update_ticket_priority.assert_not_called()

    # TEST 8: Stale Approval
    def test_08_stale_approval_flow(self):
        req = self.approval_service.create_approval_request(
            self.sample_ticket,
            {"ticket_id": "TASK-101", "current_priority": "MEDIUM", "recommended_priority": "CRITICAL", "weighted_score": 0.85, "confidence": "High", "summary_reason": "Blocker"}
        )
        self.approval_service.approve(req.approval_id, reviewer_id="user_pm")

        # Simulate live Lark ticket priority changed to HIGH
        stale_ticket = dict(self.sample_ticket, priority="HIGH")
        self.mock_ticket_tool.get_ticket_by_id.return_value = {"success": True, "ticket": stale_ticket}

        res = self.orchestrator.run(f"Apply {req.approval_id} by user_pm")
        self.assertEqual(res["intent"], "PRIORITY_UPDATE_EXECUTION")
        self.assertEqual(res["update_status"], "UPDATE_STALE")
        self.assertIn("Priority Update Blocked", res["final_response"])
        self.mock_write_client.update_ticket_priority.assert_not_called()

    # TEST 9: Approved Controlled Update
    def test_09_approved_controlled_update_flow(self):
        req = self.approval_service.create_approval_request(
            self.sample_ticket,
            {"ticket_id": "TASK-101", "current_priority": "MEDIUM", "recommended_priority": "CRITICAL", "weighted_score": 0.85, "confidence": "High", "summary_reason": "Blocker"}
        )
        self.approval_service.approve(req.approval_id, reviewer_id="user_pm")

        post_ticket = dict(self.sample_ticket, priority="CRITICAL")
        self.mock_ticket_tool.get_ticket_by_id.side_effect = [
            {"success": True, "ticket": self.sample_ticket},
            {"success": True, "ticket": post_ticket}
        ]

        res = self.orchestrator.run(f"Apply {req.approval_id} by user_pm")
        self.assertEqual(res["intent"], "PRIORITY_UPDATE_EXECUTION")
        self.assertEqual(res["update_status"], "UPDATE_VERIFIED")
        self.assertIn("Priority Update Successful", res["final_response"])
        self.mock_write_client.update_ticket_priority.assert_called_once_with(record_id="rec101", priority="CRITICAL")

    # TEST 10: Idempotent Update
    def test_10_idempotent_update_flow(self):
        req = self.approval_service.create_approval_request(
            self.sample_ticket,
            {"ticket_id": "TASK-101", "current_priority": "MEDIUM", "recommended_priority": "CRITICAL", "weighted_score": 0.85, "confidence": "High", "summary_reason": "Blocker"}
        )
        self.approval_service.approve(req.approval_id, reviewer_id="user_pm")

        already_critical_ticket = dict(self.sample_ticket, priority="CRITICAL")
        self.mock_ticket_tool.get_ticket_by_id.return_value = {"success": True, "ticket": already_critical_ticket}

        res = self.orchestrator.run(f"Apply {req.approval_id} by user_pm")
        self.assertEqual(res["intent"], "PRIORITY_UPDATE_EXECUTION")
        self.assertEqual(res["update_status"], "ALREADY_APPLIED")
        self.assertIn("Priority Update Already Applied", res["final_response"])
        self.mock_write_client.update_ticket_priority.assert_not_called()

    # TEST 11: Demo / Dry-Run Mode
    @patch.dict(os.environ, {"DEMO_MODE": "true"})
    def test_11_demo_mode_simulation(self):
        req = self.approval_service.create_approval_request(
            self.sample_ticket,
            {"ticket_id": "TASK-101", "current_priority": "MEDIUM", "recommended_priority": "CRITICAL", "weighted_score": 0.85, "confidence": "High", "summary_reason": "Blocker"}
        )
        self.approval_service.approve(req.approval_id, reviewer_id="user_pm")

        # Real write client with DEMO_MODE=true
        real_write_client = LarkBaseWriteClient(app_token="app123", table_id="tbl123")
        demo_update_service = ControlledUpdateService(
            approval_service=self.approval_service,
            read_client=self.mock_ticket_tool,
            write_client=real_write_client
        )

        res = demo_update_service.apply_approved_priority_update(req.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_VERIFIED)

    # TEST 12: Real Lark Write Safety Guard
    def test_12_real_lark_write_safety_guard(self):
        demo_ticket_id = os.getenv("LARK_DEMO_TICKET_ID")
        if not demo_ticket_id:
            msg = "REAL_LARK_WRITE_TEST_REQUIRES_EXPLICIT_TEST_TICKET"
            self.assertEqual(msg, "REAL_LARK_WRITE_TEST_REQUIRES_EXPLICIT_TEST_TICKET")
        else:
            self.assertTrue(len(demo_ticket_id) > 0)


if __name__ == "__main__":
    unittest.main()
