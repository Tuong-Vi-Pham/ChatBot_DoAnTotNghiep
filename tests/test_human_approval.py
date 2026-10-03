import unittest
import threading
from unittest.mock import MagicMock

from src.agent.approval import (
    ApprovalStore,
    ApprovalService,
    ApprovalStatus,
    ActorType,
    ApprovalError,
    InvalidStateTransitionError,
    UnauthorizedApproverError,
    ApprovalVerifier,
)
from src.agent.state import AgentState, Evidence
from src.agent.tools.ticket_tool import TicketToolAdapter
from src.agent.tools.rag_tool import RAGToolAdapter
from src.agent.graph import AgentOrchestrator
from src.lark.lark_base import LarkBaseClient


class TestHumanApprovalArchitecture(unittest.TestCase):

    def setUp(self):
        self.store = ApprovalStore()
        self.service = ApprovalService(store=self.store)

        self.sample_ticket = {
            "ticket_id": "TASK-101",
            "name": "OAuth SSO Integration",
            "status": "In Progress",
            "priority": "MEDIUM",
            "sprint": "Sprint 12",
            "sub_ticket_ids": ["TASK-102"]
        }

        self.sample_analysis = {
            "ticket_id": "TASK-101",
            "current_priority": "MEDIUM",
            "recommended_priority": "CRITICAL",
            "weighted_score": 0.82,
            "confidence": "High",
            "summary_reason": "Critical blocker for downstream tasks.",
            "factors": {},
            "project_context": []
        }

        self.mock_ticket_tool = MagicMock(spec=TicketToolAdapter)
        self.mock_rag_tool = MagicMock(spec=RAGToolAdapter)

    # 1. Create Approval Request
    def test_01_create_approval_request(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        self.assertIsNotNone(req)
        self.assertTrue(req.approval_id.startswith("APR-"))
        self.assertEqual(req.status, ApprovalStatus.PENDING_APPROVAL)
        self.assertEqual(req.snapshot.original_priority, "MEDIUM")
        self.assertEqual(req.snapshot.recommended_priority, "CRITICAL")

    # 2. Retrieve Approval Request
    def test_02_retrieve_approval_request(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        retrieved = self.store.get_request(req.approval_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.approval_id, req.approval_id)

    # 3. Approve Request
    def test_03_approve_request(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        approved_req = self.service.approve(req.approval_id, reviewer_id="user_cuong", actor_type=ActorType.APPROVER)
        self.assertEqual(approved_req.status, ApprovalStatus.APPROVED)
        self.assertEqual(approved_req.reviewed_by, "user_cuong")

    # 4. Reject Request
    def test_04_reject_request(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        rejected_req = self.service.reject(req.approval_id, reviewer_id="user_pm", rejection_reason="Not needed in Sprint 12")
        self.assertEqual(rejected_req.status, ApprovalStatus.REJECTED)
        self.assertEqual(rejected_req.rejection_reason, "Not needed in Sprint 12")

    # 5. Cancel Request
    def test_05_cancel_request(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        cancelled_req = self.service.cancel(req.approval_id, actor_id="user_cuong")
        self.assertEqual(cancelled_req.status, ApprovalStatus.CANCELLED)

    # 6. Expire Request
    def test_06_expire_request(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        expired_req = self.service.expire(req.approval_id)
        self.assertEqual(expired_req.status, ApprovalStatus.EXPIRED)

    # 7. Invalid State Transition
    def test_07_invalid_state_transition(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        self.service.cancel(req.approval_id, actor_id="user_cuong")
        with self.assertRaises(InvalidStateTransitionError):
            self.service.approve(req.approval_id, reviewer_id="user_pm")

    # 8. Double Approval Protection
    def test_08_double_approval_protection(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        req1 = self.service.approve(req.approval_id, reviewer_id="user_pm")
        req2 = self.service.approve(req.approval_id, reviewer_id="user_pm")
        self.assertEqual(req1.status, ApprovalStatus.APPROVED)
        self.assertEqual(req2.status, ApprovalStatus.APPROVED)

    # 9. Reject After Approval
    def test_09_reject_after_approval(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        self.service.approve(req.approval_id, reviewer_id="user_pm")
        with self.assertRaises(InvalidStateTransitionError):
            self.service.reject(req.approval_id, reviewer_id="user_pm", rejection_reason="Changed mind")

    # 10. Approve After Rejection
    def test_10_approve_after_rejection(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        self.service.reject(req.approval_id, reviewer_id="user_pm", rejection_reason="No budget")
        with self.assertRaises(InvalidStateTransitionError):
            self.service.approve(req.approval_id, reviewer_id="user_pm")

    # 11. Missing Reviewer
    def test_11_missing_reviewer(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        with self.assertRaises(UnauthorizedApproverError):
            self.service.approve(req.approval_id, reviewer_id="")

    # 12. Unauthorized Reviewer (Agent cannot approve as human)
    def test_12_unauthorized_reviewer(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        with self.assertRaises(UnauthorizedApproverError):
            self.service.approve(req.approval_id, reviewer_id="AGENT", actor_type=ActorType.AGENT)

    # 13. Recommendation Snapshot Preservation
    def test_13_recommendation_snapshot_preservation(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        snapshot = req.snapshot
        self.assertEqual(snapshot.original_priority, "MEDIUM")
        self.assertEqual(snapshot.recommended_priority, "CRITICAL")
        self.assertEqual(snapshot.weighted_score, 0.82)

    # 14. Stale Recommendation Detection
    def test_14_stale_recommendation_detection(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        
        # Live ticket priority unchanged
        status_valid, _ = self.service.validate_approval(req.approval_id, self.sample_ticket)
        self.assertEqual(status_valid, "APPROVAL_VALID")

        # Live ticket priority updated in Lark
        modified_ticket = dict(self.sample_ticket, priority="HIGH")
        status_stale, reason = self.service.validate_approval(req.approval_id, modified_ticket)
        self.assertEqual(status_stale, "APPROVAL_STALE")
        self.assertIn("Ticket priority changed", reason)

    # 15. Audit Event Creation
    def test_15_audit_event_creation(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        self.service.approve(req.approval_id, reviewer_id="user_cuong")
        log = self.store.get_audit_log(req.approval_id)
        self.assertTrue(len(log) >= 2)
        actions = [e.action for e in log]
        self.assertIn("APPROVAL_REQUESTED", actions)
        self.assertIn("APPROVED", actions)

    # 16. Audit Append-Only Behavior
    def test_16_audit_append_only_behavior(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        self.service.approve(req.approval_id, reviewer_id="user_cuong")
        log = self.store.get_audit_log()
        res = ApprovalVerifier.verify_audit_immutability(log)
        self.assertTrue(res["verified"])

    # 17. Audit Contains Correct States
    def test_17_audit_contains_correct_states(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        log = self.store.get_audit_log(req.approval_id)
        self.assertEqual(log[0].previous_state, "NONE")
        self.assertEqual(log[0].new_state, "PENDING_APPROVAL")

    # 18. Agent Cannot Approve as Human
    def test_18_agent_cannot_approve_as_human(self):
        v = ApprovalVerifier.verify_authorization(ActorType.AGENT, "agent")
        self.assertFalse(v["verified"])
        self.assertFalse(v["is_human"])

    # 19. Approval Does Not Modify Lark
    def test_19_approval_does_not_modify_lark(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        self.service.approve(req.approval_id, reviewer_id="user_cuong")
        # Ticket dictionary in Python remains untouched
        self.assertEqual(self.sample_ticket["priority"], "MEDIUM")

    # 20. Lark Write Methods Do Not Exist
    def test_20_lark_write_methods_do_not_exist(self):
        client = LarkBaseClient()
        self.assertFalse(hasattr(client, "update_record"))
        self.assertFalse(hasattr(client, "create_record"))
        self.assertFalse(hasattr(client, "delete_record"))

    # 21. Duplicate Approval Detection
    def test_21_duplicate_approval_detection(self):
        req1 = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        req2 = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        self.assertEqual(req1.approval_id, req2.approval_id)

    # 22. Concurrent Approval Simulation
    def test_22_concurrent_approval_simulation(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)

        results = []
        def approve_task(rev_id):
            try:
                res = self.service.approve(req.approval_id, reviewer_id=rev_id)
                results.append((True, res))
            except Exception as e:
                results.append((False, str(e)))

        t1 = threading.Thread(target=approve_task, args=("user_a",))
        t2 = threading.Thread(target=approve_task, args=("user_b",))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both completed safely under lock
        self.assertEqual(len(results), 2)
        self.assertEqual(req.status, ApprovalStatus.APPROVED)

    # 23. Approval Expiration
    def test_23_approval_expiration(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        exp = self.service.expire(req.approval_id)
        self.assertEqual(exp.status, ApprovalStatus.EXPIRED)

    # 24. Cancelled Approval
    def test_24_cancelled_approval(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        can = self.service.cancel(req.approval_id, actor_id="user_cuong")
        self.assertEqual(can.status, ApprovalStatus.CANCELLED)

    # 25. End-to-End Recommendation -> Approval Request
    def test_25_e2e_recommendation_to_approval_request(self):
        self.mock_ticket_tool.get_ticket_by_id.return_value = {"success": True, "found": True, "ticket": self.sample_ticket}
        orchestrator = AgentOrchestrator(ticket_tool=self.mock_ticket_tool, rag_tool=self.mock_rag_tool, approval_service=self.service)

        res = orchestrator.run("Apply recommended priority to TASK-101")
        self.assertEqual(res["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertIsNotNone(res["approval_id"])
        self.assertEqual(res["approval_status"], "PENDING_APPROVAL")
        self.assertIn("No Lark data has been changed", res["final_response"])

    # 26. End-to-End Recommendation -> Rejection
    def test_26_e2e_recommendation_to_rejection(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        orchestrator = AgentOrchestrator(ticket_tool=self.mock_ticket_tool, rag_tool=self.mock_rag_tool, approval_service=self.service)

        res = orchestrator.run(f"Reject {req.approval_id} by user_pm reason: not critical for current release")
        self.assertEqual(res["intent"], "PRIORITY_APPROVAL_DECISION")
        self.assertEqual(res["approval_status"], "REJECTED")
        self.assertIn("Recommendation Rejected", res["final_response"])

    # 27. End-to-End Recommendation -> Approval
    def test_27_e2e_recommendation_to_approval(self):
        req = self.service.create_approval_request(self.sample_ticket, self.sample_analysis)
        orchestrator = AgentOrchestrator(ticket_tool=self.mock_ticket_tool, rag_tool=self.mock_rag_tool, approval_service=self.service)

        res = orchestrator.run(f"Approve {req.approval_id} by user_pm")
        self.assertEqual(res["intent"], "PRIORITY_APPROVAL_DECISION")
        self.assertEqual(res["approval_status"], "APPROVED")
        self.assertIn("Approval recorded. Lark Daily_Task has NOT been modified in SESSION 06A", res["final_response"])

    # 28. Existing FAQ Flow Regression
    def test_28_existing_faq_flow_regression(self):
        orchestrator = AgentOrchestrator(ticket_tool=self.mock_ticket_tool, rag_tool=self.mock_rag_tool, approval_service=self.service)
        res = orchestrator.run("What is SmartLogi?")
        self.assertEqual(res["intent"], "FAQ")

    # 29. Existing RAG Flow Regression
    def test_29_existing_rag_flow_regression(self):
        orchestrator = AgentOrchestrator(ticket_tool=self.mock_ticket_tool, rag_tool=self.mock_rag_tool, approval_service=self.service)
        res = orchestrator.run("What does the BRD requirement say about investment risk?")
        self.assertEqual(res["intent"], "GENERAL_RAG")

    # 30. Existing Ticket Query Regression
    def test_30_existing_ticket_query_regression(self):
        self.mock_ticket_tool.get_ticket_by_id.return_value = {"success": True, "found": True, "ticket": self.sample_ticket}
        orchestrator = AgentOrchestrator(ticket_tool=self.mock_ticket_tool, rag_tool=self.mock_rag_tool, approval_service=self.service)
        res = orchestrator.run("What is the status of TASK-101?")
        self.assertEqual(res["intent"], "TICKET_QUERY")

    # 31. Existing Priority Analysis Regression
    def test_31_existing_priority_analysis_regression(self):
        self.mock_ticket_tool.get_all_tickets.return_value = {"success": True, "tickets": [self.sample_ticket]}
        orchestrator = AgentOrchestrator(ticket_tool=self.mock_ticket_tool, rag_tool=self.mock_rag_tool, approval_service=self.service)
        res = orchestrator.run("Which unfinished tickets should we prioritize first?")
        self.assertEqual(res["intent"], "PRIORITY_ANALYSIS")


if __name__ == "__main__":
    unittest.main()
