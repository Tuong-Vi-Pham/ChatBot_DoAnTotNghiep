import unittest
from unittest.mock import MagicMock, patch

from src.agent.state import AgentState
from src.agent.nodes.intent_router import IntentRouterNode
from src.agent.approval import ApprovalStore, ApprovalService, ApprovalStatus, ActorType
from src.agent.approval.approval_store import reset_shared_approval_store
from src.agent.update import ControlledUpdateService, UpdateResultStatus
from src.agent.graph import AgentOrchestrator
from src.agent.tools.ticket_tool import TicketToolAdapter
from src.agent.tools.rag_tool import RAGToolAdapter
from src.lark.lark_base_write import LarkBaseWriteClient


class TestApprovalVocabulary(unittest.TestCase):
    """
    Validates all 14 approval vocabulary variations + negative test case.
    """

    def _route(self, query: str) -> AgentState:
        state: AgentState = {
            "user_query": query,
            "intent": "GENERAL_RAG",
            "scope": "BOTH"
        }
        return IntentRouterNode.execute(state)

    def test_01_approve_recommendation_for_ticket(self):
        state = self._route("Approve this recommendation for CRM-044.")
        self.assertEqual(state["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertEqual(state["approval_action"], "APPROVE")

    def test_02_approve_recommendation_short(self):
        state = self._route("Approve recommendation for CRM-044")
        self.assertEqual(state["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertEqual(state["approval_action"], "APPROVE")

    def test_03_approve_ticket(self):
        state = self._route("Approve CRM-044")
        self.assertEqual(state["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertEqual(state["approval_action"], "APPROVE")

    def test_04_approve_single_word(self):
        state = self._route("Approve")
        self.assertEqual(state["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertEqual(state["approval_action"], "APPROVE")

    def test_05_i_approve(self):
        state = self._route("I approve")
        self.assertEqual(state["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertEqual(state["approval_action"], "APPROVE")

    def test_06_yes_update_ticket(self):
        state = self._route("Yes, update CRM-044")
        self.assertEqual(state["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertEqual(state["approval_action"], "APPROVE")

    def test_07_proceed_with_update(self):
        state = self._route("Proceed with the update")
        self.assertEqual(state["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertEqual(state["approval_action"], "APPROVE")

    def test_08_go_ahead_with_update(self):
        state = self._route("Go ahead with the update")
        self.assertEqual(state["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertEqual(state["approval_action"], "APPROVE")

    def test_09_confirm_the_update(self):
        state = self._route("Confirm the update")
        self.assertEqual(state["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertEqual(state["approval_action"], "APPROVE")

    def test_10_reject_recommendation(self):
        state = self._route("Reject this recommendation")
        self.assertEqual(state["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertEqual(state["approval_action"], "REJECT")

    def test_11_reject_single_word(self):
        state = self._route("Reject")
        self.assertEqual(state["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertEqual(state["approval_action"], "REJECT")

    def test_12_no_dont_update(self):
        state = self._route("No, don't update it")
        self.assertEqual(state["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertEqual(state["approval_action"], "REJECT")

    def test_13_keep_current_priority(self):
        state = self._route("Keep the current priority")
        self.assertEqual(state["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertEqual(state["approval_action"], "REJECT")

    def test_14_cancel(self):
        state = self._route("Cancel")
        self.assertEqual(state["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertEqual(state["approval_action"], "CANCEL")

    def test_15_negative_analyze_crm_044(self):
        """
        Turn 1 query MUST route to PRIORITY_ANALYSIS, NOT PRIORITY_APPROVAL_REQUEST.
        """
        state = self._route("Analyze CRM-044 and recommend whether its priority should be changed.")
        self.assertEqual(state["intent"], "PRIORITY_ANALYSIS")
        self.assertNotEqual(state["intent"], "PRIORITY_APPROVAL_REQUEST")


class TestE2EHumanApprovalFlow(unittest.TestCase):
    """
    Validates the end-to-end Turn 1 (Analysis & Request Creation) -> Turn 2 (Approval & Controlled Execution).
    """

    def setUp(self):
        reset_shared_approval_store()
        self.store = ApprovalStore()
        self.approval_service = ApprovalService(store=self.store)

        self.mock_ticket = {
            "ticket_id": "CRM-044",
            "name": "Superannuation Rollover Gateway Latency",
            "status": "In Progress",
            "priority": "HIGH",
            "record_id": "rec_crm044",
            "sprint": "Sprint 14",
            "parent_ticket_id": "EPIC-003",
            "sub_ticket_ids": ["TASK-101", "TASK-102"],
            "provenance": {"record_id": "rec_crm044"}
        }

        self.mock_evidence = [
            {
                "content": "Regulatory compliance requirement APRA SPS 515 requires transaction response under 2 seconds.",
                "source": "CRM",
                "source_path": "docs/specs/BRD_Australian_Pension.md",
                "document_id": "DOC-BRD-001",
                "relevance_score": 0.95
            },
            {
                "content": "Gateway timeout causing direct financial penalty and SLA breach for Australian Super fund.",
                "source": "Tech_Team",
                "source_path": "docs/architecture/Gateway_Latency_SRS.md",
                "document_id": "DOC-SRS-002",
                "relevance_score": 0.92
            }
        ]

        self.mock_read_client = MagicMock(spec=TicketToolAdapter)
        self.mock_read_client.get_ticket_by_id.return_value = {
            "success": True,
            "ticket": self.mock_ticket
        }

        self.mock_rag_tool = MagicMock(spec=RAGToolAdapter)
        self.mock_rag_tool.retrieve.return_value = {
            "success": True,
            "evidence": self.mock_evidence,
            "count": len(self.mock_evidence)
        }

        self.mock_write_client = MagicMock(spec=LarkBaseWriteClient)
        def mock_update(record_id, priority):
            self.mock_ticket["priority"] = priority
            return {
                "success": True,
                "record_id": record_id,
                "updated_priority": priority
            }
        self.mock_write_client.update_ticket_priority.side_effect = mock_update

        self.update_service = ControlledUpdateService(
            approval_service=self.approval_service,
            read_client=self.mock_read_client,
            write_client=self.mock_write_client
        )

        self.orchestrator = AgentOrchestrator(
            ticket_tool=self.mock_read_client,
            rag_tool=self.mock_rag_tool,
            approval_service=self.approval_service,
            update_service=self.update_service
        )

    def test_01_turn1_analyze_crm044_creates_pending_approval(self):
        """
        Turn 1: User asks to analyze CRM-044 and recommend priority change.
        System must generate recommendation, detect priority change, and create PENDING_APPROVAL request.
        Lark Base must NOT be modified.
        """
        session_id = "test_session_turn1"
        res = self.orchestrator.run(
            "Analyze CRM-044 and recommend whether its priority should be changed.",
            session_id=session_id
        )

        self.assertEqual(res["intent"], "PRIORITY_ANALYSIS")
        self.assertIsNotNone(res["priority_analysis"])
        self.assertTrue(res["priority_analysis"]["differs_from_current"])

        # Check Approval Request created
        self.assertIsNotNone(res["approval_id"])
        self.assertTrue(res["approval_id"].startswith("APR-"))
        self.assertEqual(res["approval_status"], "PENDING_APPROVAL")

        # Session mapping verified
        pending_apr = self.store.get_session_pending_approval(session_id)
        self.assertEqual(pending_apr, res["approval_id"])

        # Lark write must NOT have occurred
        self.mock_write_client.update_ticket_priority.assert_not_called()

        # Check response content
        response = res["final_response"]
        self.assertIn("### Human Approval Required", response)
        self.assertIn(res["approval_id"], response)
        self.assertIn("PENDING_APPROVAL", response)
        self.assertIn("Approve", response)

    def test_02_turn2_approve_recommendation_executes_controlled_update(self):
        """
        Turn 1 creates pending approval.
        Turn 2 approves recommendation, triggering ControlledUpdateService and Lark write.
        """
        session_id = "test_session_turn2"

        # Turn 1
        res1 = self.orchestrator.run(
            "Analyze CRM-044 and recommend whether its priority should be changed.",
            session_id=session_id
        )
        apr_id = res1["approval_id"]
        self.assertIsNotNone(apr_id)
        self.assertEqual(res1["approval_status"], "PENDING_APPROVAL")

        # Turn 2: Approve recommendation
        res2 = self.orchestrator.run(
            "Approve this recommendation for CRM-044.",
            session_id=session_id
        )

        self.assertEqual(res2["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertEqual(res2["approval_status"], "APPROVED")
        self.assertEqual(res2["update_status"], UpdateResultStatus.UPDATE_VERIFIED.value)

        # Verify Lark write occurred exactly once with target priority CRITICAL
        self.mock_write_client.update_ticket_priority.assert_called_once_with(
            record_id="rec_crm044",
            priority="CRITICAL"
        )

        # Response confirms successful priority update
        response = res2["final_response"]
        self.assertIn("Priority Update Successful", response)
        self.assertIn(apr_id, response)
        self.assertIn("SUCCESS", response)

    def test_03_turn2_reject_recommendation_leaves_lark_unchanged(self):
        """
        Turn 1 creates pending approval.
        Turn 2 rejects recommendation. No Lark write must occur.
        """
        session_id = "test_session_reject"

        res1 = self.orchestrator.run(
            "Analyze CRM-044 and recommend whether its priority should be changed.",
            session_id=session_id
        )
        apr_id = res1["approval_id"]
        self.assertIsNotNone(apr_id)

        # Turn 2: Reject recommendation
        res2 = self.orchestrator.run(
            "Reject this recommendation",
            session_id=session_id
        )

        self.assertEqual(res2["approval_status"], "REJECTED")
        self.assertIsNone(res2.get("update_status"))
        self.mock_write_client.update_ticket_priority.assert_not_called()

        response = res2["final_response"]
        self.assertIn("Recommendation Rejected", response)
        self.assertIn("Lark Daily_Task remains UNCHANGED", response)

    def test_04_turn2_cancel_recommendation_leaves_lark_unchanged(self):
        """
        Turn 1 creates pending approval.
        Turn 2 cancels recommendation. No Lark write must occur.
        """
        session_id = "test_session_cancel"

        res1 = self.orchestrator.run(
            "Analyze CRM-044 and recommend whether its priority should be changed.",
            session_id=session_id
        )
        apr_id = res1["approval_id"]
        self.assertIsNotNone(apr_id)

        # Turn 2: Cancel
        res2 = self.orchestrator.run(
            "Cancel",
            session_id=session_id
        )

        self.assertEqual(res2["approval_status"], "CANCELLED")
        self.assertIsNone(res2.get("update_status"))
        self.mock_write_client.update_ticket_priority.assert_not_called()

        response = res2["final_response"]
        self.assertIn("Recommendation Cancelled", response)
        self.assertIn("Lark Daily_Task remains UNCHANGED", response)

    def test_05_cross_session_isolation(self):
        """
        Session A creates pending approval.
        Session B attempts to approve without its own pending approval -> blocked.
        """
        res_a = self.orchestrator.run(
            "Analyze CRM-044 and recommend whether its priority should be changed.",
            session_id="session_A"
        )
        apr_a = res_a["approval_id"]
        self.assertIsNotNone(apr_a)

        # Session B attempts to approve
        res_b = self.orchestrator.run(
            "Approve this recommendation for CRM-044.",
            session_id="session_B"
        )

        self.assertIn("No pending approval found in current session", res_b.get("error", "") or res_b.get("approval_error", ""))
        self.mock_write_client.update_ticket_priority.assert_not_called()

    def test_06_duplicate_approval_protection(self):
        """
        Turn 2 approves request successfully.
        Turn 3 attempts to approve again -> blocked because already processed.
        """
        session_id = "session_duplicate"

        # Turn 1
        res1 = self.orchestrator.run(
            "Analyze CRM-044 and recommend whether its priority should be changed.",
            session_id=session_id
        )
        apr_id = res1["approval_id"]

        # Turn 2: First approve -> Success
        res2 = self.orchestrator.run(
            "Approve this recommendation for CRM-044.",
            session_id=session_id
        )
        self.assertEqual(res2["approval_status"], "APPROVED")
        self.mock_write_client.update_ticket_priority.assert_called_once()

        # Turn 3: Second conversational approve in same session -> Blocked (session pending was cleared upon successful update)
        res3 = self.orchestrator.run(
            "Approve this recommendation for CRM-044.",
            session_id=session_id
        )
        self.assertIn("No pending approval found in current session", res3.get("error", "") or res3.get("approval_error", ""))

        # Turn 4: Second apply execution for same approval ID -> ALREADY_APPLIED
        res4 = self.orchestrator.run(
            f"apply {apr_id}",
            session_id=session_id
        )
        self.assertEqual(res4["update_status"], UpdateResultStatus.ALREADY_APPLIED.value)
        self.assertIn("Already Applied", res4["final_response"])

        # Verify Lark write was called ONLY once throughout
        self.mock_write_client.update_ticket_priority.assert_called_once()

    def test_07_stale_approval_protection(self):
        """
        If ticket priority changes externally between recommendation and update execution,
        ControlledUpdateService detects UPDATE_STALE and rejects update without modifying Lark.
        """
        session_id = "session_stale"

        # Turn 1: Analyze CRM-044 (Current HIGH, Recommended CRITICAL)
        res1 = self.orchestrator.run(
            "Analyze CRM-044 and recommend whether its priority should be changed.",
            session_id=session_id
        )
        apr_id = res1["approval_id"]

        # Simulate external priority change in Lark: CRM-044 priority changed from HIGH to MEDIUM
        stale_ticket = dict(self.mock_ticket)
        stale_ticket["priority"] = "MEDIUM"
        self.mock_read_client.get_ticket_by_id.return_value = {
            "success": True,
            "ticket": stale_ticket
        }

        # Turn 2: Attempt approval & update
        res2 = self.orchestrator.run(
            "Approve this recommendation for CRM-044.",
            session_id=session_id
        )

        self.assertEqual(res2["update_status"], UpdateResultStatus.UPDATE_STALE.value)
        # Write client must NOT have executed update
        self.mock_write_client.update_ticket_priority.assert_not_called()
        self.assertIn("Priority Update Blocked", res2["final_response"])
        self.assertIn("stale", res2["final_response"].lower())


if __name__ == "__main__":
    unittest.main()
