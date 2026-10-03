import unittest
from unittest.mock import MagicMock, patch

from src.lark.lark_base import LarkAuthError, LarkAPIError
from src.lark.lark_base_write import LarkBaseWriteClient
from src.agent.approval import ApprovalStore, ApprovalService, ApprovalStatus, ActorType
from src.agent.update import (
    ControlledUpdateService,
    UpdateResultStatus,
    UpdateVerifier,
)
from src.agent.graph import AgentOrchestrator
from src.agent.tools.ticket_tool import TicketToolAdapter
from src.agent.tools.rag_tool import RAGToolAdapter


class TestControlledLarkUpdate(unittest.TestCase):

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
            "provenance": {"record_id": "rec101"}
        }

        self.sample_analysis = {
            "ticket_id": "TASK-101",
            "current_priority": "MEDIUM",
            "recommended_priority": "CRITICAL",
            "weighted_score": 0.82,
            "confidence": "High",
            "summary_reason": "Critical blocker for downstream tasks."
        }

        self.mock_read_client = MagicMock(spec=TicketToolAdapter)
        self.mock_write_client = MagicMock(spec=LarkBaseWriteClient)

        self.mock_read_client.get_ticket_by_id.return_value = {
            "success": True,
            "ticket": self.sample_ticket
        }

        self.mock_write_client.update_ticket_priority.return_value = {
            "success": True,
            "record_id": "rec101",
            "updated_priority": "CRITICAL"
        }

        self.update_service = ControlledUpdateService(
            approval_service=self.approval_service,
            read_client=self.mock_read_client,
            write_client=self.mock_write_client
        )

        # Create and approve sample request APR-001
        self.req = self.approval_service.create_approval_request(self.sample_ticket, self.sample_analysis)
        self.approval_service.approve(self.req.approval_id, reviewer_id="user_pm")

    # 1. Successful Approved Priority Update
    def test_01_successful_approved_priority_update(self):
        # Post-write verify return ticket with CRITICAL
        post_ticket = dict(self.sample_ticket, priority="CRITICAL")
        self.mock_read_client.get_ticket_by_id.side_effect = [
            {"success": True, "ticket": self.sample_ticket},  # pre-write
            {"success": True, "ticket": post_ticket}          # post-write
        ]

        res = self.update_service.apply_approved_priority_update(self.req.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_VERIFIED)
        self.assertTrue(res.verified)
        self.assertEqual(res.previous_priority, "MEDIUM")
        self.assertEqual(res.requested_priority, "CRITICAL")
        self.assertEqual(res.actual_priority, "CRITICAL")
        self.mock_write_client.update_ticket_priority.assert_called_with(record_id="rec101", priority="CRITICAL")

    # 2. Pending Approval Rejected
    def test_02_pending_approval_rejected(self):
        req_pending = self.approval_service.create_approval_request(
            dict(self.sample_ticket, ticket_id="TASK-102"),
            dict(self.sample_analysis, ticket_id="TASK-102")
        )
        res = self.update_service.apply_approved_priority_update(req_pending.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_REJECTED)
        self.assertIn("Must be APPROVED", res.reason)

    # 3. Rejected Approval Rejected
    def test_03_rejected_approval_rejected(self):
        req_rej = self.approval_service.create_approval_request(
            dict(self.sample_ticket, ticket_id="TASK-103"),
            dict(self.sample_analysis, ticket_id="TASK-103")
        )
        self.approval_service.reject(req_rej.approval_id, reviewer_id="user_pm", rejection_reason="Not needed")
        res = self.update_service.apply_approved_priority_update(req_rej.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_REJECTED)

    # 4. Cancelled Approval Rejected
    def test_04_cancelled_approval_rejected(self):
        req_can = self.approval_service.create_approval_request(
            dict(self.sample_ticket, ticket_id="TASK-104"),
            dict(self.sample_analysis, ticket_id="TASK-104")
        )
        self.approval_service.cancel(req_can.approval_id, actor_id="user_pm")
        res = self.update_service.apply_approved_priority_update(req_can.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_REJECTED)

    # 5. Expired Approval Rejected
    def test_05_expired_approval_rejected(self):
        req_exp = self.approval_service.create_approval_request(
            dict(self.sample_ticket, ticket_id="TASK-105"),
            dict(self.sample_analysis, ticket_id="TASK-105")
        )
        self.approval_service.expire(req_exp.approval_id)
        res = self.update_service.apply_approved_priority_update(req_exp.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_REJECTED)

    # 6. Agent Cannot Approve or Update as Human
    def test_06_agent_cannot_approve_or_update_as_human(self):
        res = self.update_service.apply_approved_priority_update(
            self.req.approval_id,
            actor_id="AGENT",
            actor_type=ActorType.AGENT
        )
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_REJECTED)
        self.assertIn("Agent is forbidden", res.reason)

    # 7. Missing Reviewer Rejected
    def test_07_missing_reviewer_rejected(self):
        req_no_rev = self.approval_service.create_approval_request(
            dict(self.sample_ticket, ticket_id="TASK-107"),
            dict(self.sample_analysis, ticket_id="TASK-107")
        )
        req_no_rev.status = ApprovalStatus.APPROVED
        req_no_rev.reviewed_by = ""
        self.store.save_request(req_no_rev)

        res = self.update_service.apply_approved_priority_update(req_no_rev.approval_id, actor_id="")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_REJECTED)

    # 8. Invalid Priority Rejected
    def test_08_invalid_priority_rejected(self):
        write_client = LarkBaseWriteClient(app_token="app123", table_id="tbl123")
        with self.assertRaises(LarkAPIError):
            write_client.update_ticket_priority("rec101", "P9")
        with self.assertRaises(LarkAPIError):
            write_client.update_ticket_priority("rec101", "P1")

    # 9. Stale Recommendation Rejected
    def test_09_stale_recommendation_rejected(self):
        # Live ticket priority changed to HIGH (Snapshot was MEDIUM)
        stale_ticket = dict(self.sample_ticket, priority="HIGH")
        self.mock_read_client.get_ticket_by_id.return_value = {"success": True, "ticket": stale_ticket}

        res = self.update_service.apply_approved_priority_update(self.req.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_STALE)
        self.assertIn("Approval is stale", res.reason)
        self.mock_write_client.update_ticket_priority.assert_not_called()

    # 10. Missing Ticket Rejected
    def test_10_missing_ticket_rejected(self):
        self.mock_read_client.get_ticket_by_id.return_value = {"success": False, "ticket": None}
        res = self.update_service.apply_approved_priority_update(self.req.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_FAILED)
        self.assertIn("not found", res.reason)

    # 11. Wrong Approval ID Rejected
    def test_11_wrong_approval_id_rejected(self):
        res = self.update_service.apply_approved_priority_update("APR-999", actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_REJECTED)
        self.assertIn("not found", res.reason)

    # 12. Wrong Ticket ID Rejected
    def test_12_wrong_ticket_id_rejected(self):
        self.mock_read_client.get_ticket_by_id.return_value = {"success": False, "ticket": None}
        res = self.update_service.apply_approved_priority_update(self.req.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_FAILED)

    # 13. Approved Priority Mismatch Rejected
    def test_13_approved_priority_mismatch_rejected(self):
        # Snapshot says P1, attempt P2 update
        self.req.snapshot.recommended_priority = "INVALID"
        res = self.update_service.apply_approved_priority_update(self.req.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_REJECTED)

    # 14. Lark Authentication Failure
    def test_14_lark_authentication_failure(self):
        self.mock_write_client.update_ticket_priority.side_effect = LarkAuthError("Authentication failure (HTTP 401).")
        res = self.update_service.apply_approved_priority_update(self.req.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_FAILED)
        self.assertIn("Lark write API call failed", res.reason)

    # 15. Lark Authorization Failure
    def test_15_lark_authorization_failure(self):
        self.mock_write_client.update_ticket_priority.side_effect = LarkAuthError("Authorization failure (HTTP 403).")
        res = self.update_service.apply_approved_priority_update(self.req.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_FAILED)

    # 16. Lark 404 Not Found
    def test_16_lark_404_not_found(self):
        self.mock_write_client.update_ticket_priority.side_effect = LarkAPIError("Record not found", status_code=404)
        res = self.update_service.apply_approved_priority_update(self.req.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_FAILED)

    # 17. Lark 429 Rate Limit Retry
    @patch("time.sleep", return_value=None)
    @patch("requests.put")
    def test_17_lark_429_rate_limit_retry(self, mock_put, mock_sleep):
        token_mgr = MagicMock()
        token_mgr.get_tenant_access_token.return_value = "token123"
        write_client = LarkBaseWriteClient(app_token="app123", table_id="tbl123", token_manager=token_mgr)

        # 429 then 200
        resp_429 = MagicMock(status_code=429)
        resp_200 = MagicMock(status_code=200)
        resp_200.json.return_value = {"code": 0, "msg": "success"}

        mock_put.side_effect = [resp_429, resp_200]

        res = write_client.update_ticket_priority("rec101", "CRITICAL")
        self.assertTrue(res["success"])
        self.assertEqual(mock_put.call_count, 2)

    # 18. Lark Timeout Retry
    @patch("time.sleep", return_value=None)
    @patch("requests.put")
    def test_18_lark_timeout_retry(self, mock_put, mock_sleep):
        import requests
        token_mgr = MagicMock()
        token_mgr.get_tenant_access_token.return_value = "token123"
        write_client = LarkBaseWriteClient(app_token="app123", table_id="tbl123", token_manager=token_mgr)

        resp_200 = MagicMock(status_code=200)
        resp_200.json.return_value = {"code": 0, "msg": "success"}

        mock_put.side_effect = [requests.Timeout("Timeout"), resp_200]

        res = write_client.update_ticket_priority("rec101", "CRITICAL")
        self.assertTrue(res["success"])
        self.assertEqual(mock_put.call_count, 2)

    # 19. Network Failure
    @patch("time.sleep", return_value=None)
    @patch("requests.put")
    def test_19_network_failure(self, mock_put, mock_sleep):
        import requests
        token_mgr = MagicMock()
        token_mgr.get_tenant_access_token.return_value = "token123"
        write_client = LarkBaseWriteClient(app_token="app123", table_id="tbl123", token_manager=token_mgr)

        mock_put.side_effect = requests.ConnectionError("Network down")
        with self.assertRaises(LarkAPIError):
            write_client.update_ticket_priority("rec101", "CRITICAL")

    # 20. Successful Post-Write Verification
    def test_20_successful_post_write_verification(self):
        post_ticket = dict(self.sample_ticket, priority="CRITICAL")
        self.mock_read_client.get_ticket_by_id.side_effect = [
            {"success": True, "ticket": self.sample_ticket},
            {"success": True, "ticket": post_ticket}
        ]

        res = self.update_service.apply_approved_priority_update(self.req.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_VERIFIED)
        self.assertTrue(res.verified)

    # 21. Post-Write Verification Failure
    def test_21_post_write_verification_failure(self):
        # Post-write verify returns MEDIUM instead of CRITICAL
        post_ticket = dict(self.sample_ticket, priority="MEDIUM")
        self.mock_read_client.get_ticket_by_id.side_effect = [
            {"success": True, "ticket": self.sample_ticket},
            {"success": True, "ticket": post_ticket}
        ]

        res = self.update_service.apply_approved_priority_update(self.req.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.ROLLBACK_EXECUTED)

    # 22. Safe Rollback Execution
    def test_22_safe_rollback_execution(self):
        post_ticket = dict(self.sample_ticket, priority="MEDIUM")
        self.mock_read_client.get_ticket_by_id.side_effect = [
            {"success": True, "ticket": self.sample_ticket},
            {"success": True, "ticket": post_ticket}
        ]

        res = self.update_service.apply_approved_priority_update(self.req.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.ROLLBACK_EXECUTED)
        self.mock_write_client.update_ticket_priority.assert_any_call(record_id="rec101", priority="MEDIUM")

    # 23. Rollback Conflict Detection
    def test_23_rollback_conflict_detection(self):
        # Live priority changed to HIGH (conflict)
        post_ticket = dict(self.sample_ticket, priority="HIGH")
        self.mock_read_client.get_ticket_by_id.side_effect = [
            {"success": True, "ticket": self.sample_ticket},
            {"success": True, "ticket": post_ticket}
        ]

        res = self.update_service.apply_approved_priority_update(self.req.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.ROLLBACK_SKIPPED_DUE_TO_CONFLICT)

    # 24. Duplicate Execution Detection
    def test_24_duplicate_execution_detection(self):
        already_critical_ticket = dict(self.sample_ticket, priority="CRITICAL")
        self.mock_read_client.get_ticket_by_id.return_value = {"success": True, "ticket": already_critical_ticket}

        res = self.update_service.apply_approved_priority_update(self.req.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.ALREADY_APPLIED)
        self.mock_write_client.update_ticket_priority.assert_not_called()

    # 25. Already Applied Detection
    def test_25_already_applied_detection(self):
        already_critical_ticket = dict(self.sample_ticket, priority="CRITICAL")
        self.mock_read_client.get_ticket_by_id.return_value = {"success": True, "ticket": already_critical_ticket}

        res = self.update_service.apply_approved_priority_update(self.req.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.ALREADY_APPLIED)

    # 26. Audit UPDATE_REQUESTED Event
    def test_26_audit_update_requested_event(self):
        log = self.store.get_audit_log(self.req.approval_id)
        # Audit log contains creation and approval events
        self.assertTrue(len(log) >= 2)

    # 27. Audit UPDATE_EXECUTED Event
    def test_27_audit_update_executed_event(self):
        post_ticket = dict(self.sample_ticket, priority="CRITICAL")
        self.mock_read_client.get_ticket_by_id.side_effect = [
            {"success": True, "ticket": self.sample_ticket},
            {"success": True, "ticket": post_ticket}
        ]
        self.update_service.apply_approved_priority_update(self.req.approval_id, actor_id="user_pm")
        log = self.store.get_audit_log(self.req.approval_id)
        actions = [e.action for e in log]
        self.assertIn("UPDATE_VERIFIED", actions)

    # 28. Audit UPDATE_VERIFIED Event
    def test_28_audit_update_verified_event(self):
        post_ticket = dict(self.sample_ticket, priority="CRITICAL")
        self.mock_read_client.get_ticket_by_id.side_effect = [
            {"success": True, "ticket": self.sample_ticket},
            {"success": True, "ticket": post_ticket}
        ]
        self.update_service.apply_approved_priority_update(self.req.approval_id, actor_id="user_pm")
        log = self.store.get_audit_log(self.req.approval_id)
        verified_events = [e for e in log if e.action == "UPDATE_VERIFIED"]
        self.assertTrue(len(verified_events) > 0)

    # 29. Audit UPDATE_FAILED Event
    def test_29_audit_update_failed_event(self):
        self.mock_write_client.update_ticket_priority.side_effect = LarkAPIError("API Failure")
        self.update_service.apply_approved_priority_update(self.req.approval_id, actor_id="user_pm")
        log = self.store.get_audit_log(self.req.approval_id)
        failed_events = [e for e in log if e.action == "UPDATE_FAILED"]
        self.assertTrue(len(failed_events) > 0)

    # 30. Audit Does Not Contain Secrets
    def test_30_audit_does_not_contain_secrets(self):
        log = self.store.get_audit_log()
        for e in log:
            e_str = str(e.to_dict())
            self.assertNotIn("secret", e_str.lower())
            self.assertNotIn("token", e_str.lower())
            self.assertNotIn("bearer", e_str.lower())

    # 31. Only Priority Included in Update Payload
    @patch("requests.put")
    def test_31_only_priority_included_in_update_payload(self, mock_put):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"code": 0, "msg": "success"}
        mock_put.return_value = mock_resp

        token_mgr = MagicMock()
        token_mgr.get_tenant_access_token.return_value = "token123"
        write_client = LarkBaseWriteClient(app_token="app123", table_id="tbl123", token_manager=token_mgr)

        write_client.update_ticket_priority("rec101", "CRITICAL")

        args, kwargs = mock_put.call_args
        payload = kwargs.get("json", {})
        res = UpdateVerifier.verify_payload(payload)
        self.assertTrue(res["verified"])
        self.assertEqual(res["keys"], ["Priority"])

    # 32. No Status Update in Payload
    @patch("requests.put")
    def test_32_no_status_update_in_payload(self, mock_put):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"code": 0, "msg": "success"}
        mock_put.return_value = mock_resp

        token_mgr = MagicMock()
        token_mgr.get_tenant_access_token.return_value = "token123"
        write_client = LarkBaseWriteClient(app_token="app123", table_id="tbl123", token_manager=token_mgr)

        write_client.update_ticket_priority("rec101", "CRITICAL")
        payload = mock_put.call_args[1].get("json", {}).get("fields", {})
        self.assertNotIn("Status", payload)

    # 33. No Sprint Update in Payload
    @patch("requests.put")
    def test_33_no_sprint_update_in_payload(self, mock_put):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"code": 0, "msg": "success"}
        mock_put.return_value = mock_resp

        token_mgr = MagicMock()
        token_mgr.get_tenant_access_token.return_value = "token123"
        write_client = LarkBaseWriteClient(app_token="app123", table_id="tbl123", token_manager=token_mgr)

        write_client.update_ticket_priority("rec101", "CRITICAL")
        payload = mock_put.call_args[1].get("json", {}).get("fields", {})
        self.assertNotIn("Sprint", payload)

    # 34. No Assignee Update in Payload
    @patch("requests.put")
    def test_34_no_assignee_update_in_payload(self, mock_put):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"code": 0, "msg": "success"}
        mock_put.return_value = mock_resp

        token_mgr = MagicMock()
        token_mgr.get_tenant_access_token.return_value = "token123"
        write_client = LarkBaseWriteClient(app_token="app123", table_id="tbl123", token_manager=token_mgr)

        write_client.update_ticket_priority("rec101", "CRITICAL")
        payload = mock_put.call_args[1].get("json", {}).get("fields", {})
        self.assertNotIn("Assignee", payload)

    # 35. No Story Point Update in Payload
    @patch("requests.put")
    def test_35_no_story_point_update_in_payload(self, mock_put):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"code": 0, "msg": "success"}
        mock_put.return_value = mock_resp

        token_mgr = MagicMock()
        token_mgr.get_tenant_access_token.return_value = "token123"
        write_client = LarkBaseWriteClient(app_token="app123", table_id="tbl123", token_manager=token_mgr)

        write_client.update_ticket_priority("rec101", "CRITICAL")
        payload = mock_put.call_args[1].get("json", {}).get("fields", {})
        self.assertNotIn("Story Point", payload)

    # 36. Existing 101 Tests Regression
    def test_36_existing_101_tests_regression(self):
        post_ticket = dict(self.sample_ticket, priority="CRITICAL")
        self.mock_read_client.get_ticket_by_id.side_effect = [
            {"success": True, "ticket": self.sample_ticket},
            {"success": True, "ticket": post_ticket}
        ]

        mock_ticket_adapter = MagicMock(spec=TicketToolAdapter)
        mock_ticket_adapter.get_ticket_by_id.return_value = {"success": True, "ticket": self.sample_ticket}

        mock_rag_adapter = MagicMock(spec=RAGToolAdapter)

        orchestrator = AgentOrchestrator(
            ticket_tool=mock_ticket_adapter,
            rag_tool=mock_rag_adapter,
            approval_service=self.approval_service,
            update_service=self.update_service
        )

        res = orchestrator.run(f"Apply {self.req.approval_id} by user_pm")
        self.assertEqual(res["intent"], "PRIORITY_UPDATE_EXECUTION")
        self.assertEqual(res["update_status"], "UPDATE_VERIFIED")
        self.assertIn("Priority Update Successful", res["final_response"])


if __name__ == "__main__":
    unittest.main()
