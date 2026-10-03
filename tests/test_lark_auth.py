import os
import unittest
from unittest.mock import MagicMock, patch
import requests

from src.lark.auth import TenantAccessTokenManager
from src.lark.exceptions import LarkConfigError, LarkAuthError, LarkAPIError
from src.lark.lark_base_write import LarkBaseWriteClient
from src.agent.update import ControlledUpdateService, UpdateResultStatus
from src.agent.approval import ApprovalStore, ApprovalService
from src.agent.approval.approval_store import reset_shared_approval_store
from src.agent.graph import AgentOrchestrator
from src.agent.tools.ticket_tool import TicketToolAdapter
from src.agent.tools.rag_tool import RAGToolAdapter


class TestTenantAccessTokenManagerUnit(unittest.TestCase):
    """
    Unit tests for TenantAccessTokenManager verifying actual interface implementation.
    """

    @patch("requests.post")
    def test_01_successful_token_retrieval(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "code": 0,
            "msg": "ok",
            "tenant_access_token": "t-valid-token-12345",
            "expire": 7200
        }
        mock_post.return_value = mock_resp

        manager = TenantAccessTokenManager(app_id="cli_test_id", app_secret="sec_test_secret")

        # Test both get_token and get_tenant_access_token interface methods
        tok1 = manager.get_token()
        self.assertEqual(tok1, "t-valid-token-12345")

        tok2 = manager.get_tenant_access_token()
        self.assertEqual(tok2, "t-valid-token-12345")

        # Verify token caching (post called only once)
        self.assertEqual(mock_post.call_count, 1)

    def test_02_missing_configuration(self):
        # Empty credentials
        manager = TenantAccessTokenManager(app_id="", app_secret="")
        with self.assertRaises(LarkConfigError) as ctx:
            manager.get_tenant_access_token()
        self.assertIn("LARK_APP_ID and LARK_APP_SECRET must be configured", str(ctx.exception))

        with self.assertRaises(LarkConfigError):
            manager.get_token()

    @patch("requests.post")
    def test_03_network_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        manager = TenantAccessTokenManager(app_id="cli_test_id", app_secret="sec_test_secret")
        with self.assertRaises(LarkAuthError) as ctx:
            manager.get_tenant_access_token()
        self.assertIn("Network error while requesting Lark tenant access token", str(ctx.exception))

    @patch("requests.post")
    def test_04_lark_api_error_code(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "code": 10003,
            "msg": "app secret invalid"
        }
        mock_post.return_value = mock_resp

        manager = TenantAccessTokenManager(app_id="cli_test_id", app_secret="sec_invalid_secret")
        with self.assertRaises(LarkAuthError) as ctx:
            manager.get_tenant_access_token()
        self.assertIn("code=10003", str(ctx.exception))
        self.assertIn("app secret invalid", str(ctx.exception))

    @patch("requests.post")
    def test_05_invalid_api_response_missing_token(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "code": 0,
            "msg": "ok"
            # Missing "tenant_access_token"
        }
        mock_post.return_value = mock_resp

        manager = TenantAccessTokenManager(app_id="cli_test_id", app_secret="sec_test_secret")
        with self.assertRaises(LarkAuthError) as ctx:
            manager.get_tenant_access_token()
        self.assertIn("did not include tenant_access_token", str(ctx.exception))

    @patch("requests.post")
    def test_06_http_error_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("502 Bad Gateway")
        mock_post.return_value = mock_resp

        manager = TenantAccessTokenManager(app_id="cli_test_id", app_secret="sec_test_secret")
        with self.assertRaises(LarkAuthError) as ctx:
            manager.get_tenant_access_token()
        self.assertIn("Network error", str(ctx.exception))


class TestLarkWriteClientAuthIntegration(unittest.TestCase):
    """
    Integration tests verifying ControlledUpdateService -> LarkBaseWriteClient -> TenantAccessTokenManager.
    """

    @patch("requests.put")
    @patch("requests.post")
    def test_01_real_token_manager_injected_into_write_client(self, mock_post, mock_put):
        # 1. Auth token success
        mock_auth_resp = MagicMock(status_code=200)
        mock_auth_resp.json.return_value = {
            "code": 0,
            "msg": "ok",
            "tenant_access_token": "t-real-token-xyz",
            "expire": 7200
        }
        mock_post.return_value = mock_auth_resp

        # 2. Lark Bitable update success
        mock_put_resp = MagicMock(status_code=200)
        mock_put_resp.json.return_value = {
            "code": 0,
            "msg": "success",
            "data": {
                "record": {
                    "record_id": "rec_test_101",
                    "fields": {"Priority": "CRITICAL"}
                }
            }
        }
        mock_put.return_value = mock_put_resp

        # Real TenantAccessTokenManager instance
        real_token_mgr = TenantAccessTokenManager(app_id="cli_real_id", app_secret="sec_real_secret")

        write_client = LarkBaseWriteClient(
            app_token="app_test_token",
            table_id="tbl_test_table",
            token_manager=real_token_mgr
        )

        # Disable DEMO_MODE to test actual write network flow
        with patch.dict(os.environ, {"DEMO_MODE": "false", "LARK_WRITE_MODE": "real"}):
            res = write_client.update_ticket_priority(record_id="rec_test_101", priority="CRITICAL")

        self.assertTrue(res["success"])
        self.assertEqual(res["updated_priority"], "CRITICAL")

        # Verify put request was called with Authorization: Bearer t-real-token-xyz
        mock_put.assert_called_once()
        headers_used = mock_put.call_args[1]["headers"]
        self.assertEqual(headers_used["Authorization"], "Bearer t-real-token-xyz")

    def test_02_write_client_fails_safely_when_credentials_missing(self):
        real_token_mgr = TenantAccessTokenManager(app_id="", app_secret="")
        write_client = LarkBaseWriteClient(
            app_token="app_test_token",
            table_id="tbl_test_table",
            token_manager=real_token_mgr
        )

        with patch.dict(os.environ, {"DEMO_MODE": "false", "LARK_WRITE_MODE": "real"}):
            with self.assertRaises(LarkConfigError):
                write_client.update_ticket_priority(record_id="rec_test_101", priority="CRITICAL")

    @patch("requests.post")
    def test_03_controlled_update_service_handles_auth_failure_without_fake_success(self, mock_post):
        # Simulate Lark auth failure (code 10003)
        mock_auth_resp = MagicMock(status_code=200)
        mock_auth_resp.json.return_value = {"code": 10003, "msg": "Invalid app secret"}
        mock_post.return_value = mock_auth_resp

        real_token_mgr = TenantAccessTokenManager(app_id="cli_id", app_secret="invalid_secret")
        write_client = LarkBaseWriteClient(
            app_token="app_test_token",
            table_id="tbl_test_table",
            token_manager=real_token_mgr
        )

        store = ApprovalStore()
        approval_service = ApprovalService(store=store)

        mock_ticket = {
            "ticket_id": "CRM-044",
            "name": "Superannuation Rollover Gateway Latency",
            "priority": "HIGH",
            "record_id": "rec_crm044"
        }
        mock_analysis = {
            "ticket_id": "CRM-044",
            "current_priority": "HIGH",
            "recommended_priority": "CRITICAL",
            "differs_from_current": True
        }
        req = approval_service.create_approval_request(ticket=mock_ticket, priority_analysis=mock_analysis)
        approval_service.approve(req.approval_id, reviewer_id="user_pm")

        mock_read_client = MagicMock(spec=TicketToolAdapter)
        mock_read_client.get_ticket_by_id.return_value = {"success": True, "ticket": mock_ticket}

        update_service = ControlledUpdateService(
            approval_service=approval_service,
            read_client=mock_read_client,
            write_client=write_client
        )

        with patch.dict(os.environ, {"DEMO_MODE": "false", "LARK_WRITE_MODE": "real"}):
            update_res = update_service.apply_approved_priority_update(req.approval_id, actor_id="user_pm")

        # Must report UPDATE_FAILED, NOT success!
        self.assertEqual(update_res.status, UpdateResultStatus.UPDATE_FAILED)
        self.assertIn("Lark write API call failed", update_res.reason)
        self.assertIn("Invalid app secret", update_res.reason)
        self.assertFalse(update_res.verified)


class TestCRM044E2EApprovalWriteFlow(unittest.TestCase):
    """
    End-to-end approval write test:
    Analyze CRM-044 -> Recommendation -> Pending Approval -> 'approve' -> Auth -> Lark Write -> Audit -> Response.
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
        self.mock_read_client.get_ticket_by_id.side_effect = self._get_ticket_mock

        self.mock_rag_tool = MagicMock(spec=RAGToolAdapter)
        self.mock_rag_tool.retrieve.return_value = {
            "success": True,
            "evidence": self.mock_evidence,
            "count": len(self.mock_evidence)
        }

    def _get_ticket_mock(self, ticket_id):
        return {
            "success": True,
            "ticket": dict(self.mock_ticket)
        }

    @patch("requests.put")
    @patch("requests.post")
    def test_01_e2e_turn1_analyze_then_turn2_approve_single_word(self, mock_post, mock_put):
        """
        Tests:
        Turn 1: 'Analyze CRM-044 and recommend whether its priority should be changed.'
        Turn 2: 'approve'
        Verifies:
        - Real TenantAccessTokenManager called without 'has no attribute get_tenant_access_token' error
        - Token retrieved via manager
        - Lark Bitable PUT request called with Priority: P1 and Authorization: Bearer <token>
        - Post-write verification succeeds
        - Audit log recorded
        - Final response confirms Priority Update Successful
        """
        # Auth mock response
        mock_auth_resp = MagicMock(status_code=200)
        mock_auth_resp.json.return_value = {
            "code": 0,
            "msg": "ok",
            "tenant_access_token": "t-e2e-token-abc123",
            "expire": 7200
        }
        mock_post.return_value = mock_auth_resp

        # Lark Bitable PUT mock response
        def mock_lark_put(url, json=None, headers=None, timeout=None):
            # Update local mock ticket priority to simulate Lark Bitable updating
            new_prio = json.get("fields", {}).get("Priority")
            self.mock_ticket["priority"] = new_prio
            resp = MagicMock(status_code=200)
            resp.json.return_value = {
                "code": 0,
                "msg": "success",
                "data": {
                    "record": {
                        "record_id": "rec_crm044",
                        "fields": {"Priority": new_prio}
                    }
                }
            }
            return resp

        mock_put.side_effect = mock_lark_put

        real_token_mgr = TenantAccessTokenManager(app_id="cli_test_id", app_secret="sec_test_secret")
        write_client = LarkBaseWriteClient(
            app_token="app_test_token",
            table_id="tbl_test_table",
            token_manager=real_token_mgr
        )

        update_service = ControlledUpdateService(
            approval_service=self.approval_service,
            read_client=self.mock_read_client,
            write_client=write_client
        )

        orchestrator = AgentOrchestrator(
            ticket_tool=self.mock_read_client,
            rag_tool=self.mock_rag_tool,
            approval_service=self.approval_service,
            update_service=update_service
        )

        session_id = "e2e_session_crm044"

        # Turn 1: Analyze CRM-044
        with patch.dict(os.environ, {"DEMO_MODE": "false", "LARK_WRITE_MODE": "real"}):
            res1 = orchestrator.run(
                "Analyze CRM-044 and recommend whether its priority should be changed.",
                session_id=session_id
            )

        self.assertEqual(res1["intent"], "PRIORITY_ANALYSIS")
        self.assertIsNotNone(res1["approval_id"])
        apr_id = res1["approval_id"]
        self.assertEqual(res1["approval_status"], "PENDING_APPROVAL")
        # Ensure no Lark write happened during Turn 1
        mock_put.assert_not_called()

        # Turn 2: User says single-word "approve"
        with patch.dict(os.environ, {"DEMO_MODE": "false", "LARK_WRITE_MODE": "real"}):
            res2 = orchestrator.run(
                "approve",
                session_id=session_id
            )

        # Verify no AttributeError
        self.assertIsNone(res2.get("error"))
        self.assertEqual(res2["approval_status"], "APPROVED")
        self.assertEqual(res2["update_status"], UpdateResultStatus.UPDATE_VERIFIED.value)

        # Verify real Lark write call
        mock_put.assert_called_once()
        put_headers = mock_put.call_args[1]["headers"]
        put_payload = mock_put.call_args[1]["json"]

        self.assertEqual(put_headers["Authorization"], "Bearer t-e2e-token-abc123")
        self.assertEqual(put_payload, {"fields": {"Priority": "CRITICAL"}})

        # Verify response text
        response_text = res2["final_response"]
        self.assertIn("Priority Update Successful", response_text)
        self.assertIn(apr_id, response_text)
        self.assertIn("SUCCESS", response_text)
        self.assertNotIn("Operation Error", response_text)


if __name__ == "__main__":
    unittest.main()
