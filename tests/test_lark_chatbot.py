import os
import json
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.api import app, _processed_events
from src.lark.lark_base_write import LarkBaseWriteClient
from src.agent.approval import ApprovalStore, ApprovalService
from src.agent.update import ControlledUpdateService, UpdateResultStatus
from src.agent.tools.ticket_tool import TicketToolAdapter
from src.agent.tools.rag_tool import RAGToolAdapter
from src.agent.graph import AgentOrchestrator
from src.agent.state import Evidence


class TestLarkChatbotWebhook(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        _processed_events.clear()

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

        import src.api.api as api_mod
        api_mod._pipeline_ready = True
        mock_pipe = MagicMock()
        mock_pipe.run.side_effect = lambda q, *args, **kwargs: {
            "answer": self.orchestrator.run(q)["final_response"],
            "retrieved_from": "agent_orchestrator",
            "sources": []
        }
        api_mod._pipeline = mock_pipe

        mock_clarifier = MagicMock()
        mock_clarifier.check_ambiguity.return_value = {"is_ambiguous": False, "options": []}
        api_mod._clarifier = mock_clarifier

    # 1. URL Challenge Event
    def test_01_url_verification_challenge(self):
        payload = {
            "type": "url_verification",
            "challenge": "challenge_token_123456"
        }
        resp = self.client.post("/api/lark/webhook", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"challenge": "challenge_token_123456"})

    # 2. Valid Message Event Format
    @patch("src.api.api._send_lark_reply", return_value=True)
    def test_02_valid_message_event(self, mock_reply):
        payload = {
            "header": {
                "event_id": "evt_001",
                "event_type": "im.message.receive_v1"
            },
            "event": {
                "message": {
                    "message_id": "msg_001",
                    "chat_id": "oc_chat123",
                    "content": json.dumps({"text": "What is the status of TASK-101?"})
                },
                "sender": {"sender_id": {"user_id": "usr_pm"}}
            }
        }
        resp = self.client.post("/api/lark/webhook", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("TASK-101", data["reply"])

    # 3. Duplicate Event Protection
    @patch("src.api.api._send_lark_reply", return_value=True)
    def test_03_duplicate_event_protection(self, mock_reply):
        payload = {
            "header": {
                "event_id": "evt_dup_001",
                "event_type": "im.message.receive_v1"
            },
            "event": {
                "message": {
                    "message_id": "msg_dup_001",
                    "chat_id": "oc_chat123",
                    "content": json.dumps({"text": "What is SmartLogi?"})
                }
            }
        }
        resp1 = self.client.post("/api/lark/webhook", json=payload)
        self.assertEqual(resp1.status_code, 200)

        resp2 = self.client.post("/api/lark/webhook", json=payload)
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertEqual(data2["message"], "duplicate_ignored")
        self.assertFalse(data2["reply_sent"])

    # 4. Empty or Malformed Message
    def test_04_empty_or_malformed_message(self):
        payload = {
            "header": {"event_id": "evt_empty_001"},
            "event": {
                "message": {
                    "message_id": "msg_empty",
                    "content": ""
                }
            }
        }
        resp = self.client.post("/api/lark/webhook", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ignored")

    # 5. FAQ Message Flow
    def test_05_faq_message_flow(self):
        res = self.orchestrator.run("What is SmartLogi?")
        self.assertEqual(res["intent"], "FAQ")
        self.assertTrue(len(res["final_response"]) > 0)

    # 6. Ticket Query Flow
    def test_06_ticket_query_flow(self):
        res = self.orchestrator.run("What is the status of TASK-101?")
        self.assertEqual(res["intent"], "TICKET_QUERY")
        self.assertIn("TASK-101", res["final_response"])

    def test_06a_crm_ticket_query_flow(self):
        crm_ticket = dict(self.sample_ticket, ticket_id="CRM-050", name="CRM Customer Dashboard")
        self.mock_ticket_tool.get_ticket_by_id.return_value = {"success": True, "found": True, "ticket": crm_ticket}
        res = self.orchestrator.run("tell me the content of ticket CRM-050")
        self.assertEqual(res["intent"], "TICKET_QUERY")
        self.assertIn("CRM-050", res["final_response"])
        self.mock_ticket_tool.get_ticket_by_id.assert_called_with("CRM-050")

    # 7. Ticket Analysis Flow
    def test_07_ticket_analysis_flow(self):
        res = self.orchestrator.run("Explain project context of TASK-101")
        self.assertEqual(res["intent"], "TICKET_ANALYSIS")
        self.assertIn("Detailed Ticket Analysis", res["final_response"])

    # 8. Priority Recommendation Flow
    def test_08_priority_recommendation_flow(self):
        res = self.orchestrator.run("Which ticket should be prioritized in Sprint 12?")
        self.assertEqual(res["intent"], "PRIORITY_ANALYSIS")
        self.assertIn("Ticket Priority Recommendation", res["final_response"])

    @patch("src.api.api._send_lark_reply", return_value=True)
    def test_08a_complex_priority_message_bypasses_llm_clarification(self, mock_reply):
        query = (
            "Which ticket should we prioritize in the Australian Retirement Fund CRM - "
            "Sprint CRM Retirement Fund - 2026 - SS02? Why should it be prioritized?"
        )
        payload = {
            "header": {"event_id": "evt_complex_priority", "event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "message_id": "msg_complex_priority",
                    "chat_id": "oc_complex_priority",
                    "content": json.dumps({"text": query}),
                }
            },
        }

        resp = self.client.post("/api/lark/webhook", json=payload)

        self.assertEqual(resp.status_code, 200)
        self.assertIn("Ticket Priority Recommendation", resp.json()["reply"])
        import src.api.api as api_mod
        api_mod._clarifier.check_ambiguity.assert_not_called()
        api_mod._pipeline.run.assert_called_once()
        self.assertEqual(api_mod._pipeline.run.call_args.args[0], query)
        self.mock_ticket_tool.get_tickets_by_sprint.assert_called_once_with(
            "Sprint CRM Retirement Fund - 2026 - SS02"
        )

    @patch("src.api.api._send_lark_reply", return_value=True)
    def test_08b_explicit_ticket_query_bypasses_llm_clarification(self, mock_reply):
        crm_ticket = dict(self.sample_ticket, ticket_id="CRM-050", name="Define Functional Requirements")
        self.mock_ticket_tool.get_ticket_by_id.return_value = {"success": True, "found": True, "ticket": crm_ticket}

        query = "tell me the content of ticket CRM-050"
        payload = {
            "header": {"event_id": "evt_crm050_explicit", "event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "message_id": "msg_crm050_explicit",
                    "chat_id": "oc_crm050_explicit",
                    "content": json.dumps({"text": query}),
                }
            },
        }

        resp = self.client.post("/api/lark/webhook", json=payload)

        self.assertEqual(resp.status_code, 200)
        self.assertIn("CRM-050", resp.json()["reply"])
        import src.api.api as api_mod
        api_mod._clarifier.check_ambiguity.assert_not_called()
        self.mock_ticket_tool.get_ticket_by_id.assert_called_with("CRM-050")

    @patch("src.api.api._send_lark_reply", return_value=True)
    def test_08c_ambiguous_query_retains_clarification(self, mock_reply):
        import src.api.api as api_mod
        api_mod._clarifier.check_ambiguity.return_value = {
            "is_ambiguous": True,
            "options": ["Option 1: Project overview", "Option 2: Sprint planning"],
            "original_query": "What should we work on first?"
        }

        query = "What should we work on first?"
        payload = {
            "header": {"event_id": "evt_ambiguous_test", "event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "message_id": "msg_ambiguous_test",
                    "chat_id": "oc_ambiguous_test",
                    "content": json.dumps({"text": query}),
                }
            },
        }

        resp = self.client.post("/api/lark/webhook", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Intent Clarification", resp.json()["reply"])
        api_mod._clarifier.check_ambiguity.assert_called_once()

    # 9. Approval Request Flow
    def test_09_approval_request_flow(self):
        res = self.orchestrator.run("Apply recommended priority to TASK-101")
        self.assertEqual(res["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertEqual(res["approval_status"], "PENDING_APPROVAL")
        self.assertIsNotNone(res["approval_id"])

    # 10. Approval Decision Flow (Approve)
    def test_10_approval_decision_approve_flow(self):
        req = self.approval_service.create_approval_request(
            self.sample_ticket,
            {"ticket_id": "TASK-101", "current_priority": "MEDIUM", "recommended_priority": "CRITICAL", "weighted_score": 0.85, "confidence": "High", "summary_reason": "Blocker"}
        )
        res = self.orchestrator.run(f"Approve {req.approval_id} by user_pm")
        self.assertEqual(res["intent"], "PRIORITY_APPROVAL_DECISION")
        self.assertEqual(res["approval_status"], "APPROVED")

    # 11. Rejected Approval Flow
    def test_11_rejected_approval_flow(self):
        req = self.approval_service.create_approval_request(
            self.sample_ticket,
            {"ticket_id": "TASK-101", "current_priority": "MEDIUM", "recommended_priority": "CRITICAL", "weighted_score": 0.85, "confidence": "High", "summary_reason": "Blocker"}
        )
        res = self.orchestrator.run(f"Reject {req.approval_id} by user_pm reason: low business value")
        self.assertEqual(res["intent"], "PRIORITY_APPROVAL_DECISION")
        self.assertEqual(res["approval_status"], "REJECTED")

    # 12. Approved Update Execution
    def test_12_approved_update_execution(self):
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

    # 13. Stale Update Detection
    def test_13_stale_update_detection(self):
        req = self.approval_service.create_approval_request(
            self.sample_ticket,
            {"ticket_id": "TASK-101", "current_priority": "MEDIUM", "recommended_priority": "CRITICAL", "weighted_score": 0.85, "confidence": "High", "summary_reason": "Blocker"}
        )
        self.approval_service.approve(req.approval_id, reviewer_id="user_pm")

        stale_ticket = dict(self.sample_ticket, priority="HIGH")
        self.mock_ticket_tool.get_ticket_by_id.return_value = {"success": True, "ticket": stale_ticket}

        res = self.orchestrator.run(f"Apply {req.approval_id} by user_pm")
        self.assertEqual(res["intent"], "PRIORITY_UPDATE_EXECUTION")
        self.assertEqual(res["update_status"], "UPDATE_STALE")

    # 14. Demo Mode Update Simulation
    @patch.dict(os.environ, {"DEMO_MODE": "true"})
    def test_14_demo_mode_update_simulation(self):
        req = self.approval_service.create_approval_request(
            self.sample_ticket,
            {"ticket_id": "TASK-101", "current_priority": "MEDIUM", "recommended_priority": "CRITICAL", "weighted_score": 0.85, "confidence": "High", "summary_reason": "Blocker"}
        )
        self.approval_service.approve(req.approval_id, reviewer_id="user_pm")

        real_write_client = LarkBaseWriteClient(app_token="app123", table_id="tbl123")
        demo_service = ControlledUpdateService(
            approval_service=self.approval_service,
            read_client=self.mock_ticket_tool,
            write_client=real_write_client
        )
        res = demo_service.apply_approved_priority_update(req.approval_id, actor_id="user_pm")
        self.assertEqual(res.status, UpdateResultStatus.UPDATE_VERIFIED)

    # 15. Exception Handling & Safety Response
    def test_15_exception_handling_safety(self):
        self.mock_ticket_tool.get_ticket_by_id.side_effect = Exception("Database transient failure")
        res = self.orchestrator.run("What is the status of TASK-999?")
        self.assertIn("The system failed safely without fabricating data", res["final_response"])

    # 16. Secret Redaction Audit
    def test_16_secret_redaction_audit(self):
        payload = {
            "type": "url_verification",
            "challenge": "challenge_123",
            "token": "secret_token_val"
        }
        resp = self.client.post("/api/lark/webhook", json=payload)
        resp_text = str(resp.json()).lower()
        self.assertNotIn("secret", resp_text)
        self.assertNotIn("lark_app_secret", resp_text)
        self.assertNotIn("bearer", resp_text)

    # 17. SESSION 08 Flow A — FAQ ("How is ticket priority determined?")
    def test_17_session08_flow_a_faq(self):
        res = self.orchestrator.run("How is ticket priority determined?")
        self.assertIn(res["intent"], ["GENERAL_RAG", "FAQ"])
        self.assertTrue(len(res["final_response"]) > 0)
        self.mock_write_client.update_ticket_priority.assert_not_called()

    # 18. SESSION 08 Flow B — Ticket Prioritization ("Which tickets should the team work on first?")
    def test_18_session08_flow_b_priority(self):
        res = self.orchestrator.run("Which tickets should the team work on first?")
        self.assertEqual(res["intent"], "PRIORITY_ANALYSIS")
        self.assertIn("Ticket Priority Recommendation", res["final_response"])
        self.assertTrue(len(res["priority_ranking"]) > 0)
        self.mock_write_client.update_ticket_priority.assert_not_called()

    # 19. SESSION 08 Flow C — Controlled Update Rejection ("Update the priority of TASK-101" -> Rejection)
    def test_19_session08_flow_c_rejection(self):
        req_res = self.orchestrator.run("Update the priority of TASK-101")
        self.assertEqual(req_res["intent"], "PRIORITY_APPROVAL_REQUEST")
        self.assertEqual(req_res["approval_status"], "PENDING_APPROVAL")
        app_id = req_res["approval_id"]

        rej_res = self.orchestrator.run(f"Reject {app_id} by user_pm reason: deferred to next sprint")
        self.assertEqual(rej_res["intent"], "PRIORITY_APPROVAL_DECISION")
        self.assertEqual(rej_res["approval_status"], "REJECTED")

        # Verify executing rejected update fails safely without write
        exec_res = self.orchestrator.run(f"Apply {app_id} by user_pm")
        self.assertEqual(exec_res["intent"], "PRIORITY_UPDATE_EXECUTION")
        self.assertEqual(exec_res["update_status"], "UPDATE_REJECTED")
        self.mock_write_client.update_ticket_priority.assert_not_called()


if __name__ == "__main__":
    unittest.main()
