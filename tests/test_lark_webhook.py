import unittest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from src.api.api import app


class LarkWebhookTests(unittest.TestCase):
    def test_lark_webhook_sends_reply(self):
        class DummyPipeline:
            def run(self, query, chat_id=None, current_message_id=None):
                return {
                    "answer": "Hello from SmartLogi assistant",
                    "retrieved_from": "faq",
                    "sources": [],
                }

        class DummyClarifier:
            def check_ambiguity(self, query, retriever=None):
                return {"is_ambiguous": False, "options": []}

        with patch("src.api.api._get_pipeline", return_value=DummyPipeline()), \
             patch("src.api.api._get_clarifier", return_value=DummyClarifier()), \
             patch("src.api.api._send_lark_reply", return_value=True) as send_reply:
            client = TestClient(app)
            response = client.post(
                "/api/lark/webhook",
                json={
                    "schema": "2.0",
                    "header": {"event_id": "evt_test_1"},
                    "event": {
                        "type": "message",
                        "message": {
                            "message_id": "msg_test_1",
                            "chat_id": "chat_test_1",
                            "content": '{"text": "hello"}',
                        },
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["reply_sent"])
        reply_text = response.json()["reply"]
        self.assertIn("Hello from SmartLogi assistant", reply_text)
        self.assertNotIn("*", reply_text)

    def test_lark_webhook_deduplication(self):
        class DummyPipeline:
            def run(self, query, chat_id=None, current_message_id=None):
                return {"answer": "Res", "retrieved_from": "faq", "sources": []}

        class DummyClarifier:
            def check_ambiguity(self, query, retriever=None):
                return {"is_ambiguous": False, "options": []}

        with patch("src.api.api._get_pipeline", return_value=DummyPipeline()), \
             patch("src.api.api._get_clarifier", return_value=DummyClarifier()), \
             patch("src.api.api._send_lark_reply", return_value=True):
            client = TestClient(app)
            payload = {
                "schema": "2.0",
                "header": {"event_id": "evt_dedup_unique"},
                "event": {
                    "type": "message",
                    "message": {
                        "message_id": "msg_dedup_unique",
                        "chat_id": "chat_dedup",
                        "content": '{"text": "test dedup"}',
                    },
                },
            }
            # First request: processed normally
            res1 = client.post("/api/lark/webhook", json=payload)
            self.assertEqual(res1.status_code, 200)
            self.assertEqual(res1.json().get("status"), "ok")
            self.assertNotEqual(res1.json().get("message"), "duplicate_ignored")

            # Second request with same event_id: duplicate ignored!
            res2 = client.post("/api/lark/webhook", json=payload)
            self.assertEqual(res2.status_code, 200)
            self.assertEqual(res2.json().get("message"), "duplicate_ignored")

    def test_lark_webhook_intent_clarification_sequential_flow(self):
        class DummyPipeline:
            def run(self, query, chat_id=None, current_message_id=None):
                return {"answer": f"Answer for: {query}", "retrieved_from": "faq", "sources": []}

        class AmbiguousClarifier:
            def check_ambiguity(self, query, retriever=None):
                if query == "bug":
                    return {
                        "is_ambiguous": True,
                        "options": [
                            "Report a new bug in the system",
                            "Find bug tracking documentation",
                            "Troubleshoot existing bugs"
                        ]
                    }
                return {"is_ambiguous": False, "options": []}

        with patch("src.api.api._get_pipeline", return_value=DummyPipeline()), \
             patch("src.api.api._get_clarifier", return_value=AmbiguousClarifier()), \
             patch("src.api.api._send_lark_reply", return_value=True):
            client = TestClient(app)

            # Step 1: User sends ambiguous query "bug"
            res1 = client.post(
                "/api/lark/webhook",
                json={
                    "schema": "2.0",
                    "header": {"event_id": "evt_ambig_seq_1"},
                    "event": {
                        "type": "message",
                        "message": {
                            "message_id": "msg_ambig_seq_1",
                            "chat_id": "chat_ambig_seq_user",
                            "content": '{"text": "bug"}',
                        },
                    },
                },
            )
            self.assertEqual(res1.status_code, 200)
            reply1 = res1.json()["reply"]
            self.assertIn("Intent Clarification", reply1)
            self.assertIn("1. Report a new bug in the system", reply1)
            self.assertNotIn("*", reply1)

            # Step 2: User responds with "1" to select option 1
            res2 = client.post(
                "/api/lark/webhook",
                json={
                    "schema": "2.0",
                    "header": {"event_id": "evt_ambig_seq_2"},
                    "event": {
                        "type": "message",
                        "message": {
                            "message_id": "msg_ambig_seq_2",
                            "chat_id": "chat_ambig_seq_user",
                            "content": '{"text": "1"}',
                        },
                    },
                },
            )
            self.assertEqual(res2.status_code, 200)
            reply2 = res2.json()["reply"]
            self.assertIn("Answer for: bug - Report a new bug in the system", reply2)
            self.assertNotIn("*", reply2)

            # Step 3: User subsequently responds with "2" to select option 2 from same session!
            res3 = client.post(
                "/api/lark/webhook",
                json={
                    "schema": "2.0",
                    "header": {"event_id": "evt_ambig_seq_3"},
                    "event": {
                        "type": "message",
                        "message": {
                            "message_id": "msg_ambig_seq_3",
                            "chat_id": "chat_ambig_seq_user",
                            "content": '{"text": "2"}',
                        },
                    },
                },
            )
            self.assertEqual(res3.status_code, 200)
            reply3 = res3.json()["reply"]
            self.assertIn("Answer for: bug - Report a new bug in the system; Find bug tracking documentation", reply3)

            # Step 4: User subsequently responds with "3" to select option 3!
            res4 = client.post(
                "/api/lark/webhook",
                json={
                    "schema": "2.0",
                    "header": {"event_id": "evt_ambig_seq_4"},
                    "event": {
                        "type": "message",
                        "message": {
                            "message_id": "msg_ambig_seq_4",
                            "chat_id": "chat_ambig_seq_user",
                            "content": '{"text": "3"}',
                        },
                    },
                },
            )
            self.assertEqual(res4.status_code, 200)
            reply4 = res4.json()["reply"]
            self.assertIn("Answer for: bug - Report a new bug in the system; Find bug tracking documentation; Troubleshoot existing bugs", reply4)

    def test_save_conversation_history_endpoint(self):
        with patch("src.api.api._save_conversation_history", return_value={"status": "saved", "chat_id": "chat_2", "message_id": "msg_2", "id": 1}):
            client = TestClient(app)
            response = client.post(
                "/api/lark/conversation-history",
                json={
                    "chat_id": "chat_2",
                    "message_id": "msg_2",
                    "sender_type": "user",
                    "sender_id": "user_42",
                    "content": "save this message",
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "saved")
            self.assertEqual(response.json()["chat_id"], "chat_2")


if __name__ == "__main__":
    unittest.main()
