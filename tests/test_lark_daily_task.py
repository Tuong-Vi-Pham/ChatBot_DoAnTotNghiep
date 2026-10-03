import unittest
from unittest.mock import MagicMock, patch
import requests

from src.lark.auth import TenantAccessTokenManager
from src.lark.lark_base import (
    LarkAPIError,
    LarkAuthError,
    LarkBaseClient,
    LarkConfigError,
    LarkRateLimitError,
    Ticket,
    TicketNormalizer,
    TicketService,
)


class TestLarkDailyTaskDataLayer(unittest.TestCase):

    def setUp(self):
        self.mock_app_token = "app_test_token_123"
        self.mock_table_id = "tbl_daily_task_456"

        self.mock_token_manager = MagicMock(spec=TenantAccessTokenManager)
        self.mock_token_manager.get_token.return_value = "mock_tenant_access_token_xyz"

        self.client = LarkBaseClient(
            token_manager=self.mock_token_manager,
            app_token=self.mock_app_token,
            table_id=self.mock_table_id
        )
        self.service = TicketService(client=self.client)

        self.sample_raw_records = [
            {
                "record_id": "rec_001",
                "fields": {
                    "Ticket ID": "TASK-101",
                    "Name": "Implement Login OAuth",
                    "Description": "Support Lark OAuth SSO login flow",
                    "Start Date": 1771650000000,
                    "End Date": 1772000000000,
                    "Status": "In Progress",
                    "Attachment": [{"file_token": "att_1", "name": "spec.png"}],
                    "Sub Ticket": ["TASK-102", "TASK-103"],
                    "Parent Ticket": "EPIC-10",
                    "Assignee": [{"name": "Nguyen Van A", "id": "ou_001"}],
                    "Creator / Reporter": {"name": "Tran Van B"},
                    "Size": "M",
                    "Story Point": 5.0,
                    "Release Note": "OAuth support added",
                    "Priority": "High",
                    "Team": "Backend",
                    "Label": ["Authentication", "Security"],
                    "Sprint": "Sprint 12"
                }
            },
            {
                "record_id": "rec_002",
                "fields": {
                    "Ticket ID": "TASK-102",
                    "Name": "OAuth Token Refresh",
                    "Description": "Implement refresh token logic",
                    "Start Date": "2026-08-21",
                    "End Date": "2026-08-25",
                    "Status": "To Do",
                    "Sub Ticket": [],
                    "Parent Ticket": "TASK-101",
                    "Assignee": "Nguyen Van A",
                    "Creator": "Tran Van B",
                    "Size": "S",
                    "Story Point": 2,
                    "Priority": "Medium",
                    "Team": "Backend",
                    "Label": "Authentication",
                    "Sprint": "Sprint 12"
                }
            },
            {
                "record_id": "rec_003",
                "fields": {
                    "Ticket ID": "EPIC-10",
                    "Name": "Authentication Overhaul Epic",
                    "Status": "In Progress",
                    "Team": "Backend",
                    "Sprint": "Sprint 12"
                }
            }
        ]

    # 1. Authentication Test
    def test_01_authentication(self):
        token = self.client.token_manager.get_token()
        self.assertEqual(token, "mock_tenant_access_token_xyz")
        headers = self.client._get_headers()
        self.assertEqual(headers["Authorization"], "Bearer mock_tenant_access_token_xyz")

    # 2. Successful Ticket Retrieval
    @patch("requests.get")
    def test_02_successful_ticket_retrieval(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "code": 0,
            "msg": "success",
            "data": {
                "has_more": False,
                "items": self.sample_raw_records
            }
        }
        mock_get.return_value = mock_response

        tickets, next_token, has_more = self.service.get_tickets()

        self.assertEqual(len(tickets), 3)
        self.assertFalse(has_more)
        self.assertIsNone(next_token)

        t1 = tickets[0]
        self.assertEqual(t1.ticket_id, "TASK-101")
        self.assertEqual(t1.name, "Implement Login OAuth")
        self.assertEqual(t1.status, "In Progress")
        self.assertEqual(t1.story_points, 5.0)
        self.assertEqual(t1.assignee, "Nguyen Van A")
        self.assertEqual(t1.parent_ticket_id, "EPIC-10")
        self.assertEqual(t1.sub_ticket_ids, ["TASK-102", "TASK-103"])
        self.assertEqual(t1.provenance["source"], "Lark")
        self.assertEqual(t1.provenance["source_table"], "Daily_Task")

    # 3. Empty Daily_Task Table
    @patch("requests.get")
    def test_03_empty_daily_task(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "code": 0,
            "msg": "success",
            "data": {
                "has_more": False,
                "items": []
            }
        }
        mock_get.return_value = mock_response

        tickets, next_token, has_more = self.service.get_tickets()
        self.assertEqual(len(tickets), 0)
        self.assertFalse(has_more)
        self.assertIsNone(next_token)

    # 4. Pagination Test
    @patch("requests.get")
    def test_04_pagination(self, mock_get):
        page1_response = MagicMock()
        page1_response.status_code = 200
        page1_response.json.return_value = {
            "code": 0,
            "msg": "success",
            "data": {
                "has_more": True,
                "page_token": "token_page_2",
                "items": [self.sample_raw_records[0]]
            }
        }

        page2_response = MagicMock()
        page2_response.status_code = 200
        page2_response.json.return_value = {
            "code": 0,
            "msg": "success",
            "data": {
                "has_more": False,
                "page_token": None,
                "items": [self.sample_raw_records[1]]
            }
        }

        mock_get.side_effect = [page1_response, page2_response]

        all_tickets = self.service.get_all_tickets()
        self.assertEqual(len(all_tickets), 2)
        self.assertEqual(all_tickets[0].ticket_id, "TASK-101")
        self.assertEqual(all_tickets[1].ticket_id, "TASK-102")
        self.assertEqual(mock_get.call_count, 2)

    # 5. Ticket by ID Lookup
    @patch.object(TicketService, "get_all_tickets")
    def test_05_ticket_by_id(self, mock_get_all):
        mock_get_all.return_value = [TicketNormalizer.normalize(r) for r in self.sample_raw_records]

        ticket = self.service.get_ticket_by_id("TASK-102")
        self.assertIsNotNone(ticket)
        self.assertEqual(ticket.name, "OAuth Token Refresh")

        ticket_by_rec = self.service.get_ticket_by_id("rec_001")
        self.assertIsNotNone(ticket_by_rec)
        self.assertEqual(ticket_by_rec.ticket_id, "TASK-101")

        not_found = self.service.get_ticket_by_id("NON_EXISTENT")
        self.assertIsNone(not_found)

    # 6. Sprint Filtering
    @patch.object(TicketService, "get_all_tickets")
    def test_06_sprint_filtering(self, mock_get_all):
        mock_get_all.return_value = [TicketNormalizer.normalize(r) for r in self.sample_raw_records]

        sprint_tickets = self.service.get_tickets_by_sprint("Sprint 12")
        self.assertEqual(len(sprint_tickets), 3)

        empty_sprint = self.service.get_tickets_by_sprint("Sprint 99")
        self.assertEqual(len(empty_sprint), 0)

    # 7. Team Filtering
    @patch.object(TicketService, "get_all_tickets")
    def test_07_team_filtering(self, mock_get_all):
        mock_get_all.return_value = [TicketNormalizer.normalize(r) for r in self.sample_raw_records]

        backend_tickets = self.service.get_tickets_by_team("Backend")
        self.assertEqual(len(backend_tickets), 3)

        frontend_tickets = self.service.get_tickets_by_team("Frontend")
        self.assertEqual(len(frontend_tickets), 0)

    # 8. Status Filtering
    @patch.object(TicketService, "get_all_tickets")
    def test_08_status_filtering(self, mock_get_all):
        mock_get_all.return_value = [TicketNormalizer.normalize(r) for r in self.sample_raw_records]

        todo_tickets = self.service.get_tickets_by_status("To Do")
        self.assertEqual(len(todo_tickets), 1)
        self.assertEqual(todo_tickets[0].ticket_id, "TASK-102")

        in_progress = self.service.get_tickets_by_status("In Progress")
        self.assertEqual(len(in_progress), 2)

    # 9. Related Ticket Retrieval
    @patch.object(TicketService, "get_all_tickets")
    def test_09_related_tickets(self, mock_get_all):
        mock_get_all.return_value = [TicketNormalizer.normalize(r) for r in self.sample_raw_records]

        related_to_101 = self.service.get_related_tickets("TASK-101")
        related_ids = [t.ticket_id for t in related_to_101]
        self.assertIn("EPIC-10", related_ids)  # Parent
        self.assertIn("TASK-102", related_ids) # Sub ticket

    # 10. Missing Optional Fields
    def test_10_missing_optional_fields(self):
        minimal_record = {
            "record_id": "rec_min",
            "fields": {
                "Ticket ID": "TASK-MINIMAL",
                "Name": "Minimal Ticket"
            }
        }
        ticket = TicketNormalizer.normalize(minimal_record)

        self.assertEqual(ticket.ticket_id, "TASK-MINIMAL")
        self.assertEqual(ticket.name, "Minimal Ticket")
        self.assertEqual(ticket.description, "")
        self.assertEqual(ticket.status, "UNKNOWN")
        self.assertEqual(ticket.priority, "UNKNOWN")
        self.assertIsNone(ticket.story_points)
        self.assertEqual(ticket.labels, [])

        validation = self.service.validate_ticket(ticket)
        self.assertTrue(validation["is_valid"])
        self.assertGreater(len(validation["warnings"]), 0)

    # 11. Invalid Field Types
    def test_11_invalid_field_types(self):
        invalid_types_record = {
            "record_id": "rec_invalid",
            "fields": {
                "Ticket ID": 9999,  # integer ID
                "Name": ["Complex", "Name", "Object"],
                "Story Point": "five",  # Non-numeric string
                "Start Date": "invalid_date_string",
                "Assignee": {"unexpected_key": 123}
            }
        }
        ticket = TicketNormalizer.normalize(invalid_types_record)

        self.assertEqual(ticket.ticket_id, "9999")
        self.assertIn("Complex", ticket.name)
        self.assertIsNone(ticket.story_points)
        self.assertEqual(ticket.start_date, "invalid_date_string")

    # 12. API Timeout
    @patch("requests.get")
    def test_12_api_timeout(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        with self.assertRaises(LarkAPIError) as ctx:
            self.client.get_records()
        self.assertIn("timed out", str(ctx.exception))

    # 13. API Authorization Failure
    @patch("requests.get")
    def test_13_api_authorization_failure(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized - Invalid Tenant Token"
        mock_get.return_value = mock_response

        with self.assertRaises(LarkAuthError) as ctx:
            self.client.get_records()
        self.assertIn("401", str(ctx.exception))

    # 14. Rate Limiting
    @patch("requests.get")
    def test_14_rate_limiting(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Too Many Requests"
        mock_get.return_value = mock_response

        with self.assertRaises(LarkRateLimitError) as ctx:
            self.client.get_records()
        self.assertIn("429", str(ctx.exception))

    # 16. Actual Lark Schema CRM-050 Normalization
    def test_16_actual_lark_schema_crm050_normalization(self):
        raw_crm_record = {
            "record_id": "recvsVfHe2b45h",
            "fields": {
                "ID Ticket": "CRM-050",
                "Name": "Define Functional Requirements",
                "Description": "Define functional requirements for all CRM modules including Customer, Contract, Investment, Support, Dashboard and Reporting.",
                "Start Date": "14/09/2026",
                "End Date": "17/09/2026",
                "Status": "To Do",
                "Assignee": "System Analyst",
                "Creator/ Reporter": "Product Owner",
                "Size": "L",
                "Story Point": 11,
                "Release Note": "Functional requirements baseline",
                "Priority": "Critical",
                "Team": "Requirements",
                "Label": ["FR", "SRS"],
                "Sprint": "CRM Retirement Fund - 2026 - SS02",
                "Sub-Ticket": ["CRM-051"]
            }
        }
        ticket = TicketNormalizer.normalize(raw_crm_record)

        self.assertEqual(ticket.ticket_id, "CRM-050")
        self.assertEqual(ticket.name, "Define Functional Requirements")
        self.assertEqual(ticket.creator, "Product Owner")
        self.assertEqual(ticket.reporter, "Product Owner")
        self.assertEqual(ticket.sub_ticket_ids, ["CRM-051"])
        self.assertEqual(ticket.status, "To Do")
        self.assertEqual(ticket.priority, "Critical")
        self.assertEqual(ticket.story_points, 11.0)
        self.assertEqual(ticket.record_id, "recvsVfHe2b45h")

    # 17. Backward-compatible Aliases and Fallback
    def test_17_backward_compatible_aliases_and_fallback(self):
        # Existing "Ticket ID" format
        rec_alias = {
            "record_id": "rec_001",
            "fields": {
                "Ticket ID": "TASK-101",
                "Sub Ticket": ["TASK-102"],
                "Creator / Reporter": "Tech Lead"
            }
        }
        t1 = TicketNormalizer.normalize(rec_alias)
        self.assertEqual(t1.ticket_id, "TASK-101")
        self.assertEqual(t1.sub_ticket_ids, ["TASK-102"])
        self.assertEqual(t1.creator, "Tech Lead")

        # Missing ticket ID falls back safely to record_id
        rec_no_id = {
            "record_id": "rec_fallback_999",
            "fields": {
                "Name": "Task without explicit ID"
            }
        }
        t2 = TicketNormalizer.normalize(rec_no_id)
        self.assertEqual(t2.ticket_id, "rec_fallback_999")

    # 18. Sprint Normalization Matching
    def test_18_sprint_normalization_matching(self):
        sample_records = [
            {"record_id": "rec1", "fields": {"ID Ticket": "CRM-044", "Sprint": "CRM Retirement Fund - 2026 - SS02"}},
            {"record_id": "rec2", "fields": {"ID Ticket": "CRM-045", "Sprint": "CRM Retirement Fund - 2026 - SS02"}},
            {"record_id": "rec3", "fields": {"ID Ticket": "CRM-001", "Sprint": "CRM Retirement Fund - 2026 - SS01"}},
            {"record_id": "rec4", "fields": {"ID Ticket": "TASK-101", "Sprint": "Sprint 12"}},
        ]
        with patch.object(self.client, "get_records", return_value=(sample_records, None, False)):
            # 1. Exact sprint name
            res_exact = self.service.get_tickets_by_sprint("CRM Retirement Fund - 2026 - SS02")
            self.assertEqual(len(res_exact), 2)
            self.assertEqual({t.ticket_id for t in res_exact}, {"CRM-044", "CRM-045"})

            # 2. User query with "Sprint " prefix
            res_user = self.service.get_tickets_by_sprint("Sprint CRM Retirement Fund - 2026 - SS02")
            self.assertEqual(len(res_user), 2)
            self.assertEqual({t.ticket_id for t in res_user}, {"CRM-044", "CRM-045"})

            # 3. Case insensitivity
            res_case = self.service.get_tickets_by_sprint("sprint crm retirement fund - 2026 - ss02")
            self.assertEqual(len(res_case), 2)

            # 4. Whitespace tolerance
            res_ws = self.service.get_tickets_by_sprint("   Sprint CRM Retirement Fund - 2026 - SS02   ")
            self.assertEqual(len(res_ws), 2)

            # 5. Unrelated sprint does not match
            res_other = self.service.get_tickets_by_sprint("Sprint CRM Retirement Fund - 2026 - SS03")
            self.assertEqual(len(res_other), 0)

            # 6. Numeric sprint with / without prefix
            res_num1 = self.service.get_tickets_by_sprint("Sprint 12")
            self.assertEqual(len(res_num1), 1)
            self.assertEqual(res_num1[0].ticket_id, "TASK-101")

            res_num2 = self.service.get_tickets_by_sprint("12")
            self.assertEqual(len(res_num2), 1)
            self.assertEqual(res_num2[0].ticket_id, "TASK-101")


if __name__ == "__main__":
    unittest.main()
