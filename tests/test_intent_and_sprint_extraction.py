import unittest
from src.agent.nodes.intent_router import IntentRouterNode
from src.agent.state import AgentState

class IntentAndSprintExtractionTests(unittest.TestCase):
    def _run_intent(self, query):
        state: AgentState = {"user_query": query}
        result = IntentRouterNode.execute(state)
        return result.get("intent"), result.get("scope")

    def test_priority_analysis_complex(self):
        query = "Which ticket should we prioritize in the Australian Retirement Fund CRM - Sprint CRM Retirement Fund - 2026 - SS02? Why should it be prioritized?"
        intent, _ = self._run_intent(query)
        self.assertEqual(intent, "PRIORITY_ANALYSIS")

    def test_priority_analysis_numeric(self):
        query = "Which ticket should we prioritize in Sprint 12?"
        intent, _ = self._run_intent(query)
        self.assertEqual(intent, "PRIORITY_ANALYSIS")

    def test_priority_analysis_rank(self):
        query = "Rank the tickets for Sprint CRM Retirement Fund - 2026 - SS02."
        intent, _ = self._run_intent(query)
        self.assertEqual(intent, "PRIORITY_ANALYSIS")

    def test_priority_analysis_work_first(self):
        query = "What ticket should we do first in the Australian Retirement Fund CRM?"
        intent, _ = self._run_intent(query)
        self.assertEqual(intent, "PRIORITY_ANALYSIS")

    def test_ambiguous_work_first(self):
        query = "What should we work on first?"
        intent, _ = self._run_intent(query)
        self.assertEqual(intent, "CLARIFICATION_REQUIRED")

    def test_explicit_priority_without_scope_preserves_existing_priority_behavior(self):
        intent, _ = self._run_intent("Which ticket should we prioritize?")
        self.assertEqual(intent, "PRIORITY_ANALYSIS")

    def test_faq_not_priority(self):
        query = "Tell me about Sprint CRM Retirement Fund - 2026 - SS02."
        intent, _ = self._run_intent(query)
        self.assertNotEqual(intent, "PRIORITY_ANALYSIS")

    def test_ticket_query(self):
        query = "What is the status of TASK-101?"
        intent, _ = self._run_intent(query)
        self.assertEqual(intent, "TICKET_QUERY")

    def test_faq(self):
        query = "What is SmartLogi?"
        intent, _ = self._run_intent(query)
        self.assertEqual(intent, "FAQ")

    def test_sprint_extraction_regex(self):
        import re
        pattern = re.compile(r'sprint\s*[^?.!;,\n]+', re.IGNORECASE)
        query = "Sprint CRM Retirement Fund - 2026 - SS02? Why should it be prioritized?"
        match = pattern.search(query)
        self.assertIsNotNone(match)
        sprint = match.group(0).strip().rstrip('?.!;')
        self.assertEqual(sprint, "Sprint CRM Retirement Fund - 2026 - SS02")

    def test_numeric_sprint_regex(self):
        import re
        pattern = re.compile(r'sprint\s*[^?.!;,\n]+', re.IGNORECASE)
        query = "Sprint 12"
        match = pattern.search(query)
        self.assertIsNotNone(match)
        sprint = match.group(0).strip().rstrip('?.!;')
        self.assertEqual(sprint, "Sprint 12")

    def test_crm_ticket_query_content(self):
        query = "tell me the content of ticket CRM-050"
        intent, _ = self._run_intent(query)
        self.assertEqual(intent, "TICKET_QUERY")

    def test_crm_ticket_queries_various(self):
        queries = [
            ("show me CRM-001", "TICKET_QUERY"),
            ("what is the status of CRM-050?", "TICKET_QUERY"),
            ("tell me the description of CRM-050", "TICKET_QUERY"),
            ("give me details about CRM-123", "TICKET_ANALYSIS"),
        ]
        for q, expected_intent in queries:
            intent, _ = self._run_intent(q)
            self.assertEqual(intent, expected_intent, f"Failed for query: {q}")

    def test_standard_ticket_prefixes(self):
        queries = [
            ("tell me the content of ticket TASK-001", "TICKET_QUERY"),
            ("show me BUG-123", "TICKET_QUERY"),
            ("show me EPIC-123", "TICKET_QUERY"),
            ("show me SUB-123", "TICKET_QUERY"),
            ("show me STORY-45", "TICKET_QUERY"),
        ]
        for q, expected_intent in queries:
            intent, _ = self._run_intent(q)
            self.assertEqual(intent, expected_intent, f"Failed for query: {q}")

    def test_non_ticket_crm_queries(self):
        queries = [
            "What are the CRM requirements?",
            "Explain the CRM module.",
            "What is the customer management process?",
        ]
        for q in queries:
            intent, scope = self._run_intent(q)
            self.assertEqual(intent, "GENERAL_RAG", f"Failed for non-ticket query: {q}")
            self.assertEqual(scope, "CRM")

    def test_tool_executor_ticket_id_extraction(self):
        from src.agent.nodes.tool_executor import ToolExecutionNode
        import re
        pattern = re.compile(r'\b(TASK|EPIC|BUG|SUB|CRM|STORY)-\d+\b', re.IGNORECASE)
        
        test_cases = [
            ("tell me the content of ticket CRM-050", "CRM-050"),
            ("show me CRM-001", "CRM-001"),
            ("what is the status of TASK-101?", "TASK-101"),
            ("details on BUG-999", "BUG-999"),
            ("info about EPIC-10", "EPIC-10"),
            ("explain SUB-05", "SUB-05"),
            ("review STORY-88", "STORY-88"),
        ]
        for q, expected_id in test_cases:
            match = pattern.search(q)
            self.assertIsNotNone(match, f"Regex failed to match ticket in '{q}'")
            self.assertEqual(match.group(0).upper(), expected_id)

    def test_rag_pipeline_agent_detection(self):
        import re
        pattern = re.compile(r'\b(task|epic|bug|sub|crm|story)-\d+\b')
        self.assertTrue(bool(pattern.search("tell me the content of ticket crm-050")))
        self.assertTrue(bool(pattern.search("what is the status of task-101?")))
        self.assertFalse(bool(pattern.search("what are the crm requirements?")))
        self.assertFalse(bool(pattern.search("explain the crm module.")))

if __name__ == '__main__':
    unittest.main()
