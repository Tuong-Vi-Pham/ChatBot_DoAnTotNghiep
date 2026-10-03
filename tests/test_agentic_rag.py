import unittest
from unittest.mock import MagicMock, patch

from src.agent.state import AgentState, Evidence
from src.agent.nodes.intent_router import IntentRouterNode
from src.agent.nodes.planner import PlannerNode
from src.agent.nodes.evidence_assessor import EvidenceAssessorNode, QueryRewriterNode
from src.agent.nodes.verifier import VerifierNode
from src.agent.nodes.response_generator import ResponseGeneratorNode
from src.agent.nodes.clarifier import ClarificationNode
from src.agent.tools.ticket_tool import TicketToolAdapter
from src.agent.tools.rag_tool import RAGToolAdapter
from src.agent.graph import AgentOrchestrator, build_agent_graph
from src.lark.lark_base import LarkAPIError


class TestLangGraphAgenticRAG(unittest.TestCase):

    def setUp(self):
        self.mock_ticket_tool = MagicMock(spec=TicketToolAdapter)
        self.mock_rag_tool = MagicMock(spec=RAGToolAdapter)

        self.sample_ticket = {
            "ticket_id": "TASK-101",
            "name": "OAuth SSO Integration",
            "description": "Implement Lark OAuth SSO",
            "status": "In Progress",
            "priority": "HIGH",
            "assignee": "Nguyen Van A",
            "story_points": 5.0,
            "sprint": "Sprint 12",
            "team": "Backend",
            "parent_ticket_id": "EPIC-10",
            "sub_ticket_ids": ["TASK-102"]
        }

        self.sample_evidence_crm = Evidence(
            source="CRM",
            document_id="doc_CRM_01_BRD_docx",
            chunk_id="doc_CRM_01_BRD_docx_chunk_0",
            source_path="CRM/01. Product/BRD.docx",
            content="CRM Investment module business goals and risk policy details.",
            relevance_score=0.85,
            evidence_type="BUSINESS"
        )

        self.sample_evidence_tech = Evidence(
            source="Tech_Team",
            document_id="doc_Tech_Architecture_docx",
            chunk_id="doc_Tech_Architecture_docx_chunk_1",
            source_path="Tech_Team/01_Requirement/Architecture.docx",
            content="OAuth 2.0 authentication service deployment architecture.",
            relevance_score=0.78,
            evidence_type="TECHNICAL"
        )

    # 1. FAQ Routing
    def test_01_faq_routing(self):
        state: AgentState = {"user_query": "What is SmartLogi?"}
        res = IntentRouterNode.execute(state)
        self.assertEqual(res["intent"], "FAQ")
        self.assertEqual(res["scope"], "BOTH")

    # 2. General RAG Routing
    def test_02_general_rag_routing(self):
        state_crm: AgentState = {"user_query": "What does the BRD requirement say about investment risk?"}
        res_crm = IntentRouterNode.execute(state_crm)
        self.assertEqual(res_crm["intent"], "GENERAL_RAG")
        self.assertEqual(res_crm["scope"], "CRM")

        state_tech: AgentState = {"user_query": "Explain the API architecture and deployment steps."}
        res_tech = IntentRouterNode.execute(state_tech)
        self.assertEqual(res_tech["intent"], "GENERAL_RAG")
        self.assertEqual(res_tech["scope"], "Tech_Team")

    # 3. Ticket Query Routing
    def test_03_ticket_query_routing(self):
        state: AgentState = {"user_query": "What is the status of TASK-101?"}
        res = IntentRouterNode.execute(state)
        self.assertEqual(res["intent"], "TICKET_QUERY")

        state_sprint: AgentState = {"user_query": "Show unfinished tickets in Sprint 12."}
        res_sprint = IntentRouterNode.execute(state_sprint)
        self.assertEqual(res_sprint["intent"], "TICKET_QUERY")

    # 4. Priority Analysis Routing
    def test_04_priority_analysis_routing(self):
        state: AgentState = {"user_query": "Which unfinished tickets should we prioritize first?"}
        res = IntentRouterNode.execute(state)
        self.assertEqual(res["intent"], "PRIORITY_ANALYSIS")

    # 5. CRM Scope Retrieval
    def test_05_crm_retrieval(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = {
            "retrieved_from": "document",
            "results": [
                {
                    "content": "CRM BRD content",
                    "metadata": {"category": "CRM", "source_path": "CRM/BRD.docx", "document_id": "doc_CRM"},
                    "score": 0.82
                },
                {
                    "content": "Tech Architecture content",
                    "metadata": {"category": "Tech_Team", "source_path": "Tech_Team/Arch.docx", "document_id": "doc_Tech"},
                    "score": 0.90
                }
            ]
        }
        rag_adapter = RAGToolAdapter(retriever=mock_retriever)
        res = rag_adapter.retrieve("business requirements", scope="CRM", top_k=5)

        self.assertTrue(res["success"])
        self.assertEqual(res["evidence_count"], 1)
        self.assertEqual(res["evidence"][0]["source"], "CRM")

    # 6. Tech_Team Scope Retrieval
    def test_06_tech_team_retrieval(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = {
            "retrieved_from": "document",
            "results": [
                {
                    "content": "CRM BRD content",
                    "metadata": {"category": "CRM", "source_path": "CRM/BRD.docx", "document_id": "doc_CRM"},
                    "score": 0.82
                },
                {
                    "content": "Tech Architecture content",
                    "metadata": {"category": "Tech_Team", "source_path": "Tech_Team/Arch.docx", "document_id": "doc_Tech"},
                    "score": 0.90
                }
            ]
        }
        rag_adapter = RAGToolAdapter(retriever=mock_retriever)
        res = rag_adapter.retrieve("deployment architecture", scope="Tech_Team", top_k=5)

        self.assertTrue(res["success"])
        self.assertEqual(res["evidence_count"], 1)
        self.assertEqual(res["evidence"][0]["source"], "Tech_Team")

    # 7. Combined CRM + Tech_Team Retrieval
    def test_07_combined_crm_tech_team_retrieval(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = {
            "retrieved_from": "document",
            "results": [
                {
                    "content": "CRM BRD content",
                    "metadata": {"category": "CRM", "source_path": "CRM/BRD.docx", "document_id": "doc_CRM"},
                    "score": 0.82
                },
                {
                    "content": "Tech Architecture content",
                    "metadata": {"category": "Tech_Team", "source_path": "Tech_Team/Arch.docx", "document_id": "doc_Tech"},
                    "score": 0.90
                }
            ]
        }
        rag_adapter = RAGToolAdapter(retriever=mock_retriever)
        res = rag_adapter.retrieve("cross functional query", scope="BOTH", top_k=5)

        self.assertTrue(res["success"])
        self.assertEqual(res["evidence_count"], 2)

    # 8. Ticket Retrieval
    def test_08_ticket_retrieval(self):
        mock_service = MagicMock()
        mock_service.get_ticket_by_id.return_value = MagicMock(to_dict=lambda: self.sample_ticket)
        ticket_adapter = TicketToolAdapter(ticket_service=mock_service)

        res = ticket_adapter.get_ticket_by_id("TASK-101")
        self.assertTrue(res["success"])
        self.assertTrue(res["found"])
        self.assertEqual(res["ticket"]["ticket_id"], "TASK-101")

    # 9. Retrieval Retry
    def test_09_retrieval_retry(self):
        low_quality_state: AgentState = {
            "intent": "GENERAL_RAG",
            "evidence_quality": 0.15,
            "iteration_count": 1,
            "retrieval_results": [],
            "ticket_context": []
        }
        self.assertFalse(EvidenceAssessorNode.is_evidence_sufficient(low_quality_state))

        rewritten = QueryRewriterNode.execute({"retrieval_query": "OAuth login", "user_query": "OAuth login"})
        self.assertIn("overview", rewritten["retrieval_query"])

    # 10. Maximum Iteration Safeguard
    def test_10_maximum_iteration(self):
        max_iter_state: AgentState = {
            "intent": "GENERAL_RAG",
            "evidence_quality": 0.10,
            "iteration_count": 2,  # Reached MAX_RETRIEVAL_ITERATIONS (2)
            "retrieval_results": [],
            "ticket_context": []
        }
        # Must evaluate to True to terminate bounded loop cleanly
        self.assertTrue(EvidenceAssessorNode.is_evidence_sufficient(max_iter_state))

    # 11. Missing Evidence Response
    def test_11_missing_evidence(self):
        unverified_state: AgentState = {
            "intent": "GENERAL_RAG",
            "user_query": "Unknown concept",
            "verification_result": {"verified": False},
            "ticket_context": [],
            "project_context": []
        }
        res_node = ResponseGeneratorNode()
        res = res_node.execute(unverified_state)
        self.assertIn("could not find sufficient information", res["final_response"])

    # 12. Lark Failure Handling
    def test_12_lark_failure(self):
        mock_service = MagicMock()
        mock_service.get_ticket_by_id.side_effect = LarkAPIError("Lark API 500 Internal Error")
        ticket_adapter = TicketToolAdapter(ticket_service=mock_service)

        res = ticket_adapter.get_ticket_by_id("TASK-101")
        self.assertFalse(res["success"])
        self.assertIn("500 Internal Error", res["error"])

    # 13. ChromaDB Failure Handling
    def test_13_chromadb_failure(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = Exception("ChromaDB connection lost")
        rag_adapter = RAGToolAdapter(retriever=mock_retriever)

        res = rag_adapter.retrieve("sample query")
        self.assertFalse(res["success"])
        self.assertIn("ChromaDB connection lost", res["error"])

    # 14. LLM Failure Handling
    def test_14_llm_failure(self):
        orchestrator = AgentOrchestrator(
            ticket_tool=self.mock_ticket_tool,
            rag_tool=self.mock_rag_tool
        )
        with patch.object(orchestrator.graph, "invoke", side_effect=Exception("LLM Timeout")):
            res = orchestrator.run("sample query")
            self.assertIn("unexpected error occurred", res["final_response"])
            self.assertEqual(res["error"], "LLM Timeout")

    # 15. End-to-End Ticket Analysis Flow
    def test_15_end_to_end_ticket_analysis_flow(self):
        self.mock_ticket_tool.get_ticket_by_id.return_value = {
            "success": True,
            "found": True,
            "ticket": self.sample_ticket
        }
        self.mock_ticket_tool.get_related_tickets.return_value = {
            "success": True,
            "related_tickets": []
        }
        self.mock_rag_tool.retrieve.return_value = {
            "success": True,
            "evidence": [self.sample_evidence_tech.to_dict()]
        }

        orchestrator = AgentOrchestrator(
            ticket_tool=self.mock_ticket_tool,
            rag_tool=self.mock_rag_tool
        )

        res = orchestrator.run("Detailed analysis of TASK-101")

        self.assertIsNotNone(res)
        self.assertEqual(res["intent"], "TICKET_ANALYSIS")
        self.assertIsNotNone(res["ticket_data"])
        self.assertIn("TASK-101", res["final_response"])
        self.assertTrue(res["verification_result"].get("verified", False))


if __name__ == "__main__":
    unittest.main()
