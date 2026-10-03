import unittest
from unittest.mock import MagicMock, patch

from src.agent.state import AgentState, Evidence
from src.agent.priority.priority_model import FactorScore, PriorityAnalysisResult, RankedTicket
from src.agent.priority.priority_rules import PriorityRuleEvaluator
from src.agent.priority.priority_ranker import PriorityRanker
from src.agent.priority.priority_verifier import PriorityVerifier
from src.agent.nodes.priority_analyzer import PriorityAnalyzerNode
from src.agent.tools.ticket_tool import TicketToolAdapter
from src.agent.tools.rag_tool import RAGToolAdapter
from src.agent.graph import AgentOrchestrator


class TestPriorityEngine(unittest.TestCase):

    def setUp(self):
        self.mock_ticket_tool = MagicMock(spec=TicketToolAdapter)
        self.mock_rag_tool = MagicMock(spec=RAGToolAdapter)

        self.t1 = {
            "ticket_id": "TASK-101",
            "name": "OAuth Login Feature",
            "status": "In Progress",
            "priority": "MEDIUM",
            "sprint": "Sprint 12",
            "story_points": 5.0,
            "sub_ticket_ids": ["TASK-102", "TASK-103", "TASK-104"],
            "release_note": "Required for v1.2 release"
        }

        self.t2 = {
            "ticket_id": "TASK-102",
            "name": "Refactor Logging",
            "status": "To Do",
            "priority": "HIGH",
            "sprint": "Sprint 12",
            "story_points": 2.0,
            "sub_ticket_ids": []
        }

        self.t_done = {
            "ticket_id": "TASK-100",
            "name": "Initial Setup",
            "status": "Done",
            "priority": "CRITICAL",
            "sprint": "Sprint 11"
        }

        self.t_cancelled = {
            "ticket_id": "TASK-099",
            "name": "Deprecate Legacy API",
            "status": "Cancelled",
            "priority": "MEDIUM"
        }

        self.crm_ev = [
            Evidence(
                source="CRM",
                document_id="doc_CRM_BRD",
                chunk_id="doc_CRM_BRD_c0",
                source_path="CRM/BRD.docx",
                content="Mandatory core requirement for user authentication and SRS compliance.",
                relevance_score=0.85,
                evidence_type="BUSINESS"
            ).to_dict()
        ]

        self.tech_ev = [
            Evidence(
                source="Tech_Team",
                document_id="doc_Tech_Arch",
                chunk_id="doc_Tech_Arch_c1",
                source_path="Tech_Team/Architecture.docx",
                content="Authentication service core architecture and API deployment specifications.",
                relevance_score=0.80,
                evidence_type="TECHNICAL"
            ).to_dict()
        ]

    # 1. Single Ticket Priority Analysis
    def test_01_single_ticket_priority_analysis(self):
        res = PriorityRuleEvaluator.evaluate_ticket(self.t1, [self.t1, self.t2], self.crm_ev, self.tech_ev)
        self.assertEqual(res.ticket_id, "TASK-101")
        self.assertEqual(res.current_priority, "MEDIUM")
        self.assertEqual(res.recommended_priority, "CRITICAL")
        self.assertTrue(res.differs_from_current)

    # 2. Multiple Ticket Ranking
    def test_02_multiple_ticket_ranking(self):
        ranked = PriorityRanker.rank_tickets([self.t1, self.t2], self.crm_ev, self.tech_ev)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].rank, 1)
        self.assertEqual(ranked[1].rank, 2)
        self.assertTrue(ranked[0].weighted_score >= ranked[1].weighted_score)

    # 3. Current Priority Preservation
    def test_03_current_priority_preservation(self):
        res = PriorityRuleEvaluator.evaluate_ticket(self.t1, [self.t1], self.crm_ev, self.tech_ev)
        self.assertEqual(res.current_priority, "MEDIUM")  # Original preserved
        self.assertEqual(self.t1["priority"], "MEDIUM")  # Original dict untouched

    # 4. Business Impact Evidence
    def test_04_business_impact_evidence(self):
        res_with = PriorityRuleEvaluator.evaluate_ticket(self.t1, [self.t1], self.crm_ev, [])
        res_without = PriorityRuleEvaluator.evaluate_ticket(self.t1, [self.t1], [], [])
        self.assertGreater(res_with.factors["BUSINESS"].score, res_without.factors["BUSINESS"].score)

    # 5. Technical Impact Evidence
    def test_05_technical_impact_evidence(self):
        res_with = PriorityRuleEvaluator.evaluate_ticket(self.t1, [self.t1], [], self.tech_ev)
        res_without = PriorityRuleEvaluator.evaluate_ticket(self.t1, [self.t1], [], [])
        self.assertGreater(res_with.factors["TECHNICAL"].score, res_without.factors["TECHNICAL"].score)

    # 6. Dependency Analysis
    def test_06_dependency_analysis(self):
        res_blocker = PriorityRuleEvaluator.evaluate_ticket(self.t1, [self.t1, self.t2], [], [])
        res_leaf = PriorityRuleEvaluator.evaluate_ticket(self.t2, [self.t1, self.t2], [], [])
        self.assertGreater(res_blocker.factors["DEPENDENCY"].score, res_leaf.factors["DEPENDENCY"].score)

    # 7. Sprint Urgency
    def test_07_sprint_urgency(self):
        t_active = {"ticket_id": "T1", "sprint": "Sprint 12", "status": "In Progress"}
        t_future = {"ticket_id": "T2", "sprint": "Sprint 15", "status": "To Do"}
        res_active = PriorityRuleEvaluator.evaluate_ticket(t_active, [t_active, t_future], [], [])
        res_future = PriorityRuleEvaluator.evaluate_ticket(t_future, [t_active, t_future], [], [])
        self.assertGreater(res_active.factors["SCHEDULE"].score, res_future.factors["SCHEDULE"].score)

    # 8. Release Impact
    def test_08_release_impact(self):
        res = PriorityRuleEvaluator.evaluate_ticket(self.t1, [self.t1], [], [])
        self.assertEqual(res.factors["RELEASE"].score, 0.85)

    # 9. Story Point Handling
    def test_09_story_point_handling(self):
        res_small = PriorityRuleEvaluator.evaluate_ticket(self.t2, [self.t2], [], [])
        res_large = PriorityRuleEvaluator.evaluate_ticket(self.t1, [self.t1], [], [])
        self.assertGreater(res_small.factors["EFFORT"].score, res_large.factors["EFFORT"].score)

    # 10. Completed Ticket Exclusion
    def test_10_completed_ticket_exclusion(self):
        active = PriorityRanker.filter_active_candidates([self.t1, self.t_done])
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["ticket_id"], "TASK-101")

    # 11. Cancelled Ticket Exclusion
    def test_11_cancelled_ticket_exclusion(self):
        active = PriorityRanker.filter_active_candidates([self.t2, self.t_cancelled])
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["ticket_id"], "TASK-102")

    # 12. Missing CRM Evidence
    def test_12_missing_crm_evidence(self):
        res = PriorityRuleEvaluator.evaluate_ticket(self.t2, [self.t2], None, self.tech_ev)
        self.assertEqual(res.factors["BUSINESS"].evidence_quality, "INSUFFICIENT_EVIDENCE")

    # 13. Missing Tech Evidence
    def test_13_missing_tech_evidence(self):
        res = PriorityRuleEvaluator.evaluate_ticket(self.t2, [self.t2], self.crm_ev, None)
        self.assertEqual(res.factors["TECHNICAL"].evidence_quality, "INSUFFICIENT_EVIDENCE")

    # 14. Missing Dependency Data
    def test_14_missing_dependency_data(self):
        res = PriorityRuleEvaluator.evaluate_ticket(self.t2, [self.t2], [], [])
        self.assertEqual(res.factors["DEPENDENCY"].score, 0.20)

    # 15. Low Evidence Confidence
    def test_15_low_evidence_confidence(self):
        res = PriorityRuleEvaluator.evaluate_ticket(self.t2, [self.t2], [], [])
        self.assertEqual(res.confidence, "Low")

    # 16. Priority Change Recommendation
    def test_16_priority_change_recommendation(self):
        res = PriorityRuleEvaluator.evaluate_ticket(self.t1, [self.t1, self.t2], self.crm_ev, self.tech_ev)
        self.assertEqual(res.current_priority, "MEDIUM")
        self.assertEqual(res.recommended_priority, "CRITICAL")
        self.assertTrue(res.differs_from_current)

    # 17. No Priority Change Recommendation
    def test_17_no_priority_change_recommendation(self):
        res = PriorityRuleEvaluator.evaluate_ticket(self.t2, [self.t2], [], [])
        self.assertEqual(res.current_priority, "HIGH")
        self.assertEqual(res.recommended_priority, "HIGH")
        self.assertFalse(res.differs_from_current)

    # 18. Deterministic Tie-Breaking
    def test_18_deterministic_tie_breaking(self):
        # Run ranking twice on identical inputs
        ranked1 = PriorityRanker.rank_tickets([self.t1, self.t2], self.crm_ev, self.tech_ev)
        ranked2 = PriorityRanker.rank_tickets([self.t1, self.t2], self.crm_ev, self.tech_ev)
        self.assertEqual([r.ticket_id for r in ranked1], [r.ticket_id for r in ranked2])

    # 19. Scoped Sprint Ranking
    def test_19_scoped_sprint_ranking(self):
        t_sprint11 = {"ticket_id": "T11", "status": "In Progress", "sprint": "Sprint 11"}
        ranked = PriorityRanker.rank_tickets([self.t1, t_sprint11], self.crm_ev, self.tech_ev)
        self.assertEqual(ranked[0].ticket_id, "TASK-101")

    # 20. Scoped Team Ranking
    def test_20_scoped_team_ranking(self):
        t_be = {"ticket_id": "TBE", "team": "Backend", "status": "In Progress"}
        t_fe = {"ticket_id": "TFE", "team": "Frontend", "status": "In Progress"}
        ranked = PriorityRanker.rank_tickets([t_be, t_fe], [], [])
        self.assertEqual(len(ranked), 2)

    # 21. Release-Focused Ranking
    def test_21_release_focused_ranking(self):
        t_rel = {"ticket_id": "T_REL", "release_note": "v1.0 Blocker", "status": "In Progress"}
        t_norel = {"ticket_id": "T_NOREL", "status": "In Progress"}
        ranked = PriorityRanker.rank_tickets([t_norel, t_rel], [], [])
        self.assertEqual(ranked[0].ticket_id, "T_REL")

    # 22. Verification Failure
    def test_22_verification_failure(self):
        analysis = PriorityRuleEvaluator.evaluate_ticket(self.t1, [self.t1], [], [])
        v = PriorityVerifier.verify_analysis(analysis, self.t1)
        self.assertTrue(v["verified"])

        # Tamper current_priority
        analysis.current_priority = "P0"
        v_bad = PriorityVerifier.verify_analysis(analysis, self.t1)
        self.assertFalse(v_bad["verified"])

    # 23. Lark Failure Graceful
    def test_23_lark_failure_graceful(self):
        state: AgentState = {
            "intent": "PRIORITY_ANALYSIS",
            "ticket_context": [],
            "project_context": []
        }
        res = PriorityAnalyzerNode.execute(state)
        self.assertEqual(res.get("priority_ranking"), [])

    # 24. RAG Failure Graceful
    def test_24_rag_failure_graceful(self):
        state: AgentState = {
            "intent": "PRIORITY_ANALYSIS",
            "ticket_context": [self.t2],
            "project_context": []  # Empty RAG
        }
        res = PriorityAnalyzerNode.execute(state)
        self.assertIsNotNone(res.get("priority_analysis"))
        self.assertEqual(res["priority_analysis"]["confidence"], "Low")

    # 25. End-to-End LangGraph Priority Analysis
    def test_25_end_to_end_langgraph_priority_analysis(self):
        self.mock_ticket_tool.get_all_tickets.return_value = {
            "success": True,
            "tickets": [self.t1, self.t2]
        }
        self.mock_rag_tool.retrieve.return_value = {
            "success": True,
            "evidence": self.crm_ev + self.tech_ev
        }

        orchestrator = AgentOrchestrator(
            ticket_tool=self.mock_ticket_tool,
            rag_tool=self.mock_rag_tool
        )

        res = orchestrator.run("Which unfinished tickets should we prioritize first?")
        self.assertEqual(res["intent"], "PRIORITY_ANALYSIS")
        self.assertIsNotNone(res["priority_ranking"])
        self.assertIn("Priority Recommendation", res["final_response"])
        self.assertTrue(res["verification_result"].get("verified", False))

    # 26. Exact Boundary Threshold Values
    def test_26_boundary_threshold_values(self):
        expected = {
            0.80: "CRITICAL", 0.65: "CRITICAL", 0.66: "CRITICAL",
            0.64: "HIGH", 0.50: "HIGH", 0.45: "HIGH",
            0.44: "MEDIUM", 0.30: "MEDIUM", 0.10: "MEDIUM"
        }
        for score, prio in expected.items():
            self.assertEqual(PriorityRuleEvaluator.map_score_to_priority(score), prio)


    # 27. Target Query End-to-End Priority Flow
    def test_27_target_query_e2e_priority_flow(self):
        sample_crm_tickets = [
            {
                "ticket_id": "CRM-044",
                "name": "Create Product Roadmap",
                "status": "To Do",
                "priority": "Critical",
                "sprint": "CRM Retirement Fund - 2026 - SS02",
                "story_points": 8.0,
                "sub_ticket_ids": ["CRM-045", "CRM-046"],
                "release_note": "Required for MVP release"
            },
            {
                "ticket_id": "CRM-045",
                "name": "Define MVP Scope",
                "status": "To Do",
                "priority": "Critical",
                "sprint": "CRM Retirement Fund - 2026 - SS02",
                "story_points": 3.0,
                "parent_ticket_id": "CRM-044"
            }
        ]
        self.mock_ticket_tool.get_tickets_by_sprint.return_value = {
            "success": True,
            "tickets": sample_crm_tickets,
            "sprint": "Sprint CRM Retirement Fund - 2026 - SS02",
            "count": 2
        }
        self.mock_rag_tool.retrieve.return_value = {
            "success": True,
            "evidence": self.crm_ev
        }

        orchestrator = AgentOrchestrator(
            ticket_tool=self.mock_ticket_tool,
            rag_tool=self.mock_rag_tool
        )

        query = "Which ticket should we prioritize in the Australian Retirement Fund CRM - Sprint CRM Retirement Fund - 2026 - SS02?"
        res = orchestrator.run(query)

        self.assertEqual(res["intent"], "PRIORITY_ANALYSIS")
        self.assertEqual(len(res["priority_ranking"]), 2)
        self.assertEqual(res["priority_ranking"][0]["ticket_id"], "CRM-044")
        self.assertIn("### Recommended Ticket", res["final_response"])
        self.assertIn("CRM-044", res["final_response"])
        self.assertIn("### Priority Ranking", res["final_response"])
        self.assertIn("### Current vs Recommended Priority", res["final_response"])
        self.mock_ticket_tool.get_tickets_by_sprint.assert_called_with("Sprint CRM Retirement Fund - 2026 - SS02")

    # 28. Empty Lark Tickets Produces Safe Insufficient Evidence
    def test_28_empty_lark_tickets_returns_safe_insufficient_evidence(self):
        self.mock_ticket_tool.get_tickets_by_sprint.return_value = {
            "success": True,
            "tickets": [],
            "sprint": "Sprint CRM Retirement Fund - 2026 - NonExistent",
            "count": 0
        }
        self.mock_rag_tool.retrieve.return_value = {
            "success": True,
            "evidence": self.crm_ev  # RAG returns chunks, but Lark returns 0 tickets
        }

        orchestrator = AgentOrchestrator(
            ticket_tool=self.mock_ticket_tool,
            rag_tool=self.mock_rag_tool
        )

        query = "Which ticket should we prioritize in Sprint CRM Retirement Fund - 2026 - NonExistent?"
        res = orchestrator.run(query)

        self.assertEqual(res["intent"], "PRIORITY_ANALYSIS")
        self.assertEqual(len(res["priority_ranking"]), 0)
        self.assertFalse(res["verification_result"]["verified"])
        self.assertIn("Unable to determine the highest-priority ticket because no eligible tickets were retrieved", res["final_response"])
        self.assertNotIn("Here is the recommended ticket execution priority order", res["final_response"])

    # 29. Recommended Priority is Advisory and Preserves Original Lark Priority
    def test_29_priority_advisory_preserves_lark_data(self):
        raw_ticket = {
            "ticket_id": "CRM-050",
            "name": "Define Functional Requirements",
            "status": "To Do",
            "priority": "Critical",
            "sprint": "CRM Retirement Fund - 2026 - SS02",
            "story_points": 11.0
        }
        self.mock_ticket_tool.get_tickets_by_sprint.return_value = {
            "success": True,
            "tickets": [raw_ticket],
            "count": 1
        }
        self.mock_rag_tool.retrieve.return_value = {"success": True, "evidence": self.crm_ev}

        orchestrator = AgentOrchestrator(ticket_tool=self.mock_ticket_tool, rag_tool=self.mock_rag_tool)
        res = orchestrator.run("Which ticket should we prioritize in Sprint CRM Retirement Fund - 2026 - SS02?")

        # Check raw ticket was not mutated
        self.assertEqual(raw_ticket["priority"], "Critical")
        self.assertIn("Current Priority (Lark Daily_Task):** Preserved", res["final_response"])
        self.assertIn("Recommended Priority:** Advisory", res["final_response"])

    # 30. CRM-050 Ticket Query Retrieval Remains Intact
    def test_30_crm050_ticket_retrieval_regression(self):
        crm_050 = {
            "ticket_id": "CRM-050",
            "name": "Define Functional Requirements",
            "status": "To Do",
            "priority": "Critical",
            "sprint": "CRM Retirement Fund - 2026 - SS02"
        }
        self.mock_ticket_tool.get_ticket_by_id.return_value = {"success": True, "ticket": crm_050, "found": True}

        orchestrator = AgentOrchestrator(ticket_tool=self.mock_ticket_tool, rag_tool=self.mock_rag_tool)
        res = orchestrator.run("tell me the content of ticket CRM-050")

        self.assertEqual(res["intent"], "TICKET_QUERY")
        self.assertIn("CRM-050", res["final_response"])
        self.assertIn("Define Functional Requirements", res["final_response"])


    # 31. Session 14 Test A — CRM-044 Exact Score Calculation
    def test_31_session14_test_a_crm044_score_calculation(self):
        crm_044_ticket = {
            "ticket_id": "CRM-044",
            "name": "Create Product Roadmap",
            "description": "Establish the high-level product roadmap covering requirements, development, QA, deployment, Agentic RAG integration and release.",
            "status": "To Do",
            "priority": "Critical",
            "sprint": "CRM Retirement Fund - 2026 - SS02",
            "story_points": 8.0,
            "sub_ticket_ids": ["CRM-013", "CRM-014"],
            "release_note": "Product roadmap v1.0"
        }
        crm_evidence = [
            Evidence(
                source="CRM", document_id="SRS_part2", chunk_id="c1",
                source_path="CRM/SRS_part2.docx",
                content="Requirements for Retirement Fund Management CRM system ensure reliability and SRS compliance.",
                relevance_score=0.4825, evidence_type="BUSINESS"
            ).to_dict()
        ]
        # No Tech_Team evidence
        res = PriorityRuleEvaluator.evaluate_ticket(
            ticket=crm_044_ticket,
            all_tickets=[crm_044_ticket],
            crm_evidence=crm_evidence,
            tech_evidence=[]
        )
        self.assertEqual(res.ticket_id, "CRM-044")
        self.assertEqual(res.current_priority, "CRITICAL")
        self.assertEqual(res.recommended_priority, "CRITICAL")
        self.assertEqual(res.weighted_score, 0.7290)
        self.assertEqual(res.factors["DEPENDENCY"].score, 1.0)
        self.assertEqual(res.factors["BUSINESS"].score, 0.6825)
        self.assertEqual(res.factors["SCHEDULE"].score, 0.50)
        self.assertEqual(res.factors["RELEASE"].score, 0.85)
        self.assertEqual(res.factors["TECHNICAL"].score, 0.40)
        self.assertEqual(res.factors["CURRENT_PRIORITY"].score, 1.00)
        self.assertEqual(res.factors["EFFORT"].score, 0.50)

    # 32. Session 14 Test B — Top 5 Ranking Order
    def test_32_session14_test_b_top_5_ranking_order(self):
        tickets = [
            {"ticket_id": "CRM-044", "name": "Create Product Roadmap", "description": "High-level product roadmap covering release.", "priority": "Critical", "sprint": "CRM Retirement Fund - 2026 - SS02", "story_points": 8.0, "sub_ticket_ids": ["CRM-013", "CRM-014"], "release_note": "v1.0", "status": "To Do"},
            {"ticket_id": "CRM-049", "name": "Create SRS Baseline", "description": "SRS baseline for CRM release.", "priority": "Critical", "sprint": "CRM Retirement Fund - 2026 - SS02", "story_points": 11.0, "sub_ticket_ids": ["CRM-050", "CRM-051", "CRM-052"], "release_note": "v1.0", "status": "To Do"},
            {"ticket_id": "CRM-057", "name": "Define API Specification", "description": "Define API requirements for CRM release.", "priority": "High", "sprint": "CRM Retirement Fund - 2026 - SS02", "story_points": 11.0, "sub_ticket_ids": ["CRM-026", "CRM-027"], "release_note": "v1.0", "status": "To Do"},
            {"ticket_id": "CRM-045", "name": "Define MVP Scope", "description": "Identify MVP capabilities for production-ready CRM release.", "priority": "Critical", "sprint": "CRM Retirement Fund - 2026 - SS02", "story_points": 3.0, "sub_ticket_ids": ["—"], "release_note": "MVP scope", "status": "To Do"},
            {"ticket_id": "CRM-046", "name": "Define Release Milestones", "description": "Define major release milestones.", "priority": "High", "sprint": "CRM Retirement Fund - 2026 - SS02", "story_points": 3.0, "sub_ticket_ids": ["—"], "release_note": "Release milestones", "status": "To Do"},
        ]
        crm_evidence = [
            Evidence(
                source="CRM", document_id="SRS_part2", chunk_id="c1",
                source_path="CRM/SRS_part2.docx",
                content="SRS core requirement for CRM system.",
                relevance_score=0.4825, evidence_type="BUSINESS"
            ).to_dict()
        ]
        ranked = PriorityRanker.rank_tickets(tickets, crm_evidence=crm_evidence, tech_evidence=[])
        ranked_ids = [r.ticket_id for r in ranked]
        self.assertEqual(ranked_ids, ["CRM-044", "CRM-049", "CRM-057", "CRM-045", "CRM-046"])

    # 33. Session 14 Test C — Textual Priority Mapping
    def test_33_session14_test_c_textual_priority_mapping(self):
        expected_mappings = {
            "CRITICAL": 1.0,
            "URGENT": 1.0,
            "HIGH": 0.8,
            "MEDIUM": 0.5,
            "LOW": 0.2,
            "P0": 1.0,
            "P1": 0.8,
            "P2": 0.5,
            "P3": 0.2,
        }
        for prio_str, expected_score in expected_mappings.items():
            factor = PriorityRuleEvaluator._eval_current_priority(prio_str)
            self.assertEqual(factor.score, expected_score, f"Failed for priority '{prio_str}'")

    # 34. Session 14 Test D — Priority Preservation Under Evaluation
    def test_34_session14_test_d_priority_preservation(self):
        orig_ticket = {"ticket_id": "CRM-044", "priority": "Critical", "name": "Test", "status": "To Do"}
        res = PriorityRuleEvaluator.evaluate_ticket(orig_ticket, [orig_ticket], [], [])
        self.assertEqual(orig_ticket["priority"], "Critical")
        self.assertEqual(res.current_priority, "CRITICAL")
        self.assertEqual(res.recommended_priority, "MEDIUM")  # Low evidence baseline (< 0.45 score)

    # 35. Session 14 Test E — Deterministic Ranking
    def test_35_session14_test_e_deterministic_ranking(self):
        tickets = [
            {"ticket_id": "CRM-044", "name": "Create Product Roadmap", "priority": "Critical", "status": "To Do", "sub_ticket_ids": ["C1", "C2"]},
            {"ticket_id": "CRM-049", "name": "Create SRS Baseline", "priority": "Critical", "status": "To Do", "sub_ticket_ids": ["C1", "C2", "C3"]},
            {"ticket_id": "CRM-057", "name": "Define API Specification", "priority": "High", "status": "To Do", "sub_ticket_ids": ["C1", "C2"]}
        ]
        results = [
            [r.ticket_id for r in PriorityRanker.rank_tickets(tickets, [], [])]
            for _ in range(10)
        ]
        first_res = results[0]
        for r in results[1:]:
            self.assertEqual(r, first_res)

    # 36. Session 14 Test F — Explainability Grounding
    def test_36_session14_test_f_explainability_evidence_support(self):
        crm_044 = {
            "ticket_id": "CRM-044",
            "name": "Create Product Roadmap",
            "description": "Roadmap for release v1.0",
            "status": "To Do",
            "priority": "Critical",
            "sub_ticket_ids": ["CRM-013", "CRM-014"]
        }
        res = PriorityRuleEvaluator.evaluate_ticket(crm_044, [crm_044], [], [])
        self.assertIn("Critical blocker for 2 downstream tasks", res.summary_reason)
        self.assertIn("Ticket directly impacts release milestone", res.summary_reason)


if __name__ == "__main__":
    unittest.main()
