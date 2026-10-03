import unittest
from unittest.mock import MagicMock, patch

from src.intent_clarification.clarifier import IntentClarifier, reconstruct_query
from src.pipeline.rag_pipeline import RAGPipeline


class IntentClarificationGroundedTests(unittest.TestCase):
    def test_reconstruct_query_single_and_multi_options(self):
        """Verify query reconstruction preserves original_query and combines selected options."""
        # Single option selection
        q1 = reconstruct_query("bug", ["Defect reporting"])
        self.assertEqual(q1, "bug - Defect reporting")

        # Multi-option selection
        q2 = reconstruct_query("bug", ["Defect reporting", "Bug tracking documentation"])
        self.assertEqual(q2, "bug - Defect reporting; Bug tracking documentation")

        # Original query combined with detailed option
        q3 = reconstruct_query("onboarding process", ["onboarding process for new devs"])
        self.assertEqual(q3, "onboarding process - onboarding process for new devs")

    def test_candidate_verification_filters_unsupported_candidate(self):
        """Verify unsupported candidate with low/no retrieval score is rejected during verification."""
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "1. Supported option\n2. Unsupported hallucinated topic"

        mock_retriever = MagicMock()

        def mock_retrieve(query, top_k=3):
            if "Supported" in query:
                return {
                    "retrieved_from": "document",
                    "results": [{"content": "Doc context", "score": 0.82, "retrieval_score": 0.82, "metadata": {}}]
                }
            else: # Unsupported candidate
                return {
                    "retrieved_from": "document",
                    "results": [{"content": "Unrelated", "score": 0.10, "retrieval_score": 0.10, "metadata": {}}]
                }

        mock_retriever.retrieve.side_effect = mock_retrieve

        clarifier = IntentClarifier(llm_client=mock_llm, retriever=mock_retriever)
        res = clarifier.check_ambiguity("bug", retriever=mock_retriever)

        self.assertTrue(res["is_ambiguous"])
        self.assertEqual(len(res["options"]), 1)
        self.assertEqual(res["options"][0], "Supported option")

    def test_only_one_or_two_valid_intents(self):
        """Verify that when only 1 or 2 options are valid, exactly 1 or 2 options are returned without inventing fake 3rd options."""
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "1. First valid option\n2. Second valid option"

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = {
            "retrieved_from": "document",
            "results": [{"content": "Valid doc", "score": 0.75, "retrieval_score": 0.75, "metadata": {}}]
        }

        clarifier = IntentClarifier(llm_client=mock_llm, retriever=mock_retriever)
        res = clarifier.check_ambiguity("deployment", retriever=mock_retriever)

        self.assertTrue(res["is_ambiguous"])
        self.assertEqual(len(res["options"]), 2)
        self.assertIn("First valid option", res["options"])
        self.assertIn("Second valid option", res["options"])

    def test_ambiguous_query_individual_options_execution(self):
        """Verify executing individual options 1, 2, and 3 produces reconstructed queries and successful RAG answers."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = {
            "retrieved_from": "document",
            "results": [{"content": "Matching document chunk", "score": 0.85, "retrieval_score": 0.85, "metadata": {"source": "guide.pdf"}}]
        }

        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "Answer supported by document context."

        pipeline = RAGPipeline(retriever=mock_retriever, llm_client=mock_llm)
        pipeline.validator.verify_grounding = MagicMock(return_value=True)

        for opt in ["Option 1 text", "Option 2 text", "Option 3 text"]:
            reconstructed = reconstruct_query("bug", [opt])
            res = pipeline.run(query=reconstructed, chat_id="chat_test_123")

            self.assertIn("Answer supported by document context", res["answer"])
            self.assertEqual(res["retrieved_from"], "document")
            self.assertEqual(res["sources"][0]["source"], "guide.pdf")

    def test_ambiguous_query_select_all_options_multi_query(self):
        """Verify selecting all options combines all selected intents and retrieves merged context."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = lambda q, **kw: {
            "retrieved_from": "document",
            "results": [{"content": f"Context for query {q}", "score": 0.80, "retrieval_score": 0.80, "metadata": {"source": f"doc_{q[:5]}.pdf"}}]
        }

        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "Combined summary covering all selected options."

        pipeline = RAGPipeline(retriever=mock_retriever, llm_client=mock_llm)
        pipeline.validator.verify_grounding = MagicMock(return_value=True)

        all_options = ["Option 1", "Option 2", "Option 3"]
        reconstructed = reconstruct_query("bug", all_options)
        res = pipeline.run(query=reconstructed, chat_id="chat_multi_123")

        self.assertIn("Combined summary", res["answer"])
        self.assertEqual(len(res["sources"]), 3)  # Merged sources from all sub-queries

    def test_terminology_different_from_user_wording_retrieves_correct_docs(self):
        """Verify option using different terminology than user's original query (e.g. 'Defect tracking procedure' vs 'bug') retrieves correctly."""
        orig_q = "bug"
        diff_term_opt = "Defect tracking procedure and resolution workflow"
        reconstructed = reconstruct_query(orig_q, [diff_term_opt])

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = {
            "retrieved_from": "document",
            "results": [{"content": "Defect tracking workflow documentation details", "score": 0.88, "retrieval_score": 0.88, "metadata": {"source": "defect_mgmt.docx"}}]
        }

        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "Defect resolution workflow answer."

        pipeline = RAGPipeline(retriever=mock_retriever, llm_client=mock_llm)
        pipeline.validator.verify_grounding = MagicMock(return_value=True)
        res = pipeline.run(query=reconstructed)

        self.assertIn("Defect resolution workflow answer", res["answer"])
        self.assertEqual(res["sources"][0]["source"], "defect_mgmt.docx")


if __name__ == "__main__":
    unittest.main()
