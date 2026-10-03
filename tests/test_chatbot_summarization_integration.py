import unittest
from unittest.mock import MagicMock
from src.loaders.document import Document
from src.pipeline.rag_pipeline import RAGPipeline

class TestChatbotSummarizationIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_llm_client = MagicMock()
        self.mock_validator = MagicMock()

        # Default validator adequacy mock
        self.mock_validator.check_adequacy.return_value = True
        self.mock_validator.verify_grounding.return_value = True
        self.mock_validator.default_insufficient_msg = (
            "I could not find sufficient information in the knowledge base. "
            "Please rephrase your question or provide more context."
        )

        self.mock_map_summarizer = MagicMock()
        self.mock_reduce_summarizer = MagicMock()

        self.pipeline = RAGPipeline(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm_client,
            validator=self.mock_validator,
            map_summarizer=self.mock_map_summarizer,
            reduce_summarizer=self.mock_reduce_summarizer
        )

    def test_faq_intent_unaffected(self):
        """1. Test that FAQ pipeline operates completely unaffected."""
        faq_query = "What is the project budget?"
        self.mock_retriever.retrieve.return_value = {
            "query": faq_query,
            "internal_query": faq_query,
            "retrieved_from": "faq",
            "results": [{
                "content": "Question: What is the project budget?\nAnswer: The total budget is $500,000.",
                "metadata": {"source": "Dataset_QandA.xlsx", "category": "Budget", "source_type": "faq"},
                "score": 0.95
            }]
        }
        self.mock_llm_client.generate_response.return_value = "The total project budget is $500,000."

        response = self.pipeline.run(faq_query)

        self.assertEqual(response["retrieved_from"], "faq")
        self.assertIn("The total project budget is $500,000.", response["answer"])
        self.assertIn("References & Citations:", response["answer"])
        self.assertIn("Dataset_QandA.xlsx", response["answer"])
        # Map/Reduce summarizers should NOT be called for FAQ
        self.mock_map_summarizer.summarize_chunks.assert_not_called()
        self.mock_reduce_summarizer.reduce_summaries.assert_not_called()

    def test_standard_qa_unaffected(self):
        """2. Test that standard Knowledge Search Q&A operates completely unaffected."""
        qa_query = "Explain the system architecture"
        self.mock_retriever.retrieve.return_value = {
            "query": qa_query,
            "internal_query": qa_query,
            "retrieved_from": "document",
            "results": [{
                "content": "SmartLogi system uses FastAPI for backend services.",
                "metadata": {"source": "arch.pdf", "section": "Backend", "chunk_id": "arch.pdf_chunk_1"},
                "score": 0.82
            }]
        }
        self.mock_llm_client.generate_response.return_value = "SmartLogi utilizes FastAPI for backend microservices."

        response = self.pipeline.run(qa_query)

        self.assertEqual(response["retrieved_from"], "document")
        self.assertIn("FastAPI", response["answer"])
        # Map/Reduce summarizers should NOT be called for standard Q&A
        self.mock_map_summarizer.summarize_chunks.assert_not_called()
        self.mock_reduce_summarizer.reduce_summaries.assert_not_called()

    def test_summarization_intent_success(self):
        """3. Test that Summarization intent triggers Map-Reduce Summarization pipeline successfully."""
        sum_query = "Tóm tắt module Customer Management"
        self.mock_retriever.retrieve.return_value = {
            "query": sum_query,
            "internal_query": sum_query,
            "retrieved_from": "document",
            "results": [
                {
                    "content": "Customer Management handles user onboarding, profiles, and permissions.",
                    "metadata": {"source": "cust_mgmt.docx", "section": "Overview", "chunk_id": "cust_1"},
                    "score": 0.88
                },
                {
                    "content": "Includes OAuth2 authentication and role-based access control.",
                    "metadata": {"source": "cust_mgmt.docx", "section": "Security", "chunk_id": "cust_2"},
                    "score": 0.85
                }
            ]
        }
        self.mock_map_summarizer.summarize_chunks.return_value = [
            {"chunk_id": "cust_1", "summary": "Handles user onboarding and profiles."},
            {"chunk_id": "cust_2", "summary": "Includes OAuth2 and RBAC."}
        ]
        self.mock_reduce_summarizer.reduce_summaries.return_value = (
            "Final Summary: The Customer Management module manages user onboarding, profiles, and RBAC security."
        )

        response = self.pipeline.run(sum_query)

        self.assertEqual(response["intent"], "Summarization")
        self.assertEqual(response["retrieved_from"], "summarization")
        self.assertIn("Customer Management module manages user onboarding", response["answer"])
        self.assertIn("References & Citations:", response["answer"])
        
        # Verify Map & Reduce pipeline components were invoked
        self.mock_map_summarizer.summarize_chunks.assert_called_once()
        self.mock_reduce_summarizer.reduce_summaries.assert_called_once()

    def test_summarization_insufficient_data(self):
        """4. Test Summarization intent when insufficient evidence is found in the knowledge base."""
        sum_query = "Tóm tắt module Blockchain Payment"
        self.mock_retriever.retrieve.return_value = {
            "query": sum_query,
            "internal_query": sum_query,
            "retrieved_from": "document",
            "results": []
        }
        self.mock_validator.check_adequacy.return_value = False

        response = self.pipeline.run(sum_query)

        self.assertEqual(response["intent"], "Summarization")
        self.assertEqual(response["retrieved_from"], "summarization")
        self.assertEqual(response["answer"], self.mock_validator.default_insufficient_msg)
        
        # Map/Reduce should NOT be called when data is insufficient
        self.mock_map_summarizer.summarize_chunks.assert_not_called()
        self.mock_reduce_summarizer.reduce_summaries.assert_not_called()

if __name__ == "__main__":
    unittest.main()
