import unittest
from unittest.mock import MagicMock
from src.summarization.service import SummarizationService
from src.summarization.reducer import ReduceSummarizer

class TestReduceSummarizer(unittest.TestCase):
    def setUp(self):
        self.mock_llm_client = MagicMock()
        self.service = SummarizationService(llm_client=self.mock_llm_client)

    def test_reduce_few_summaries(self):
        """Test reducing a small list of summaries (fits within context limit)."""
        self.mock_llm_client.generate_response.return_value = "Final consolidated summary of project charter and architecture."

        summaries = [
            {"chunk_id": "c1", "summary": "Project charter outlines scope and budget."},
            {"chunk_id": "c2", "summary": "Architecture uses FastAPI and ChromaDB."}
        ]

        reducer = ReduceSummarizer(summarization_service=self.service, max_chars_per_batch=4000)
        final_summary = reducer.reduce_summaries(summaries)

        self.assertEqual(final_summary, "Final consolidated summary of project charter and architecture.")
        self.mock_llm_client.generate_response.assert_called_once()
        # Verify Reduce System Prompt was used
        args, kwargs = self.mock_llm_client.generate_response.call_args
        self.assertEqual(kwargs["system_prompt"], ReduceSummarizer.REDUCE_SYSTEM_PROMPT)

    def test_reduce_many_summaries_with_batching(self):
        """Test reducing a large list of summaries that triggers batching and intermediate summaries."""
        def mock_generate_response(system_prompt, user_prompt, **kwargs):
            if system_prompt == ReduceSummarizer.INTERMEDIATE_SYSTEM_PROMPT:
                return "Intermediate summary of batch."
            return "Final reduced summary across all batches."

        self.mock_llm_client.generate_response.side_effect = mock_generate_response

        # Generate 10 summaries
        summaries = [f"Chunk summary number {i} containing important specs." for i in range(10)]

        # Set max_chars_per_batch low to force batching into multiple groups
        reducer = ReduceSummarizer(summarization_service=self.service, max_chars_per_batch=150)
        final_summary = reducer.reduce_summaries(summaries)

        self.assertEqual(final_summary, "Final reduced summary across all batches.")
        # Called multiple times for batches + final reduce
        self.assertGreater(self.mock_llm_client.generate_response.call_count, 1)

    def test_reduce_exceeding_context_limit(self):
        """Test multi-stage intermediate batch reduction when total content significantly exceeds context limit."""
        # Setup side effects for multi-pass hierarchical batch reduction
        self.mock_llm_client.generate_response.side_effect = lambda system_prompt, user_prompt, **kwargs: (
            f"InterSummary: {user_prompt[:30]}..."
        )

        large_summaries = [
            f"Detailed section {i} summary with critical metrics {i * 100} and timeline 2026."
            for i in range(15)
        ]

        # Low batch size limit forces multi-batch intermediate steps
        reducer = ReduceSummarizer(summarization_service=self.service, max_chars_per_batch=200)
        final_summary = reducer.reduce_summaries(large_summaries)

        self.assertTrue(final_summary.startswith("InterSummary:"))
        self.assertGreater(self.mock_llm_client.generate_response.call_count, 2)

    def test_reduce_duplicate_information(self):
        """Test reducing summaries that contain duplicate or redundant information."""
        self.mock_llm_client.generate_response.return_value = "SmartLogi system uses BAAI/bge-m3 embeddings and ChromaDB."

        duplicate_summaries = [
            {"chunk_id": "c1", "summary": "SmartLogi system uses BAAI/bge-m3 embeddings."},
            {"chunk_id": "c2", "summary": "SmartLogi system uses BAAI/bge-m3 embeddings."},
            {"chunk_id": "c3", "summary": "Database engine is ChromaDB vector DB."},
            {"chunk_id": "c4", "summary": "Database engine is ChromaDB vector DB."}
        ]

        reducer = ReduceSummarizer(summarization_service=self.service, max_chars_per_batch=4000)
        final_summary = reducer.reduce_summaries(duplicate_summaries)

        self.assertEqual(final_summary, "SmartLogi system uses BAAI/bge-m3 embeddings and ChromaDB.")
        # Ensure all summaries were passed to prompt for deduplication
        args, kwargs = self.mock_llm_client.generate_response.call_args
        user_prompt = kwargs["user_prompt"]
        self.assertIn("BAAI/bge-m3", user_prompt)
        self.assertIn("ChromaDB", user_prompt)

    def test_empty_or_invalid_summaries(self):
        """Test reducing empty or invalid summaries lists."""
        reducer = ReduceSummarizer(summarization_service=self.service)
        
        self.assertEqual(reducer.reduce_summaries([]), "")
        self.assertEqual(reducer.reduce_summaries(["", "   "]), "")
        self.assertEqual(reducer.reduce_summaries([{"summary": ""}, {"summary": "  "}]), "")

if __name__ == "__main__":
    unittest.main()
