import unittest
from unittest.mock import MagicMock
from src.loaders.document import Document
from src.summarization.service import SummarizationService
from src.summarization.map_summarizer import MapSummarizer

class TestMapSummarizer(unittest.TestCase):
    def setUp(self):
        # Create a mock LLMClient so tests run fast without needing an active local LLM server
        self.mock_llm_client = MagicMock()
        self.mock_llm_client.generate_response.return_value = "This is a mock summary of the chunk."
        
        self.service = SummarizationService(llm_client=self.mock_llm_client)
        self.map_summarizer = MapSummarizer(summarization_service=self.service)

    def test_single_chunk_summarization(self):
        """Test summarizing 1 chunk."""
        chunk = Document(
            page_content="System architecture overview of SmartLogi microservices.",
            metadata={
                "source": "arch_doc.pdf",
                "filename": "arch_doc.pdf",
                "section": "System Architecture",
                "chunk_id": "arch_doc.pdf_chunk_1"
            }
        )

        result = self.map_summarizer.summarize_chunk(chunk)

        self.assertIsNotNone(result)
        self.assertEqual(result["chunk_id"], "arch_doc.pdf_chunk_1")
        self.assertEqual(result["summary"], "This is a mock summary of the chunk.")
        self.assertEqual(result["metadata"]["source"], "arch_doc.pdf")
        self.assertEqual(result["metadata"]["section"], "System Architecture")
        self.mock_llm_client.generate_response.assert_called_once()

    def test_multiple_chunks_summarization(self):
        """Test summarizing multiple chunks concurrently."""
        chunks = [
            Document(
                page_content=f"Chunk content number {i}.",
                metadata={
                    "source": f"doc_{i}.pdf",
                    "filename": f"doc_{i}.pdf",
                    "section": f"Section {i}",
                    "chunk_id": f"doc_{i}.pdf_chunk_{i}"
                }
            )
            for i in range(1, 4)
        ]

        results = self.map_summarizer.summarize_chunks(chunks, max_workers=2)

        self.assertEqual(len(results), 3)
        for i, res in enumerate(results, start=1):
            self.assertEqual(res["chunk_id"], f"doc_{i}.pdf_chunk_{i}")
            self.assertEqual(res["metadata"]["source"], f"doc_{i}.pdf")
            self.assertEqual(res["metadata"]["section"], f"Section {i}")
            self.assertEqual(res["summary"], "This is a mock summary of the chunk.")

    def test_chunk_metadata_preservation(self):
        """Test that all metadata fields (source, section, chunk_id, custom fields) are preserved intact."""
        metadata = {
            "source": "requirement_spec.docx",
            "filename": "requirement_spec.docx",
            "document_type": "docx",
            "category": "Requirement",
            "title": "Requirement Specification",
            "section": "Functional Requirements",
            "chunk_index": 2,
            "total_chunks": 10,
            "chunk_id": "req_spec_chunk_3",
            "custom_tag": "high_priority"
        }
        chunk = Document(
            page_content="Functional requirements detail user authentication and RBAC permissions.",
            metadata=metadata
        )

        result = self.map_summarizer.summarize_chunk(chunk)

        self.assertEqual(result["chunk_id"], "req_spec_chunk_3")
        self.assertEqual(result["metadata"]["source"], "requirement_spec.docx")
        self.assertEqual(result["metadata"]["section"], "Functional Requirements")
        self.assertEqual(result["metadata"]["document_type"], "docx")
        self.assertEqual(result["metadata"]["custom_tag"], "high_priority")
        self.assertEqual(result["metadata"], metadata)

    def test_empty_or_invalid_chunk(self):
        """Test behavior when given empty or invalid chunk objects."""
        # 1. Empty string chunk
        empty_chunk = Document(page_content="", metadata={"source": "empty.txt", "chunk_id": "empty_1"})
        res_empty = self.map_summarizer.summarize_chunk(empty_chunk)
        self.assertEqual(res_empty["summary"], "")
        self.assertEqual(res_empty["chunk_id"], "empty_1")

        # 2. Whitespace-only chunk
        space_chunk = Document(page_content="   \n\t  ", metadata={"source": "blank.txt", "chunk_id": "blank_1"})
        res_space = self.map_summarizer.summarize_chunk(space_chunk)
        self.assertEqual(res_space["summary"], "")

        # 3. None object handling
        res_none = self.map_summarizer.summarize_chunk(None)
        self.assertEqual(res_none["summary"], "")
        self.assertEqual(res_none["chunk_id"], "invalid_chunk")

        # 4. List with mix of valid, empty, and None chunks
        mixed_chunks = [
            Document(page_content="Valid text", metadata={"chunk_id": "valid_1"}),
            Document(page_content="", metadata={"chunk_id": "empty_1"}),
            None
        ]
        results = self.map_summarizer.summarize_chunks(mixed_chunks)
        self.assertEqual(len(results), 2)  # Valid non-None chunks processed
        self.assertEqual(results[0]["summary"], "This is a mock summary of the chunk.")
        self.assertEqual(results[1]["summary"], "")

if __name__ == "__main__":
    unittest.main()
