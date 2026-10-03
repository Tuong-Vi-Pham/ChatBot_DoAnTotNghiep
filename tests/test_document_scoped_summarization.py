import unittest
from unittest.mock import MagicMock
from src.loaders.document import Document
from src.pipeline.rag_pipeline import RAGPipeline
from src.summarization.document_resolver import DocumentResolver


class TestDocumentScopedSummarization(unittest.TestCase):
    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_llm_client = MagicMock()
        self.mock_validator = MagicMock()
        self.mock_map_summarizer = MagicMock()
        self.mock_reduce_summarizer = MagicMock()

        self.mock_validator.check_adequacy.return_value = True
        self.mock_validator.verify_grounding.return_value = True
        self.mock_validator.default_insufficient_msg = (
            "I could not find sufficient information in the knowledge base. "
            "Please rephrase your question or provide more context."
        )

        self.known_docs = [
            "SRS_part1.docx",
            "SRS_part2.docx",
            "1_SRS_Module_User_v1.0.docx",
            "4_SRS_Module_User_v4.0.docx",
            "Product Roadmap.docx",
            "Business_Process_Flow.docx",
            "Business_Process_Flow.pdf"
        ]
        self.mock_retriever.get_indexed_documents.return_value = self.known_docs

        self.pipeline = RAGPipeline(
            retriever=self.mock_retriever,
            llm_client=self.mock_llm_client,
            validator=self.mock_validator,
            map_summarizer=self.mock_map_summarizer,
            reduce_summarizer=self.mock_reduce_summarizer
        )

    def test_scenario_1_full_document_summary_srs_part1(self):
        """
        Test 1 — Full document summary
        Query: 'Summarize all content in SRS_part1.docx (including definitions and references)'
        Expected:
        - target document is resolved as SRS_part1.docx
        - all chunks used for summarization belong to SRS_part1.docx
        - summary covers the complete document
        - Definitions are included if present
        - References are included if present
        - no SRS_part2.docx content is included
        - no other SRS version is included in the summary
        - citations contain only SRS_part1.docx.
        """
        query = "Summarize all content in SRS_part1.docx (including definitions and references)"

        # Prepare 15 mock chunks spanning the document from Introduction to References
        mock_chunks = [
            {"chunk_id": "chunk_0", "content": "1. Introduction to CRM system", "metadata": {"source": "SRS_part1.docx", "section": "Introduction", "chunk_index": 0, "total_chunks": 15}},
            {"chunk_id": "chunk_1", "content": "1.1 System Objectives", "metadata": {"source": "SRS_part1.docx", "section": "Objectives", "chunk_index": 1, "total_chunks": 15}},
            {"chunk_id": "chunk_2", "content": "2. Scope of the project", "metadata": {"source": "SRS_part1.docx", "section": "Scope", "chunk_index": 2, "total_chunks": 15}},
            {"chunk_id": "chunk_3", "content": "3. Definitions and Acronyms used throughout", "metadata": {"source": "SRS_part1.docx", "section": "3 Definitions", "chunk_index": 3, "total_chunks": 15}},
            {"chunk_id": "chunk_4", "content": "4. References to standards and external docs", "metadata": {"source": "SRS_part1.docx", "section": "4 References", "chunk_index": 4, "total_chunks": 15}},
            {"chunk_id": "chunk_5", "content": "5. Functional Requirements: User management", "metadata": {"source": "SRS_part1.docx", "section": "Functional Requirements", "chunk_index": 5, "total_chunks": 15}},
            {"chunk_id": "chunk_6", "content": "5.1 Contract management", "metadata": {"source": "SRS_part1.docx", "section": "Contract Management", "chunk_index": 6, "total_chunks": 15}},
            {"chunk_id": "chunk_7", "content": "6. Non-Functional Requirements: Performance", "metadata": {"source": "SRS_part1.docx", "section": "Non-Functional", "chunk_index": 7, "total_chunks": 15}},
            {"chunk_id": "chunk_8", "content": "7. Appendix and System Architecture", "metadata": {"source": "SRS_part1.docx", "section": "Appendix", "chunk_index": 8, "total_chunks": 15}},
        ]
        self.mock_retriever.get_document_chunks.return_value = mock_chunks

        def mock_summarize_chunks(chunks):
            # Capture that all chunks belong to SRS_part1.docx
            for c in chunks:
                self.assertEqual(c.metadata["source"], "SRS_part1.docx")
            return [{"chunk_id": c.metadata.get("chunk_id", f"c_{i}"), "summary": f"Summary of {c.metadata.get('section')}"} for i, c in enumerate(chunks)]

        self.mock_map_summarizer.summarize_chunks.side_effect = mock_summarize_chunks
        self.mock_reduce_summarizer.reduce_summaries.return_value = (
            "Complete summary of SRS_part1.docx covering Introduction, Scope, "
            "Definitions (Key terms), References (External standards), and Functional Requirements."
        )

        response = self.pipeline.run(query)

        # 1. Target document resolved
        self.assertEqual(response["retrieved_from"], "document_scoped")
        self.assertEqual(response["target_document"], "SRS_part1.docx")

        # 2. Check chunks passed to MapSummarizer
        call_args = self.mock_map_summarizer.summarize_chunks.call_args[0][0]
        self.assertTrue(len(call_args) > 0)
        sections_included = [c.metadata.get("section") for c in call_args]
        self.assertIn("3 Definitions", sections_included)
        self.assertIn("4 References", sections_included)

        # 3. Assert chunk ordering is strictly preserved (ascending chunk_index)
        indices = [c.metadata.get("chunk_index") for c in call_args]
        self.assertEqual(indices, sorted(indices))

        # 4. Assert answer contains expected summary
        self.assertIn("Complete summary of SRS_part1.docx", response["answer"])
        self.assertIn("Definitions", response["answer"])
        self.assertIn("References", response["answer"])

        # 5. Assert citations contain ONLY SRS_part1.docx
        self.assertIn("References & Citations:", response["answer"])
        self.assertIn("SRS_part1.docx", response["answer"])
        self.assertNotIn("SRS_part2.docx", response["answer"])
        self.assertNotIn("1_SRS_Module_User_v1.0.docx", response["answer"])

        for src in response["sources"]:
            self.assertEqual(src["source"], "SRS_part1.docx")

    def test_scenario_2_normal_rag_question(self):
        """
        Test 2 — Normal RAG question
        Query: 'What is the user authentication requirement?'
        Expected:
        - existing normal RAG behavior remains unchanged.
        """
        query = "What is the user authentication requirement?"

        self.mock_retriever.retrieve.return_value = {
            "query": query,
            "internal_query": query,
            "retrieved_from": "document",
            "results": [
                {
                    "content": "Users must authenticate via SSO and MFA.",
                    "metadata": {"source": "1_SRS_Module_User_v1.0.docx", "section": "Security", "chunk_id": "chunk_1"},
                    "score": 0.85
                }
            ]
        }
        self.mock_llm_client.generate_response.return_value = "The user authentication requirement specifies SSO with MFA."

        response = self.pipeline.run(query)

        # Standard RAG Q&A path
        self.assertEqual(response["retrieved_from"], "document")
        self.assertIn("SSO with MFA", response["answer"])
        self.assertIn("1_SRS_Module_User_v1.0.docx", response["answer"])
        # Map/Reduce should NOT be invoked for normal RAG Q&A
        self.mock_map_summarizer.summarize_chunks.assert_not_called()
        self.mock_reduce_summarizer.reduce_summaries.assert_not_called()

    def test_scenario_3_ambiguous_document(self):
        """
        Test 3 — Ambiguous document
        Query: 'Summarize SRS_part1' when multiple documents match
        Expected:
        - system asks for clarification
        - system does not arbitrarily select a document.
        """
        query = "Summarize SRS_part1"

        # Mock known documents where multiple documents match 'SRS_part1'
        ambiguous_known_docs = [
            "SRS_part1.docx",
            "SRS_part1.pdf",
            "SRS_part2.docx"
        ]
        self.mock_retriever.get_indexed_documents.return_value = ambiguous_known_docs

        response = self.pipeline.run(query)

        self.assertEqual(response["retrieved_from"], "clarification")
        self.assertIn("multiple documents matching", response["answer"].lower())
        self.assertIn("SRS_part1.docx", response["answer"])
        self.assertIn("SRS_part1.pdf", response["answer"])
        self.assertEqual(response["sources"], [])

        # Neither retriever.get_document_chunks nor MapSummarizer should be called
        self.mock_retriever.get_document_chunks.assert_not_called()
        self.mock_map_summarizer.summarize_chunks.assert_not_called()

    def test_scenario_4_another_document_srs_part2(self):
        """
        Test 4 — Another document
        Query: 'Summarize SRS_part2.docx'
        Expected:
        - only SRS_part2.docx is used.
        """
        query = "Summarize SRS_part2.docx"

        mock_chunks = [
            {"chunk_id": "chunk_p2_0", "content": "SRS Part 2: Reporting module specifications", "metadata": {"source": "SRS_part2.docx", "section": "Reports", "chunk_index": 0, "total_chunks": 2}},
            {"chunk_id": "chunk_p2_1", "content": "SRS Part 2: Export formats and batch processing", "metadata": {"source": "SRS_part2.docx", "section": "Batch Export", "chunk_index": 1, "total_chunks": 2}},
        ]
        self.mock_retriever.get_document_chunks.return_value = mock_chunks

        self.mock_map_summarizer.summarize_chunks.return_value = [
            {"chunk_id": "chunk_p2_0", "summary": "Reports module specifications."},
            {"chunk_id": "chunk_p2_1", "summary": "Export formats and batch processing."}
        ]
        self.mock_reduce_summarizer.reduce_summaries.return_value = (
            "SRS_part2.docx defines the Reporting module specifications and batch export capabilities."
        )

        response = self.pipeline.run(query)

        self.assertEqual(response["retrieved_from"], "document_scoped")
        self.assertEqual(response["target_document"], "SRS_part2.docx")
        self.assertIn("SRS_part2.docx defines the Reporting module", response["answer"])
        self.assertIn("SRS_part2.docx", response["answer"])
        self.assertNotIn("SRS_part1.docx", response["answer"])
        self.assertNotIn("1_SRS_Module_User_v1.0.docx", response["answer"])

        for src in response["sources"]:
            self.assertEqual(src["source"], "SRS_part2.docx")

    def test_scenario_5_no_document_specified(self):
        """
        Test 5 — No document specified
        Query: 'Summarize the requirements for authentication.'
        Expected:
        - do not incorrectly force a document-level filter
        - retain the intended normal retrieval behavior.
        """
        query = "Summarize the requirements for authentication."

        self.mock_retriever.retrieve.return_value = {
            "query": query,
            "internal_query": query,
            "retrieved_from": "document",
            "results": [
                {
                    "content": "Authentication requires bcrypt password hashing and 2FA.",
                    "metadata": {"source": "1_SRS_Module_User_v1.0.docx", "section": "Auth", "chunk_id": "auth_1"},
                    "score": 0.88
                },
                {
                    "content": "Role-based authorization verifies tokens against the RBAC table.",
                    "metadata": {"source": "2_SRS_Module_User_v2.0.docx", "section": "RBAC", "chunk_id": "auth_2"},
                    "score": 0.84
                }
            ]
        }
        self.mock_map_summarizer.summarize_chunks.return_value = [
            {"chunk_id": "auth_1", "summary": "Requires bcrypt and 2FA."},
            {"chunk_id": "auth_2", "summary": "Requires RBAC token checks."}
        ]
        self.mock_reduce_summarizer.reduce_summaries.return_value = (
            "Authentication requirements encompass bcrypt password hashing, 2FA, and RBAC token checks."
        )

        response = self.pipeline.run(query)

        # Standard topic-based summarization (not scoped to single document)
        self.assertEqual(response["intent"], "Summarization")
        self.assertEqual(response["retrieved_from"], "summarization")
        self.assertNotIn("target_document", response)
        self.assertIn("Authentication requirements encompass bcrypt", response["answer"])
        self.assertIn("1_SRS_Module_User_v1.0.docx", response["answer"])
        self.assertIn("2_SRS_Module_User_v2.0.docx", response["answer"])


if __name__ == "__main__":
    unittest.main()
