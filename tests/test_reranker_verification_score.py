import unittest
from unittest.mock import MagicMock
from src.retrieval.reranker import BGEReranker
from src.verification.validator import VerificationLoop


class RerankerVerificationScoreTests(unittest.TestCase):
    def test_retrieval_score_preservation(self):
        """Verify that BGEReranker preserves original retrieval_score and score without overwriting them with raw logit scores."""
        reranker = BGEReranker.__new__(BGEReranker)
        reranker.model = MagicMock()
        # Mock model predict to return raw CrossEncoder logits (e.g. -2.5 and 1.8)
        reranker.model.predict.return_value = [-2.5, 1.8]

        documents = [
            {"content": "Doc A", "score": 0.85, "retrieval_score": 0.85},
            {"content": "Doc B", "score": 0.72, "retrieval_score": 0.72},
        ]

        reranked = reranker.rerank("test query", documents, top_n=2)

        # Doc B had higher logit (1.8), so it should be first
        self.assertEqual(reranked[0]["content"], "Doc B")
        self.assertEqual(reranked[0]["rerank_score"], 1.8)
        self.assertEqual(reranked[0]["retrieval_score"], 0.72)
        self.assertEqual(reranked[0]["score"], 0.72)  # Score preserved!

        # Doc A had lower logit (-2.5), so it should be second
        self.assertEqual(reranked[1]["content"], "Doc A")
        self.assertEqual(reranked[1]["rerank_score"], -2.5)
        self.assertEqual(reranked[1]["retrieval_score"], 0.85)
        self.assertEqual(reranked[1]["score"], 0.85)  # Vector similarity preserved!

    def test_relevant_document_with_low_raw_reranker_logit(self):
        """Verify that a relevant document with high vector similarity (0.85) but low raw reranker logit (-2.5) passes check_adequacy."""
        validator = VerificationLoop(llm_client=MagicMock())
        results = [
            {"content": "Relevant Chunk", "score": 0.85, "retrieval_score": 0.85, "rerank_score": -2.5}
        ]

        # Prior to the fix, -2.5 overwrote doc["score"], causing -2.5 < 0.40 check to return False.
        # Now, check_adequacy reads retrieval_score (0.85), so 0.85 >= 0.40 evaluates to True!
        is_adequate = validator.check_adequacy(results, score_threshold=0.40)
        self.assertTrue(is_adequate)

    def test_relevant_document_with_high_reranker_logit(self):
        """Verify that a document with high vector similarity (0.88) and high reranker logit (3.2) passes check_adequacy."""
        validator = VerificationLoop(llm_client=MagicMock())
        results = [
            {"content": "High score Chunk", "score": 0.88, "retrieval_score": 0.88, "rerank_score": 3.2}
        ]

        is_adequate = validator.check_adequacy(results, score_threshold=0.40)
        self.assertTrue(is_adequate)

    def test_irrelevant_document(self):
        """Verify that an irrelevant document with low vector similarity (0.15 < 0.40) fails check_adequacy."""
        validator = VerificationLoop(llm_client=MagicMock())
        results = [
            {"content": "Irrelevant Chunk", "score": 0.15, "retrieval_score": 0.15, "rerank_score": -5.0}
        ]

        is_adequate = validator.check_adequacy(results, score_threshold=0.40)
        self.assertFalse(is_adequate)

    def test_verification_behavior_with_explicit_reranker_threshold(self):
        """Verify verification logic when optional reranker_threshold logit check is specified."""
        validator = VerificationLoop(llm_client=MagicMock())
        results = [
            {"content": "Borderline Chunk", "score": 0.75, "retrieval_score": 0.75, "rerank_score": -10.0}
        ]

        # Passes vector similarity check (0.75 >= 0.40)
        self.assertTrue(validator.check_adequacy(results, score_threshold=0.40))

        # Fails explicit reranker logit check (-10.0 < -5.0)
        self.assertFalse(validator.check_adequacy(results, score_threshold=0.40, reranker_threshold=-5.0))


if __name__ == "__main__":
    unittest.main()
