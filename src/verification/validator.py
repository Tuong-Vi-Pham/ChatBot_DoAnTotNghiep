from typing import List, Dict, Any
from src.llm.client import LLMClient

class VerificationLoop:
    """
    Validation layer to reduce LLM hallucinations.
    Implements a pre-inference score check and a post-inference LLM grounding audit.
    """
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.default_insufficient_msg = (
            "I could not find sufficient information in the knowledge base. "
            "Please rephrase your question or provide more context."
        )

    def check_adequacy(
        self, 
        results: List[Dict[str, Any]], 
        score_threshold: float = 0.40,
        reranker_threshold: Optional[float] = None
    ) -> bool:
        """
        Pre-inference check: verifies if any retrieved chunk meets the minimum similarity score.
        Explicitly uses vector retrieval_score for similarity threshold validation (e.g. 0.40),
        preventing raw reranker CrossEncoder logits from being miscompared against cosine threshold.
        """
        if not results:
            return False
            
        # Check if the highest vector retrieval score meets the similarity threshold
        max_retrieval_score = max(item.get("retrieval_score", item.get("score", 0.0)) for item in results)
        if max_retrieval_score < score_threshold:
            return False

        # Optional reranker logit threshold validation if specified
        if reranker_threshold is not None:
            max_rerank_score = max(
                (item["rerank_score"] for item in results if item.get("rerank_score") is not None),
                default=None
            )
            if max_rerank_score is not None and max_rerank_score < reranker_threshold:
                return False

        return True

    def verify_grounding(self, context: str, answer: str) -> bool:
        """
        Post-inference check: audits the generated answer using the LLM to detect if
        any external facts or hallucinations are present.
        """
        # If the model already returned the default insufficient response, skip audit
        if self.default_insufficient_msg.lower() in answer.lower():
            return True
            
        system_prompt = (
            "You are a strict factual validator for a Retrieval-Augmented Generation system.\n"
            "Analyze the given Context and the proposed Answer.\n"
            "Your sole task is to check if the proposed Answer is fully supported by the Context.\n\n"
            "Guidelines:\n"
            "- The Answer must not introduce any new facts, figures, metrics, names, or timelines "
            "not explicitly stated in the Context.\n"
            "- If the Answer contains any ungrounded assertions or extrapolations, output exactly:\n"
            "GROUNDED: NO\n\n"
            "- If the Answer is 100% supported by the Context, output exactly:\n"
            "GROUNDED: YES\n\n"
            "Do not include any reasoning, summaries, or other text. Just output 'GROUNDED: YES' or 'GROUNDED: NO'."
        )
        
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Proposed Answer:\n{answer}"
        )
        
        try:
            response = self.llm_client.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0  # Zero temperature for deterministic classification
            )
            
            if "GROUNDED: YES" in response:
                return True
            else:
                print(f"[Verification Loop Warning] Grounding check failed! Model response: {response.strip()}")
                return False
                
        except Exception as e:
            # If evaluation client fails, we default to True to avoid blocking,
            # but log the failure.
            print(f"[Verification Loop Warning] Grounding audit failed to execute: {e}")
            return True
