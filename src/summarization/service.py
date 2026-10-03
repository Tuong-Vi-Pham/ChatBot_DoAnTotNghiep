from typing import Optional
from src.llm.client import LLMClient

class SummarizationService:
    """
    Summarization Service wrapping the LLM Client.
    Provides concise, factual chunk-level text summarization.
    """
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client if llm_client is not None else LLMClient()

    def summarize_text(
        self, 
        text: str, 
        max_tokens: int = 250, 
        temperature: float = 0.2
    ) -> str:
        """
        Summarizes the given text content concisely.
        Returns an empty string if text is empty or whitespace-only.
        """
        if not text or not text.strip():
            return ""

        system_prompt = (
            "You are a technical document summarizer.\n"
            "Summarize the provided document chunk concisely, accurately, and professionally.\n"
            "Rules:\n"
            "1. Focus only on key facts, requirements, specifications, or main points.\n"
            "2. Do NOT add external information or speculate.\n"
            "3. Keep the summary direct, structured, and easy to read."
        )

        user_prompt = f"Text to summarize:\n{text.strip()}\n\nSummary:"

        try:
            summary = self.llm_client.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return summary.strip()
        except Exception as e:
            print(f"[SummarizationService Warning] Failed to summarize text: {e}")
            return text.strip()[:200]  # Fallback to truncated text if LLM call fails
