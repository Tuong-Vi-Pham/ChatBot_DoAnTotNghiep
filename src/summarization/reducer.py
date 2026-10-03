from typing import List, Dict, Any, Union, Optional
from src.summarization.service import SummarizationService

class ReduceSummarizer:
    """
    Reduce Summarizer component.
    Synthesizes multiple chunk summaries into a unified, non-redundant final summary.
    Supports batching and intermediate summaries when input size exceeds context limits.
    """
    REDUCE_SYSTEM_PROMPT = (
        "You are an expert technical editor and document synthesizer.\n"
        "Your task is to synthesize multiple individual document summaries into a single cohesive, consolidated summary.\n\n"
        "Strict Requirements:\n"
        "1. Combine related points and eliminate all duplicate or redundant information.\n"
        "2. Retain all key technical specifications, numbers, metrics, dates, decisions, and statuses.\n"
        "3. Do NOT add external facts, extrapolate, or fabricate information not present in the input summaries.\n"
        "4. Structure the output into a unified, fluent, and professional summary."
    )

    INTERMEDIATE_SYSTEM_PROMPT = (
        "You are a technical document synthesizer.\n"
        "Summarize and consolidate this batch of partial document summaries into a concise intermediate summary.\n"
        "Remove duplicates and retain all key facts, metrics, and technical details."
    )

    def __init__(
        self, 
        summarization_service: Optional[SummarizationService] = None,
        max_chars_per_batch: int = 4000
    ):
        self.service = summarization_service if summarization_service is not None else SummarizationService()
        self.max_chars_per_batch = max_chars_per_batch

    def _extract_summary_text(self, item: Union[str, Dict[str, Any]]) -> str:
        """Helper to extract text from string or Phase 2 dictionary output."""
        if isinstance(item, str):
            return item.strip()
        elif isinstance(item, dict):
            return str(item.get("summary") or item.get("text") or "").strip()
        return ""

    def reduce_summaries(
        self, 
        summaries: List[Union[str, Dict[str, Any]]],
        max_tokens: int = 500,
        temperature: float = 0.2
    ) -> str:
        """
        Synthesizes a list of summaries into a final consolidated summary.
        Handles batching and intermediate summaries if content exceeds max_chars_per_batch.
        """
        if not summaries:
            return ""

        # Extract and clean input text items
        clean_summaries = []
        for item in summaries:
            text = self._extract_summary_text(item)
            if text:
                clean_summaries.append(text)

        if not clean_summaries:
            return ""

        if len(clean_summaries) == 1:
            # Single summary input: return directly or run single-pass polish
            return clean_summaries[0]

        # Calculate total combined character length
        total_len = sum(len(s) for s in clean_summaries)

        # Case 1: Fits in single context batch -> Final Direct Reduce
        if total_len <= self.max_chars_per_batch:
            return self._final_reduce(clean_summaries, max_tokens=max_tokens, temperature=temperature)

        # Case 2: Exceeds batch limit -> Partition into batches, compute Intermediate Summaries
        batches = self._partition_into_batches(clean_summaries)
        intermediate_summaries = []

        for batch in batches:
            batch_text = "\n\n".join([f"- {s}" for s in batch])
            user_prompt = f"Batch summaries to synthesize:\n{batch_text}\n\nIntermediate Summary:"
            
            try:
                inter_summary = self.service.llm_client.generate_response(
                    system_prompt=self.INTERMEDIATE_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                if inter_summary.strip():
                    intermediate_summaries.append(inter_summary.strip())
            except Exception as e:
                print(f"[ReduceSummarizer Warning] Intermediate batch reduction failed: {e}")
                # Fallback: keep first summary of batch
                intermediate_summaries.append(batch[0])

        # Recursive reduce step on intermediate summaries
        return self.reduce_summaries(
            intermediate_summaries, 
            max_tokens=max_tokens, 
            temperature=temperature
        )

    def _partition_into_batches(self, summaries: List[str]) -> List[List[str]]:
        """Partitions summaries into batches where each batch size <= max_chars_per_batch."""
        batches = []
        current_batch = []
        current_length = 0

        for text in summaries:
            text_len = len(text)
            if current_batch and (current_length + text_len > self.max_chars_per_batch):
                batches.append(current_batch)
                current_batch = [text]
                current_length = text_len
            else:
                current_batch.append(text)
                current_length += text_len

        if current_batch:
            batches.append(current_batch)

        return batches

    def _final_reduce(self, summaries: List[str], max_tokens: int, temperature: float) -> str:
        """Executes the final single-pass reduce synthesis."""
        formatted_inputs = "\n\n".join([f"Summary [{i+1}]:\n{s}" for i, s in enumerate(summaries)])
        user_prompt = f"Input Summaries to Consolidate:\n{formatted_inputs}\n\nFinal Consolidated Summary:"

        try:
            final_summary = self.service.llm_client.generate_response(
                system_prompt=self.REDUCE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return final_summary.strip()
        except Exception as e:
            print(f"[ReduceSummarizer Warning] Final reduce failed: {e}")
            return "\n\n".join(summaries)  # Fallback to concatenated summaries
