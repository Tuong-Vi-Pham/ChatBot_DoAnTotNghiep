from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from src.loaders.document import Document
from src.summarization.service import SummarizationService

class MapSummarizer:
    """
    Map Summarizer component.
    Processes individual document chunks and produces structured summaries while preserving full lineage metadata.
    """
    def __init__(self, summarization_service: Optional[SummarizationService] = None):
        self.service = summarization_service if summarization_service is not None else SummarizationService()

    def summarize_chunk(self, chunk: Document) -> Dict[str, Any]:
        """
        Summarizes a single Document chunk and returns a structured result.
        Preserves all metadata including source, filename, section, chunk_id.
        """
        if chunk is None:
            return {
                "chunk_id": "invalid_chunk",
                "metadata": {},
                "summary": ""
            }

        metadata = getattr(chunk, "metadata", {}) or {}
        page_content = getattr(chunk, "page_content", "") or ""

        # Extract or generate chunk_id
        chunk_id = metadata.get("chunk_id")
        if not chunk_id:
            filename = metadata.get("filename") or metadata.get("source") or "doc"
            c_idx = metadata.get("chunk_index", 0)
            chunk_id = f"{filename}_chunk_{c_idx + 1}"

        # Summarize text content
        summary_text = self.service.summarize_text(page_content)

        return {
            "chunk_id": chunk_id,
            "metadata": dict(metadata),
            "summary": summary_text
        }

    def summarize_chunks(
        self, 
        chunks: List[Document], 
        max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Summarizes a list of Document chunks concurrently.
        Preserves original ordering of chunks.
        """
        if not chunks:
            return []

        # Filter or handle invalid items if any in list
        valid_chunks = [c for c in chunks if c is not None]
        if not valid_chunks:
            return []

        # If single chunk or max_workers <= 1, process sequentially
        if len(valid_chunks) == 1 or max_workers <= 1:
            return [self.summarize_chunk(c) for c in valid_chunks]

        # Process concurrently using ThreadPoolExecutor
        workers = min(max_workers, len(valid_chunks))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(self.summarize_chunk, valid_chunks))

        return results
