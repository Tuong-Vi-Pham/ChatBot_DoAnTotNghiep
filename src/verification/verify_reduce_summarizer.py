import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from unittest.mock import MagicMock
from src.loaders.document_loader import DocumentLoader
from src.chunking.splitter import DocumentSplitter
from src.summarization.service import SummarizationService
from src.summarization.map_summarizer import MapSummarizer
from src.summarization.reducer import ReduceSummarizer

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    docs_dir = os.path.join(base_dir, "data/documents/Tech_Team")

    print("=========================================")
    print("PHASE 3 VERIFICATION: REDUCE SUMMARIZATION")
    print("=========================================")

    # 1. Load documents
    print("1. Loading source documents...")
    doc_loader = DocumentLoader(show_ocr_log=False)
    documents = doc_loader.load_directory(docs_dir)
    print(f"   [OK] Loaded {len(documents)} documents.")

    # 2. Chunk documents
    print("\n2. Chunking documents...")
    splitter = DocumentSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents, print_stats=False)
    print(f"   [OK] Generated {len(chunks)} chunks.")

    # 3. Map Summarization
    print("\n3. Executing Map Summarization (Phase 2)...")
    mock_llm_client = MagicMock()
    mock_llm_client.generate_response.side_effect = lambda system_prompt, user_prompt, **kwargs: (
        "Consolidated Final Summary: SmartLogi is a comprehensive logistics software system with microservices architecture."
        if "Consolidate" in system_prompt or "synthesize" in system_prompt.lower()
        else f"ChunkSummary: {user_prompt.splitlines()[1][:50]}..."
    )

    service = SummarizationService(llm_client=mock_llm_client)
    map_summarizer = MapSummarizer(summarization_service=service)

    # Process first 8 chunks
    map_summaries = map_summarizer.summarize_chunks(chunks[:8], max_workers=2)
    print(f"   [OK] Generated {len(map_summaries)} chunk summaries.")

    # 4. Reduce Summarization
    print("\n4. Executing Reduce Summarization (Phase 3)...")
    reducer = ReduceSummarizer(summarization_service=service, max_chars_per_batch=400)
    final_summary = reducer.reduce_summaries(map_summaries)

    print("\n5. Verifying Reduce Constraints:")
    if final_summary:
        print("   [OK] Successfully generated Final Summary.")
        print(f"   [OK] Final Summary length: {len(final_summary)} characters.")
    else:
        print("   [FAIL] Final summary is empty!")

    print("\nFinal Summary Preview:")
    print(f"{final_summary}")
    print("=========================================")

if __name__ == "__main__":
    main()
