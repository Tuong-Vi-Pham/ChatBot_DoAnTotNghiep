import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from unittest.mock import MagicMock
from src.loaders.document_loader import DocumentLoader
from src.chunking.splitter import DocumentSplitter
from src.summarization.service import SummarizationService
from src.summarization.map_summarizer import MapSummarizer

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    docs_dir = os.path.join(base_dir, "data/documents/Tech_Team")

    print("=========================================")
    print("PHASE 2 VERIFICATION: MAP SUMMARIZATION")
    print("=========================================")

    # 1. Load documents using existing document loader
    print("1. Loading source documents using DocumentLoader...")
    doc_loader = DocumentLoader(show_ocr_log=False)
    documents = doc_loader.load_directory(docs_dir)
    print(f"   [OK] Loaded {len(documents)} source documents.")

    # 2. Chunk documents using existing DocumentSplitter (unmodified)
    print("\n2. Chunking documents using existing DocumentSplitter...")
    splitter = DocumentSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents, print_stats=False)
    print(f"   [OK] Generated {len(chunks)} document chunks.")

    # 3. Initialize MapSummarizer with mock service (for fast offline verification)
    print("\n3. Initializing MapSummarizer and summarizing chunks...")
    mock_llm_client = MagicMock()
    mock_llm_client.generate_response.side_effect = lambda system_prompt, user_prompt, **kwargs: (
        f"Summary: {user_prompt.splitlines()[1][:60]}..."
    )

    service = SummarizationService(llm_client=mock_llm_client)
    map_summarizer = MapSummarizer(summarization_service=service)

    # Process first 5 chunks as a sample batch
    sample_chunks = chunks[:5] if len(chunks) >= 5 else chunks
    map_summaries = map_summarizer.summarize_chunks(sample_chunks, max_workers=2)

    # 4. Verify assertions
    print("\n4. Verifying Map Summarization Constraints:")
    
    # Check count
    if len(map_summaries) == len(sample_chunks):
        print(f"   [OK] Processed {len(map_summaries)} chunk summaries matching input count.")
    else:
        print(f"   [FAIL] Count mismatch! Expected {len(sample_chunks)}, got {len(map_summaries)}")

    # Check structure & metadata retention
    structure_ok = True
    for item in map_summaries:
        if "chunk_id" not in item or "metadata" not in item or "summary" not in item:
            structure_ok = False
            break
        meta = item["metadata"]
        if "source" not in meta or "filename" not in meta or "section" not in meta:
            structure_ok = False
            break

    if structure_ok:
        print("   [OK] Output structure matches specification: chunk_id, metadata (source, filename, section), summary.")
    else:
        print("   [FAIL] Output structure or metadata missing required fields!")

    # Preview output logic
    print("\nSample Output Item Structure:")
    if map_summaries:
        sample_item = map_summaries[0]
        print(f"chunk_id: {sample_item['chunk_id']}")
        print(f"metadata: {sample_item['metadata']}")
        print(f"summary:  {sample_item['summary']}")

    print("=========================================")

if __name__ == "__main__":
    main()
