import os
import sys

# Add project root to path to run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.loaders.faq_loader import FAQLoader
from src.loaders.document_loader import DocumentLoader
from src.chunking.splitter import DocumentSplitter

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    faq_path = os.path.join(base_dir, "data/questions/Dataset_QandA.xlsx")
    docs_dir = os.path.join(base_dir, "data/documents/Tech_Team")
    
    print("=========================================")
    print("PHASE 2 VERIFICATION: PREPROCESSING & CHUNKING")
    print("=========================================")
    
    # 1. Load Data
    print("Loading data...")
    faq_loader = FAQLoader(faq_path)
    doc_loader = DocumentLoader(show_ocr_log=False)
    
    try:
        faq_docs = faq_loader.load()
        documents = doc_loader.load_directory(docs_dir)
        all_docs = faq_docs + documents
        print(f"[OK] Loaded {len(faq_docs)} FAQs and {len(documents)} source documents.")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return
        
    # 2. Split / Chunk
    print("\nSplitting documents (size=500, overlap=50)...")
    splitter = DocumentSplitter(chunk_size=500, chunk_overlap=50)
    
    try:
        chunks = splitter.split_documents(all_docs, print_stats=True)
        
        # 3. Perform assertions
        faq_chunks = [c for c in chunks if c.metadata.get("source_type") == "faq"]
        doc_chunks = [c for c in chunks if c.metadata.get("source_type") == "document"]
        
        print("\nVerifying Constraints:")
        
        # Check FAQ count matches loaded FAQs
        if len(faq_chunks) == len(faq_docs):
            print(f"  [OK] FAQ Count match: {len(faq_chunks)} (Curated FAQs were kept intact).")
        else:
            print(f"  [FAIL] Curated FAQs changed count! Expected {len(faq_docs)}, got {len(faq_chunks)}")
            
        # Check that empty chunks are removed
        empty_chunks = [c for c in chunks if not c.page_content.strip()]
        if not empty_chunks:
            print("  [OK] No empty chunks found.")
        else:
            print(f"  [FAIL] Found {len(empty_chunks)} empty chunks!")
            
        # Check metadata preservation
        meta_ok = True
        for c in chunks:
            if "source" not in c.metadata or "source_type" not in c.metadata:
                meta_ok = False
                break
        if meta_ok:
            print("  [OK] Document metadata preserved in all chunks.")
        else:
            print("  [FAIL] Some chunks are missing critical metadata.")
            
        # Check that document chunks have index
        doc_index_ok = all("chunk_index" in c.metadata for c in doc_chunks)
        if doc_index_ok:
            print("  [OK] All document chunks have chunk_index in metadata.")
        else:
            print("  [FAIL] Some document chunks do not have chunk_index.")
            
        # Sample Document Chunk preview
        if doc_chunks:
            print("\nSample Document Chunk Preview:")
            sample = doc_chunks[0]
            print(f"Source: {sample.metadata.get('source')} | Chunk Index: {sample.metadata.get('chunk_index')}/{sample.metadata.get('total_chunks')}")
            print(f"Content:\n{sample.page_content}")
            
    except Exception as e:
        print(f"[ERROR] Failed to process splitting: {e}")

if __name__ == "__main__":
    main()
