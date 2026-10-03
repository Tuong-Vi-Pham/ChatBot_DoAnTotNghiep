import os
import sys

# Add project root to path to run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.loaders.faq_loader import FAQLoader
from src.loaders.document_loader import DocumentLoader

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    faq_path = os.path.join(base_dir, "data/questions/Dataset_QandA.xlsx")
    docs_dir = os.path.join(base_dir, "data/documents/Tech_Team")
    
    print("=========================================")
    print("PHASE 1 VERIFICATION: DATA LOADING")
    print("=========================================")
    
    # 1. Load FAQs
    print(f"Loading FAQs from: {faq_path}")
    faq_loader = FAQLoader(faq_path)
    try:
        faq_docs = faq_loader.load()
        print(f"[OK] Successfully loaded {len(faq_docs)} FAQs.")
        if faq_docs:
            print("\nSample FAQ Document:")
            print(f"Content:\n{faq_docs[0].page_content}")
            print(f"Metadata: {faq_docs[0].metadata}")
    except Exception as e:
        print(f"[ERROR] Error loading FAQs: {e}")
        faq_docs = []

    print("\n-----------------------------------------")
    
    # 2. Load Documents
    print(f"Loading documents from: {docs_dir}")
    doc_loader = DocumentLoader(show_ocr_log=False)
    try:
        documents = doc_loader.load_directory(docs_dir)
        print(f"[OK] Successfully loaded {len(documents)} documents.")
        
        # Count document types
        ext_counts = {}
        for doc in documents:
            source = doc.metadata.get("source", "")
            ext = os.path.splitext(source)[1].lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            
        print("\nLoaded Document Types Breakdown:")
        for ext, count in ext_counts.items():
            print(f"  {ext}: {count} files")
            
        if documents:
            print("\nSample Document (first 300 chars):")
            sample_doc = documents[0]
            print(f"Source: {sample_doc.metadata.get('source')}")
            print(f"Category (Subfolder): {sample_doc.metadata.get('category')}")
            print(f"Content Preview:\n{sample_doc.page_content[:300]}...")
    except Exception as e:
        print(f"[ERROR] Error loading documents: {e}")
        documents = []

    print("=========================================")
    print(f"Total parsed entities: {len(faq_docs)} FAQs + {len(documents)} Documents")
    print("=========================================")

if __name__ == "__main__":
    main()
