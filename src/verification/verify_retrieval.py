import os
import sys

# Add project root to path to run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.embeddings.embedder import Embedder
from src.vectordb.database import VectorDBManager
from src.retrieval.hybrid import HybridRetriever

def test_query(retriever: HybridRetriever, query: str, faq_threshold: float = 0.85):
    print(f"\nQuery: '{query}'")
    print(f"Retrieving (threshold={faq_threshold})...")
    
    res = retriever.retrieve(query, faq_threshold=faq_threshold, top_k=3)
    source = res["retrieved_from"]
    print(f"  [OK] Retrieved from: {source.upper()}")
    
    for i, item in enumerate(res["results"]):
        print(f"  Match {i+1}:")
        print(f"    Similarity Score: {item['score']:.4f}")
        print(f"    Source File:      {item['metadata'].get('source')}")
        print(f"    Category:         {item['metadata'].get('category')}")
        # Print first 150 chars of content
        content_preview = item['content'].replace('\n', ' ')[:150]
        print(f"    Content Preview:  {content_preview}...")

def main():
    base_dir = "c:/Users/hp/OneDrive/Uit/HK2_2025_2026/DoAnTotNghiep/faq-chatbot-tech-team"
    db_path = os.path.join(base_dir, "chroma_db")
    
    print("=========================================")
    print("PHASE 5 VERIFICATION: HYBRID RETRIEVAL")
    print("=========================================")
    
    # Initialize components (expecting DB is already built in Phase 4)
    embedder = Embedder()
    db_manager = VectorDBManager(db_path=db_path, embedder=embedder)
    retriever = HybridRetriever(db_manager=db_manager, embedder=embedder)
    
    # 1. Test FAQ-First Direct Hit
    # Exact FAQ question matches get a similarity around 0.75, so we use threshold=0.70 to test the FAQ path
    test_query(retriever, "What is the total budget for the project?", faq_threshold=0.70)
    
    # 2. Test Document Fallback Hit
    # General queries do not match FAQs with high score, so they fall back to document chunks
    test_query(retriever, "Summarize the routing optimization strategies of SmartLogi.", faq_threshold=0.70)
    
    print("=========================================")

if __name__ == "__main__":
    main()
