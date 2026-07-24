import os
import sys

# Add project root to path to run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.embeddings.embedder import Embedder
from src.vectordb.database import VectorDBManager
from src.retrieval.hybrid import HybridRetriever
from src.llm.client import LLMClient
from src.pipeline.rag_pipeline import RAGPipeline

def run_test_case(pipeline: RAGPipeline, query: str, faq_threshold: float = 0.70):
    print(f"\nUser Query: '{query}'")
    print("--------------------------------------------------")
    
    try:
        response = pipeline.run(
            query=query,
            faq_threshold=faq_threshold,
            top_k=3,
            temperature=0.2
        )
        
        print(f"Retrieval Source: {response['retrieved_from'].upper()}")
        print(f"Final Answer    : {response['answer']}")
        print(f"Sources Referenced:")
        for idx, src in enumerate(response["sources"]):
            src_type = src.get("type", "document")
            if src_type == "faq":
                print(f"  [{idx+1}] FAQ Source: {src.get('source')} | Category: {src.get('category')} | Score: {src.get('score'):.4f}")
            else:
                print(f"  [{idx+1}] Doc Source: {src.get('source')} | Category: {src.get('category')} | Chunk: {src.get('chunk_index')+1}/{src.get('total_chunks')} | Score: {src.get('score'):.4f}")
    except Exception as e:
        print(f"[ERROR] RAG Pipeline run failed: {e}")
    print("--------------------------------------------------")

def main():
    base_dir = "c:/Users/hp/OneDrive/Uit/HK2_2025_2026/DoAnTotNghiep/faq-chatbot-tech-team"
    db_path = os.path.join(base_dir, "chroma_db")
    
    print("=========================================")
    print("PHASE 7 VERIFICATION: RAG PIPELINE")
    print("=========================================")
    
    # 1. Initialize all subcomponents
    print("Initializing RAG subcomponents...")
    embedder = Embedder()
    db_manager = VectorDBManager(db_path=db_path, embedder=embedder)
    retriever = HybridRetriever(db_manager=db_manager, embedder=embedder)
    llm_client = LLMClient()
    
    # Health Check
    if not llm_client.health_check():
        print("[FAIL] LM Studio is offline. Please start the local server before running this verification script.")
        return
    else:
        print("[OK] LM Studio local server is online.")
        
    pipeline = RAGPipeline(retriever=retriever, llm_client=llm_client)
    
    # 2. Test Case A: FAQ Match (threshold=0.70)
    print("\nExecuting Test Case A: FAQ Direct Retrieval...")
    run_test_case(pipeline, "What is the total budget for the project?", faq_threshold=0.70)
    
    # 3. Test Case B: Document fallback and generation
    print("\nExecuting Test Case B: Document Fallback & Generation...")
    run_test_case(pipeline, "What is SmartLogi and what is its goal?", faq_threshold=0.70)
    
    print("\n=========================================")
    print("PHASE 7 VERIFICATION COMPLETE")
    print("=========================================")

if __name__ == "__main__":
    main()
