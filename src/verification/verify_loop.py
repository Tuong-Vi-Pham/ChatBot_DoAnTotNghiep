import os
import sys

# Add project root to path to run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.embeddings.embedder import Embedder
from src.vectordb.database import VectorDBManager
from src.retrieval.hybrid import HybridRetriever
from src.llm.client import LLMClient
from src.pipeline.rag_pipeline import RAGPipeline

def test_query(pipeline: RAGPipeline, query: str):
    print(f"\nUser Query: '{query}'")
    print("Running RAG Pipeline with Verification Loop...")
    
    try:
        response = pipeline.run(
            query=query,
            faq_threshold=0.70,
            top_k=3,
            temperature=0.2
        )
        
        print(f"   Retrieval Source: {response['retrieved_from'].upper()}")
        print(f"   Final Answer    : {response['answer']}")
        print(f"   Sources Count   : {len(response['sources'])}")
        for idx, src in enumerate(response["sources"]):
            print(f"     - [{idx+1}] File: {src.get('source')} | Score: {src.get('score'):.4f}")
            
    except Exception as e:
        print(f"   [ERROR] Run failed: {e}")
    print("-" * 50)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    db_path = os.path.join(base_dir, "chroma_db")
    
    print("=========================================")
    print("PHASE 9 VERIFICATION: VERIFICATION LOOP")
    print("=========================================")
    
    # Initialize components
    print("Initializing components...")
    embedder = Embedder()
    db_manager = VectorDBManager(db_path=db_path, embedder=embedder)
    retriever = HybridRetriever(db_manager=db_manager, embedder=embedder)
    llm_client = LLMClient()
    
    if not llm_client.health_check():
        print("[FAIL] LM Studio is offline. Please start local server.")
        return
        
    pipeline = RAGPipeline(retriever=retriever, llm_client=llm_client)
    
    # 1. Test In-scope query (Expected: SUCCESS with sources)
    print("\n1. Testing in-scope question (Grounding OK):")
    test_query(pipeline, "What is SmartLogi and what is its goal?")
    
    # 2. Test Out-of-scope query (Expected: INSUFFICIENT INFO)
    print("\n2. Testing out-of-scope question (Recipe):")
    test_query(pipeline, "What is the recipe for cooking spaghetti?")
    
    # 3. Test Out-of-scope query 2 (Expected: INSUFFICIENT INFO)
    print("\n3. Testing out-of-scope question (Admin password):")
    test_query(pipeline, "Can you give me the admin password for the system?")
    
    print("=========================================")
    print("PHASE 9 VERIFICATION COMPLETE")
    print("=========================================")

if __name__ == "__main__":
    main()
