import os
import sys

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch  # Critical: Import torch first to prevent shm.dll load conflicts with PaddlePaddle
from src.embeddings.embedder import Embedder
from src.vectordb.database import VectorDBManager
from src.retrieval.hybrid import HybridRetriever
from src.llm.client import LLMClient
from src.pipeline.rag_pipeline import RAGPipeline
from src.intent_clarification.clarifier import IntentClarifier

def main():
    print("=============================================================")
    print("      FAQ CHATBOT FOR SOFTWARE TEAMS (TERMINAL CLI)          ")
    print("=============================================================")
    print("Initializing components. Please wait...")
    
    # 1. Initialize components
    embedder = Embedder()
    db_manager = VectorDBManager(embedder=embedder)
    retriever = HybridRetriever(db_manager=db_manager, embedder=embedder)
    
    print("Connecting to LM Studio local server...")
    llm_client = LLMClient()
    if not llm_client.health_check():
        print("[FAIL] LM Studio is offline! Please start LM Studio on http://localhost:1234")
        sys.exit(1)
        
    pipeline = RAGPipeline(retriever=retriever, llm_client=llm_client)
    clarifier = IntentClarifier(llm_client=llm_client)
    
    print("\nSystem ready! Type your question below.")
    print("Type 'exit' or 'quit' to terminate the session.\n")
    
    while True:
        try:
            query = input("User > ").strip()
            if not query:
                continue
                
            if query.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
                
            # 2. Check Ambiguity
            print("Analyzing query...")
            clarify_result = clarifier.check_ambiguity(query)
            
            target_query = query
            if clarify_result["is_ambiguous"]:
                print("\n[Intent Clarification] Your query is ambiguous. Did you mean:")
                options = clarify_result["options"]
                for idx, opt in enumerate(options):
                    print(f"  {idx + 1}. {opt}")
                print("  4. Cancel and rephrase query")
                
                while True:
                    choice = input("\nSelect an option (1-4) > ").strip()
                    if choice in {"1", "2", "3"}:
                        target_query = options[int(choice) - 1]
                        print(f"Selected: '{target_query}'")
                        break
                    elif choice == "4":
                        print("Clarification cancelled.")
                        target_query = None
                        break
                    else:
                        print("Invalid choice. Please enter 1, 2, 3, or 4.")
                        
            if target_query is None:
                continue
                
            # 3. Run RAG Pipeline
            print("Retrieving and generating answer...")
            response = pipeline.run(
                query=target_query,
                faq_threshold=0.70,
                top_k=3,
                temperature=0.2
            )
            
            # 4. Display final response
            print(f"\nChatbot > {response['answer']}")
            print("\n-------------------------------------------------------------")
            print(f"Retrieval Source: {response['retrieved_from'].upper()}")
            print("Sources referenced:")
            if not response["sources"]:
                print("  No sources referenced.")
            else:
                for idx, src in enumerate(response["sources"]):
                    src_type = src.get("type", "document")
                    if src_type == "faq":
                        print(f"  [{idx+1}] FAQ: {src.get('source')} | Category: {src.get('category')} | Score: {src.get('score'):.4f}")
                    else:
                        print(f"  [{idx+1}] File: {src.get('source')} | Category: {src.get('category')} | Chunk: {src.get('chunk_index')+1}/{src.get('total_chunks')} | Score: {src.get('score'):.4f}")
            print("=============================================================\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}\n")

if __name__ == "__main__":
    main()
