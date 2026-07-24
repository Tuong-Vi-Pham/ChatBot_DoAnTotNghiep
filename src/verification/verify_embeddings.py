import os
import sys

# Add project root to path to run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.embeddings.embedder import Embedder

def test_model(model_name: str):
    print(f"\nTesting embedding model: {model_name}...")
    try:
        embedder = Embedder(model_name)
        dimension = embedder.get_dimension()
        print(f"  [OK] Model loaded. Expected dimension: {dimension}")
        
        sample_texts = [
            "What is the project budget?",
            "How do we onboard new developers?",
            "The testing phase begins next week."
        ]
        
        # Test batch embedding
        embeddings = embedder.embed_texts(sample_texts)
        print(f"  [OK] Embedded {len(embeddings)} texts successfully.")
        
        # Test dimensions
        for i, emb in enumerate(embeddings):
            if len(emb) == dimension:
                print(f"    - Text {i} embedding matches dimension: {len(emb)}")
            else:
                print(f"    - [FAIL] Text {i} dimension mismatch! Expected {dimension}, got {len(emb)}")
                
        # Test single query embedding
        query_emb = embedder.embed_query("Query search")
        if len(query_emb) == dimension:
            print(f"  [OK] Query embedding matches dimension: {len(query_emb)}")
        else:
            print(f"  [FAIL] Query embedding dimension mismatch! Expected {dimension}, got {len(query_emb)}")
            
    except Exception as e:
        print(f"  [FAIL] Error testing model {model_name}: {e}")

def main():
    print("=========================================")
    print("PHASE 3 VERIFICATION: EMBEDDINGS")
    print("=========================================")
    
    # 1. Test Preferred Model
    test_model("sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Test Alternative Model
    test_model("BAAI/bge-small-en-v1.5")
    
    print("=========================================")

if __name__ == "__main__":
    main()
