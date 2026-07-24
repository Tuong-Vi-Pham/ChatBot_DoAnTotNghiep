import os
import sys

# Add project root to path to run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.llm.client import LLMClient
from src.intent_clarification.clarifier import IntentClarifier

def test_query(clarifier: IntentClarifier, query: str):
    print(f"\nUser Query: '{query}'")
    print("Analyzing for ambiguity...")
    
    result = clarifier.check_ambiguity(query)
    
    if result["is_ambiguous"]:
        print("   -> Status: [AMBIGUOUS] (Pausing RAG pipeline)")
        print("   -> Generated Clarification Options:")
        for idx, option in enumerate(result["options"]):
            print(f"      {idx + 1}. {option}")
    else:
        print("   -> Status: [CLEAR] (Proceeding to RAG retrieval directly)")

def main():
    print("=========================================")
    print("PHASE 8 VERIFICATION: INTENT CLARIFICATION")
    print("=========================================")
    
    # 1. Initialize components
    print("Initializing components...")
    llm_client = LLMClient()
    
    if not llm_client.health_check():
        print("[FAIL] LM Studio is offline. Please start the local server before running this verification script.")
        return
        
    clarifier = IntentClarifier(llm_client)
    
    # 2. Test Case A: Clear Query
    # Should proceed immediately without ambiguity trigger
    test_query(clarifier, "What is the total budget for the project?")
    
    # 3. Test Case B: Ambiguous Heuristic Query (very short)
    # Should trigger ambiguity and options expansion
    test_query(clarifier, "onboarding")
    
    # 4. Test Case C: Ambiguous Multi-context Query
    # Should trigger ambiguity and options expansion
    test_query(clarifier, "search order feature")
    
    print("=========================================")
    print("PHASE 8 VERIFICATION COMPLETE")
    print("=========================================")

if __name__ == "__main__":
    main()
