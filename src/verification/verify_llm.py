import os
import sys

# Add project root to path to run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.llm.client import LLMClient

def main():
    print("=========================================")
    print("PHASE 6 VERIFICATION: LM STUDIO INTEGRATION")
    print("=========================================")
    
    # Initialize client
    print("1. Initializing LLM Client...")
    client = LLMClient()
    print(f"   Configured Endpoint: {client.base_url}")
    print(f"   Configured Model   : {client.default_model}")
    
    # 1. Run Health Check
    print("\n2. Running Health Check...")
    is_healthy = client.health_check()
    if is_healthy:
        print("   [OK] LM Studio server is ONLINE and responding to queries.")
    else:
        print("   [FAIL] LM Studio server is OFFLINE or unreachable.")
        print("          Please verify that LM Studio is running on http://localhost:1234")
        return

    # 2. Run Connectivity Test
    test_prompt = "Say hello."
    print(f"\n3. Running Connectivity Test...")
    print(f"   Input Query: '{test_prompt}'")
    print("   Sending request to local LLM, waiting for response...")
    
    result = client.simple_test(prompt=test_prompt)
    
    if result["success"]:
        print("\n   [OK] Connection successful! Response received:")
        print(f"   --------------------------------------------------")
        print(f"   {result['response']}")
        print(f"   --------------------------------------------------")
        print(f"   Response Latency: {result['latency_seconds']:.2f} seconds")
    else:
        print("\n   [FAIL] Connection failed!")
        print(f"   Error: {result['error']}")
        
    print("\n=========================================")
    print("VERIFICATION COMPLETED")
    print("=========================================")

if __name__ == "__main__":
    main()
