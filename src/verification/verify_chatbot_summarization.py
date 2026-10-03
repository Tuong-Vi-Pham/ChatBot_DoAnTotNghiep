import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from unittest.mock import MagicMock
from src.loaders.document import Document
from src.pipeline.rag_pipeline import RAGPipeline

def main():
    print("=========================================")
    print("PHASE 4 VERIFICATION: CHATBOT SUMMARIZATION INTEGRATION")
    print("=========================================")

    # 1. Setup mock components
    mock_retriever = MagicMock()
    mock_llm_client = MagicMock()
    mock_validator = MagicMock()
    mock_validator.check_adequacy.return_value = True
    mock_validator.verify_grounding.return_value = True
    mock_validator.default_insufficient_msg = (
        "I could not find sufficient information in the knowledge base. "
        "Please rephrase your question or provide more context."
    )

    mock_map_summarizer = MagicMock()
    mock_reduce_summarizer = MagicMock()

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm_client=mock_llm_client,
        validator=mock_validator,
        map_summarizer=mock_map_summarizer,
        reduce_summarizer=mock_reduce_summarizer
    )

    # 2. Test FAQ Query
    print("1. Testing FAQ Query Flow...")
    faq_query = "What is the project budget?"
    mock_retriever.retrieve.return_value = {
        "query": faq_query,
        "internal_query": faq_query,
        "retrieved_from": "faq",
        "results": [{
            "content": "Question: What is the project budget?\nAnswer: The total budget is $500,000.",
            "metadata": {"source": "Dataset_QandA.xlsx", "category": "Budget", "source_type": "faq"},
            "score": 0.95
        }]
    }
    mock_llm_client.generate_response.return_value = "The total project budget is $500,000."

    res_faq = pipeline.run(faq_query)
    print(f"   Intent / Retrieval: {res_faq['retrieved_from']}")
    if res_faq['retrieved_from'] == 'faq':
        print("   [OK] FAQ flow unaffected and hit correctly.")
    else:
        print("   [FAIL] FAQ flow broken!")

    # 3. Test Standard Knowledge Search Query
    print("\n2. Testing Standard Q&A Query Flow...")
    qa_query = "How does onboarding work?"
    mock_retriever.retrieve.return_value = {
        "query": qa_query,
        "internal_query": qa_query,
        "retrieved_from": "document",
        "results": [{
            "content": "Onboarding requires registering team members in SmartLogi portal.",
            "metadata": {"source": "onboarding.pdf", "section": "Process", "chunk_id": "onb_1"},
            "score": 0.82
        }]
    }
    mock_llm_client.generate_response.return_value = "Onboarding requires registering team members in the SmartLogi portal."

    res_qa = pipeline.run(qa_query)
    print(f"   Retrieval Source: {res_qa['retrieved_from']}")
    if res_qa['retrieved_from'] == 'document':
        print("   [OK] Standard Q&A flow unaffected and executed correctly.")
    else:
        print("   [FAIL] Standard Q&A flow broken!")

    # 4. Test Summarization Query
    print("\n3. Testing Summarization Query Flow...")
    sum_query = "Tóm tắt quy trình onboarding"
    mock_retriever.retrieve.return_value = {
        "query": sum_query,
        "internal_query": sum_query,
        "retrieved_from": "document",
        "results": [{
            "content": "Onboarding details and step by step setup.",
            "metadata": {"source": "onboarding.pdf", "section": "Steps", "chunk_id": "onb_2"},
            "score": 0.89
        }]
    }
    mock_map_summarizer.summarize_chunks.return_value = [{"chunk_id": "onb_2", "summary": "Setup onboarding steps."}]
    mock_reduce_summarizer.reduce_summaries.return_value = "Final Summary: Comprehensive overview of onboarding setup."

    res_sum = pipeline.run(sum_query)
    print(f"   Intent: {res_sum.get('intent')}")
    print(f"   Retrieval Source: {res_sum['retrieved_from']}")
    if res_sum.get('intent') == 'Summarization' and res_sum['retrieved_from'] == 'summarization':
        print("   [OK] Summarization intent recognized and Map-Reduce executed successfully.")
        print(f"   Answer: {res_sum['answer'][:100]}...")
    else:
        print("   [FAIL] Summarization flow failed!")

    # 5. Test Insufficient Information Summarization Query
    print("\n4. Testing Summarization Query with Insufficient Data...")
    empty_sum_query = "Tóm tắt module Quantum AI"
    mock_retriever.retrieve.return_value = {
        "query": empty_sum_query,
        "internal_query": empty_sum_query,
        "retrieved_from": "document",
        "results": []
    }
    mock_validator.check_adequacy.return_value = False

    res_empty = pipeline.run(empty_sum_query)
    if mock_validator.default_insufficient_msg in res_empty['answer']:
        print("   [OK] Correctly returned insufficient information message without hallucination.")
    else:
        print("   [FAIL] Did not return insufficient information message!")

    print("=========================================")

if __name__ == "__main__":
    main()
