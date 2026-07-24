from typing import Dict, Any, List
from src.retrieval.hybrid import HybridRetriever
from src.llm.client import LLMClient
from src.verification.validator import VerificationLoop

class RAGPipeline:
    """
    RAG Pipeline orchestrator that connects retrieval and LLM generation.
    Formats the context from hybrid retrieval and structures prompts for LM Studio.
    """
    def __init__(self, retriever: HybridRetriever, llm_client: LLMClient, validator: VerificationLoop = None):
        self.retriever = retriever
        self.llm_client = llm_client
        self.validator = validator if validator is not None else VerificationLoop(llm_client)

    def run(
        self, 
        query: str, 
        faq_threshold: float = 0.70, 
        top_k: int = 4,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Executes the RAG flow: Retrieve -> Prompt Construct -> LLM -> Response.
        """
        # Check if the query is a simple greeting or chitchat
        import re
        normalized_query = re.sub(r'[^\w\s]', '', query).strip().lower()
        greetings = {
            "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", 
            "howdy", "yo", "hi there", "hello there", "xin chào", "xin chao", "chào", "chao", 
            "chào bạn", "chao ban", "chào ad", "chao ad", "alo", "halo", "hê lô", "he lo",
            "thanks", "thank you", "cảm ơn", "cam on", "cám ơn", "how are you", "hows it going"
        }
        if normalized_query in greetings:
            return {
                "query": query,
                "answer": self.validator.default_insufficient_msg,
                "retrieved_from": "none",
                "sources": []
            }

        # 1. Retrieve context
        retrieval_res = self.retriever.retrieve(query, faq_threshold=faq_threshold, top_k=top_k)
        source_type = retrieval_res["retrieved_from"]
        results = retrieval_res["results"]
        
        # 2. Handle empty retrieval or inadequate evidence
        is_adequate = True
        if source_type == "document":
            is_adequate = self.validator.check_adequacy(results, score_threshold=0.40)
            
        if not results or not is_adequate:
            return {
                "query": query,
                "answer": self.validator.default_insufficient_msg,
                "retrieved_from": "none",
                "sources": []
            }
            
        # 3. Construct prompt based on retrieval source
        context_str = ""
        sources_list = []
        
        if source_type == "faq":
            # Direct FAQ hit: context is the matched QA pair
            faq_item = results[0]
            context_str = faq_item["content"]
            sources_list.append({
                "source": faq_item["metadata"].get("source"),
                "category": faq_item["metadata"].get("category"),
                "score": faq_item["score"],
                "type": "faq"
            })
            
            system_prompt = (
                "You are a helpful project team support assistant. "
                "An exact matching FAQ has been found. Use it to answer the user's question. "
                "Keep the answer aligned with the FAQ answer. Do not extrapolate."
            )
            user_prompt = f"FAQ Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
            
        else:
            # Document chunks fallback
            context_parts = []
            for idx, item in enumerate(results):
                src = item["metadata"].get("source")
                cat = item["metadata"].get("category")
                c_idx = item["metadata"].get("chunk_index", 0)
                tot_c = item["metadata"].get("total_chunks", 1)
                
                context_parts.append(
                    f"--- Context {idx+1} [Source: {src} | Category: {cat} | Chunk: {c_idx+1}/{tot_c}] ---\n"
                    f"{item['content']}"
                )
                sources_list.append({
                    "source": src,
                    "category": cat,
                    "score": item["score"],
                    "chunk_index": c_idx,
                    "total_chunks": tot_c,
                    "type": "document"
                })
                
            context_str = "\n\n".join(context_parts)
            
            system_prompt = (
                "You are an expert technical support assistant for a software team. "
                "Answer the user's question using ONLY the provided context. "
                "If the context does not contain the answer, say exactly: "
                "\"I could not find sufficient information in the knowledge base. Please rephrase your question or provide more context.\"\n"
                "Do not make up facts or extrapolate beyond the provided text."
            )
            
            user_prompt = (
                f"Context from project documents:\n{context_str}\n\n"
                f"User Question: {query}\n"
                f"Answer:"
            )

        # 4. Generate response using LLM
        answer = self.llm_client.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature
        )
        
        # 5. Post-inference grounding audit
        if source_type == "document":
            is_grounded = self.validator.verify_grounding(context_str, answer)
            if not is_grounded:
                answer = self.validator.default_insufficient_msg
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_from": source_type,
            "sources": sources_list
        }
