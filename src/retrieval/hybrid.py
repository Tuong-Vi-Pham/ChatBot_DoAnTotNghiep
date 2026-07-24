from typing import Dict, Any, List
from src.embeddings.embedder import Embedder
from src.vectordb.database import VectorDBManager

class HybridRetriever:
    """
    Implements FAQ-First Hybrid Retrieval.
    Queries the FAQ collection first. If the top result exceeds the similarity threshold,
    returns the FAQ. Otherwise, falls back to the document chunks collection.
    """
    def __init__(self, db_manager: VectorDBManager, embedder: Embedder):
        self.db_manager = db_manager
        self.embedder = embedder

    def retrieve(self, query: str, faq_threshold: float = 0.85, top_k: int = 4) -> Dict[str, Any]:
        # 1. Embed query
        query_vector = self.embedder.embed_query(query)
        
        # 2. Search FAQ Collection
        faq_results = self.db_manager.faq_collection.query(
            query_embeddings=[query_vector],
            n_results=1
        )
        
        # Check if we got a high similarity match in FAQs
        if faq_results and faq_results["documents"] and faq_results["documents"][0]:
            distance = faq_results["distances"][0][0]
            # Since collection hnsw:space is cosine, similarity = 1 - distance
            similarity = 1.0 - distance
            
            if similarity >= faq_threshold:
                return {
                    "query": query,
                    "retrieved_from": "faq",
                    "results": [{
                        "content": faq_results["documents"][0][0],
                        "metadata": faq_results["metadatas"][0][0],
                        "score": similarity
                    }]
                }
                
        # 3. Fallback to Document Collection
        doc_results = self.db_manager.document_collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        results = []
        if doc_results and doc_results["documents"] and doc_results["documents"][0]:
            for i in range(len(doc_results["documents"][0])):
                dist = doc_results["distances"][0][i]
                sim = 1.0 - dist
                results.append({
                    "content": doc_results["documents"][0][i],
                    "metadata": doc_results["metadatas"][0][i],
                    "score": sim
                })
                
        return {
            "query": query,
            "retrieved_from": "document",
            "results": results
        }
