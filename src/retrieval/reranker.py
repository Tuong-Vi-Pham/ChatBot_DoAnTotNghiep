import torch
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class BGEReranker:
    """
    Reranking stage using BAAI/bge-reranker-v2-m3.
    Re-scores candidate document chunks retrieved by vector search to maximize retrieval precision.
    Supports MPS (Apple Silicon), CUDA, and CPU backends.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None):
        self.model_name = model_name
        
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
                
        self.device = device
        print(f"[Reranker] Loading {self.model_name} on device '{self.device}'...")
        self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_n: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of retrieved candidate documents based on the query.
        Each item in `documents` must contain a 'content' key.
        """
        if not documents:
            return []
            
        if len(documents) <= 1:
            return documents[:top_n]

        # Prepare (query, doc_text) pairs
        pairs = [(query, doc["content"]) for doc in documents]
        
        # Predict cross-attention relevance scores
        scores = self.model.predict(pairs)
        
        # Attach rerank score and re-sort documents without overwriting vector similarity score
        reranked_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            retrieval_sim = doc.get("retrieval_score", doc.get("score", 0.0))
            doc_copy["retrieval_score"] = float(retrieval_sim)
            doc_copy["score"] = float(retrieval_sim)
            doc_copy["rerank_score"] = float(score)
            reranked_docs.append(doc_copy)

        # Sort descending by rerank score
        reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked_docs[:top_n]
