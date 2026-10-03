from typing import Dict, Any, List, Optional
from src.embeddings.embedder import Embedder
from src.vectordb.database import VectorDBManager
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.reranker import BGEReranker

class HybridRetriever:
    """
    Implements FAQ-First Hybrid Retrieval enhanced with:
    1. Internal Query Rewriting & Expansion.
    2. Candidate Oversampling & Content Deduplication.
    3. Cross-Encoder Reranking using BAAI/bge-reranker-v2-m3.
    """
    def __init__(
        self, 
        db_manager: VectorDBManager, 
        embedder: Embedder, 
        reranker: Optional[BGEReranker] = None,
        rewriter: Optional[QueryRewriter] = None
    ):
        self.db_manager = db_manager
        self.embedder = embedder
        self.rewriter = rewriter if rewriter is not None else QueryRewriter()
        self.reranker = reranker

    def _get_reranker(self) -> BGEReranker:
        if self.reranker is None:
            self.reranker = BGEReranker()
        return self.reranker

    def retrieve(self, query: str, faq_threshold: float = 0.85, top_k: int = 4) -> Dict[str, Any]:
        # 1. Internal Query Rewriting (invisible to user)
        internal_query = self.rewriter.rewrite(query)

        # 2. Embed internal query
        query_vector = self.embedder.embed_query(internal_query)
        
        # 3. Search FAQ Collection first
        faq_results = self.db_manager.faq_collection.query(
            query_embeddings=[query_vector],
            n_results=1
        )
        
        # Check if we got a high similarity match in FAQs
        if faq_results and faq_results["documents"] and faq_results["documents"][0]:
            distance = faq_results["distances"][0][0]
            similarity = 1.0 - distance
            
            if similarity >= faq_threshold:
                return {
                    "query": query,
                    "internal_query": internal_query,
                    "retrieved_from": "faq",
                    "results": [{
                        "content": faq_results["documents"][0][0],
                        "metadata": faq_results["metadatas"][0][0],
                        "score": similarity,
                        "retrieval_score": similarity,
                        "rerank_score": None
                    }]
                }
                
        # 4. Fallback to Document Collection with candidate oversampling
        candidate_count = max(top_k * 3, 12)
        doc_results = self.db_manager.document_collection.query(
            query_embeddings=[query_vector],
            n_results=candidate_count
        )
        
        raw_candidates = []
        seen_contents = set()
        
        if doc_results and doc_results["documents"] and doc_results["documents"][0]:
            for i in range(len(doc_results["documents"][0])):
                content = doc_results["documents"][0][i]
                meta = doc_results["metadatas"][0][i]
                dist = doc_results["distances"][0][i]
                sim = 1.0 - dist
                
                # Content deduplication check
                content_key = content.strip().lower()
                if content_key in seen_contents:
                    continue
                seen_contents.add(content_key)
                
                raw_candidates.append({
                    "content": content,
                    "metadata": meta,
                    "score": sim,
                    "retrieval_score": sim
                })
                
        if not raw_candidates:
            return {
                "query": query,
                "internal_query": internal_query,
                "retrieved_from": "document",
                "results": []
            }

        # 5. Rerank candidate chunks using Cross-Encoder BAAI/bge-reranker-v2-m3
        try:
            reranker = self._get_reranker()
            final_results = reranker.rerank(query=internal_query, documents=raw_candidates, top_n=top_k)
        except Exception as e:
            print(f"[HybridRetriever Warning] Reranking fallback due to error: {e}")
            final_results = raw_candidates[:top_k]

        return {
            "query": query,
            "internal_query": internal_query,
            "retrieved_from": "document",
            "results": final_results
        }

    def get_document_chunks(self, target_document: str) -> List[Dict[str, Any]]:
        """
        Retrieves all chunks strictly belonging to target_document in ascending natural order.
        """
        if hasattr(self.db_manager, "get_document_chunks"):
            return self.db_manager.get_document_chunks(target_document)
        return []

    def get_indexed_documents(self) -> List[str]:
        """
        Returns a sorted list of all indexed document filenames.
        """
        if hasattr(self.db_manager, "get_indexed_documents"):
            return self.db_manager.get_indexed_documents()
        return []

