import logging
from typing import Any, Dict, List, Optional
from src.agent.state import Evidence
from src.retrieval.hybrid import HybridRetriever
from src.embeddings.embedder import Embedder
from src.vectordb.database import VectorDBManager
from src.retrieval.reranker import BGEReranker

logger = logging.getLogger(__name__)


class RAGToolAdapter:
    """
    Adapter exposing RAG retrieval operations as LangGraph-compatible tools.
    Supports source-aware filtering (CRM, Tech_Team, BOTH) and packages results into Evidence objects.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        db_manager: Optional[VectorDBManager] = None,
        embedder: Optional[Embedder] = None,
        reranker: Optional[BGEReranker] = None
    ):
        if retriever is not None:
            self.retriever = retriever
        else:
            emb = embedder or Embedder()
            vdb = db_manager or VectorDBManager(embedder=emb)
            self.retriever = HybridRetriever(db_manager=vdb, embedder=emb, reranker=reranker)

    def retrieve(
        self,
        query: str,
        scope: str = "BOTH",
        top_k: int = 5,
        faq_threshold: float = 0.70
    ) -> Dict[str, Any]:
        """
        Executes source-aware retrieval and formats results into structured Evidence list.
        """
        try:
            raw_res = self.retriever.retrieve(query=query, faq_threshold=faq_threshold, top_k=top_k * 2)
            retrieved_from = raw_res.get("retrieved_from", "document")
            results = raw_res.get("results", [])

            evidence_list: List[Evidence] = []
            
            for item in results:
                content = item.get("content", "")
                metadata = item.get("metadata", {})
                score = float(item.get("score") or item.get("rerank_score") or item.get("retrieval_score") or 0.0)
                
                cat = str(metadata.get("category", "")).lower()
                rel_path = str(metadata.get("relative_path") or metadata.get("source_path") or metadata.get("source") or "").lower()

                # Source Classification
                if retrieved_from == "faq" or metadata.get("source_type") == "faq":
                    src = "FAQ"
                    ev_type = "BUSINESS"
                elif "crm" in rel_path or "crm" in cat or "brd" in rel_path or "srs" in rel_path:
                    src = "CRM"
                    ev_type = "BUSINESS"
                else:
                    src = "Tech_Team"
                    ev_type = "TECHNICAL"

                # Scope Filtering
                if scope == "CRM" and src == "Tech_Team":
                    continue
                if scope == "Tech_Team" and src == "CRM":
                    continue

                doc_id = str(metadata.get("document_id") or metadata.get("source") or "unknown_doc")
                chunk_id = str(metadata.get("chunk_id") or f"{doc_id}_chunk_0")
                source_path = str(metadata.get("source_path") or metadata.get("relative_path") or metadata.get("source") or "")

                evidence = Evidence(
                    source=src,
                    document_id=doc_id,
                    chunk_id=chunk_id,
                    source_path=source_path,
                    content=content,
                    relevance_score=score,
                    ticket_id=metadata.get("ticket_id"),
                    evidence_type=ev_type
                )
                evidence_list.append(evidence)

                if len(evidence_list) >= top_k:
                    break

            return {
                "success": True,
                "query": query,
                "scope": scope,
                "retrieved_from": retrieved_from,
                "evidence_count": len(evidence_list),
                "evidence": [ev.to_dict() for ev in evidence_list]
            }
        except Exception as e:
            logger.error(f"RAGToolAdapter.retrieve failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "evidence_count": 0,
                "evidence": []
            }
