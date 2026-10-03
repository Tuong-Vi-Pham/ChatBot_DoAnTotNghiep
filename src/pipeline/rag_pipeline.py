import re
import logging
from typing import Dict, Any, List, Optional
from src.loaders.document import Document
from src.retrieval.hybrid import HybridRetriever
from src.llm.client import LLMClient
from src.verification.validator import VerificationLoop
from src.intent_clarification.intent_classifier import IntentClassifier
from src.summarization.service import SummarizationService
from src.summarization.map_summarizer import MapSummarizer
from src.summarization.reducer import ReduceSummarizer
from src.summarization.document_resolver import DocumentResolver
from models.conversation_history import build_context_prompt
from src.agent.graph import AgentOrchestrator
from src.agent.nodes.intent_router import IntentRouterNode

logger = logging.getLogger(__name__)



def strip_asterisks(text: str) -> str:
    """Removes all asterisk (*) characters from text."""
    if not text:
        return ""
    return text.replace("*", "")


class RAGPipeline:
    """
    RAG Pipeline orchestrator optimized for Qwen3-14B-Instruct and LangGraph Agent integration.
    Connects intent classification, hybrid vector search, cross-encoder reranking,
    Map-Reduce Summarization, LangGraph Orchestration, LLM response generation, and source citation formatting.
    """
    def __init__(
        self, 
        retriever: HybridRetriever, 
        llm_client: LLMClient, 
        validator: VerificationLoop = None,
        intent_classifier: IntentClassifier = None,
        map_summarizer: MapSummarizer = None,
        reduce_summarizer: ReduceSummarizer = None,
        orchestrator: Optional[AgentOrchestrator] = None,
        document_resolver: Optional[DocumentResolver] = None
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.validator = validator if validator is not None else VerificationLoop(llm_client)
        self.intent_classifier = intent_classifier if intent_classifier is not None else IntentClassifier()
        self.orchestrator = orchestrator
        self.document_resolver = document_resolver if document_resolver is not None else DocumentResolver()
        
        # Summarization components
        summarization_service = SummarizationService(llm_client=self.llm_client)
        self.map_summarizer = map_summarizer if map_summarizer is not None else MapSummarizer(summarization_service)
        self.reduce_summarizer = reduce_summarizer if reduce_summarizer is not None else ReduceSummarizer(summarization_service)

    def format_citations(self, sources_list: List[Dict[str, Any]]) -> str:
        """
        Formats a clean, deduplicated markdown citation block from retrieved sources metadata.
        """
        if not sources_list:
            return ""

        seen = set()
        formatted_citations = []

        for src in sources_list:
            src_name = strip_asterisks(src.get("source") or src.get("filename") or "Document")
            section = strip_asterisks(src.get("section") or src.get("category") or "General")
            chunk_id = strip_asterisks(src.get("chunk_id") or f"chunk_{src.get('chunk_index', 0)+1}")
            score = src.get("score")
            score_str = f" (Similarity: {score:.2f})" if score is not None else ""

            key = (src_name, section)
            if key in seen:
                continue
            seen.add(key)

            if src.get("type") == "faq":
                formatted_citations.append(f"• FAQ: {src_name} | Category: {section}{score_str}")
            else:
                formatted_citations.append(f"• Document: {src_name} | Section: {section} | Chunk: {chunk_id}{score_str}")

        if not formatted_citations:
            return ""

        return "\n\n---\nReferences & Citations:\n" + "\n".join(formatted_citations)

    def run(
        self, 
        query: str, 
        faq_threshold: float = 0.70, 
        top_k: int = 4,
        temperature: float = 0.2,
        chat_id: Optional[str] = None,
        current_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes the optimized RAG / Agentic flow:
        Checks for Agentic Ticket/Priority/Approval query -> LangGraph AgentOrchestrator OR Intent Classify -> Hybrid Retrieval + Reranking -> Citations.
        """
        query_lower = query.lower()
        router_intent = IntentRouterNode.execute({"user_query": query}).get("intent")
        is_agent_query = (
            router_intent in (
                "PRIORITY_ANALYSIS", "TICKET_QUERY", "TICKET_ANALYSIS",
                "PRIORITY_APPROVAL_REQUEST", "PRIORITY_APPROVAL_DECISION", "PRIORITY_UPDATE_EXECUTION"
            ) or
            re.search(r'\b(task|epic|bug|sub|crm|story)-\d+\b', query_lower) is not None or
            re.search(r'\bapr-\d+\b', query_lower) is not None
        )

        if is_agent_query:
            if self.orchestrator is None:
                self.orchestrator = AgentOrchestrator(rag_tool=None)
            logger.info(f"[RAGPipeline] Delegating agentic query to AgentOrchestrator: '{query}' (session_id: {chat_id})")
            orchestrator_res = self.orchestrator.run(query, session_id=chat_id)
            return {
                "query": query,
                "answer": strip_asterisks(orchestrator_res.get("final_response", "")),
                "retrieved_from": "agent_orchestrator",
                "sources": [],
                "agent_state": orchestrator_res
            }


        # 1. Intent Classification Check
        intent_res = self.intent_classifier.classify(query)
        if intent_res.get("bypass_rag"):
            return {
                "query": query,
                "answer": intent_res.get("response", self.validator.default_insufficient_msg),
                "retrieved_from": "none",
                "sources": [],
                "intent": intent_res.get("intent")
            }

        # 2. Handle Summarization Intent
        if intent_res.get("intent") == "Summarization":
            return self._run_summarization_flow(
                query=query,
                faq_threshold=faq_threshold,
                top_k=top_k,
                temperature=temperature
            )

        logger.info(f"[RAGPipeline] Executing query: '{query}' (chat_id: {chat_id})")

        # 3. Retrieve context for standard RAG Q&A
        sub_queries = [s.strip() for s in query.split(";") if s.strip()]
        if len(sub_queries) > 1:
            merged_results = []
            seen_contents = set()
            for sub_q in sub_queries:
                try:
                    sub_res = self.retriever.retrieve(sub_q, faq_threshold=faq_threshold, top_k=top_k)
                    for r in sub_res.get("results", []):
                        ckey = r.get("content", "").strip().lower()
                        if ckey not in seen_contents:
                            seen_contents.add(ckey)
                            merged_results.append(r)
                except Exception as err:
                    logger.warning(f"[RAGPipeline] Sub-query retrieval failed for '{sub_q}': {err}")
            results = merged_results
            source_type = "document"
        else:
            retrieval_res = self.retriever.retrieve(query, faq_threshold=faq_threshold, top_k=top_k)
            source_type = retrieval_res["retrieved_from"]
            results = retrieval_res["results"]

        logger.info(f"[RAGPipeline] Final retrieval results count: {len(results)} (source_type: {source_type})")
        
        # 3. Handle empty retrieval or inadequate evidence
        is_adequate = True
        if source_type == "document":
            is_adequate = self.validator.check_adequacy(results, score_threshold=0.40)
            
        if not results or not is_adequate:
            logger.warning(f"[RAGPipeline] Evidence inadequate for query: '{query}'")
            return {
                "query": query,
                "answer": self.validator.default_insufficient_msg,
                "retrieved_from": "none",
                "sources": []
            }
            
        # 4. Construct prompt based on retrieval source
        context_str = ""
        sources_list = []
        conversation_context = ""
        if chat_id:
            conversation_context = build_context_prompt(
                chat_id=chat_id,
                current_question=query,
                limit=8,
                exclude_message_id=current_message_id,
            )
        
        if source_type == "faq":
            # Direct FAQ hit
            faq_item = results[0]
            context_str = faq_item["content"]
            meta = faq_item["metadata"]
            sources_list.append({
                "source": meta.get("source"),
                "filename": meta.get("filename", meta.get("source")),
                "category": meta.get("category"),
                "section": meta.get("section", meta.get("category")),
                "score": faq_item.get("retrieval_score", faq_item.get("score", 1.0)),
                "retrieval_score": faq_item.get("retrieval_score", faq_item.get("score", 1.0)),
                "rerank_score": faq_item.get("rerank_score"),
                "type": "faq"
            })
            
            system_prompt = (
                "You are an expert, factually precise AI assistant for a software team.\n"
                "An exact matching FAQ entry was found in the project knowledge base.\n"
                "Provide a clear, direct, and concise response using ONLY the provided FAQ entry.\n"
                "Do not hallucinate, fabricate facts, or extrapolate beyond the FAQ text."
            )
            user_prompt = (
                f"Conversation Context:\n{conversation_context}\n\n"
                if conversation_context and conversation_context != query else ""
            ) + f"FAQ Entry:\n{context_str}\n\nUser Question: {query}\nAnswer:"
            
        else:
            # Document chunks fallback
            context_parts = []
            for idx, item in enumerate(results):
                meta = item["metadata"]
                src = meta.get("source") or meta.get("filename") or "Doc"
                section = meta.get("section") or meta.get("category") or "General"
                c_idx = meta.get("chunk_index", 0)
                tot_c = meta.get("total_chunks", 1)
                chunk_id = meta.get("chunk_id", f"chunk_{c_idx+1}")
                
                context_parts.append(
                    f"--- Reference [{idx+1}] File: {src} | Section: {section} | ID: {chunk_id} (Chunk {c_idx+1}/{tot_c}) ---\n"
                    f"{item['content']}"
                )
                sources_list.append({
                    "source": src,
                    "filename": src,
                    "category": meta.get("category"),
                    "section": section,
                    "score": item.get("retrieval_score", item.get("score", 0.0)),
                    "retrieval_score": item.get("retrieval_score", item.get("score", 0.0)),
                    "rerank_score": item.get("rerank_score"),
                    "chunk_index": c_idx,
                    "total_chunks": tot_c,
                    "chunk_id": chunk_id,
                    "type": "document"
                })
                
            context_str = "\n\n".join(context_parts)
            
            system_prompt = (
                "You are a Senior Technical AI Assistant specializing in software engineering documentation.\n"
                "Answer the User Question accurately, concisely, and professionally using ONLY the provided Document Context.\n\n"
                "Strict Rules:\n"
                "1. Base your answer strictly on the provided Context. Do NOT use external knowledge or fabricate details.\n"
                "2. If the context does not contain sufficient facts to answer the question, state exactly:\n"
                "   \"I could not find sufficient information in the knowledge base. Please rephrase your question or provide more context.\"\n"
                "3. Keep your answer factual, direct, and free of speculation."
            )
            
            user_prompt = (
                (f"Conversation Context:\n{conversation_context}\n\n" if conversation_context and conversation_context != query else "")
                + f"Document Context:\n{context_str}\n\n"
                + f"User Question: {query}\n\n"
                + "Answer:"
            )

        # 5. Generate response using LLM
        answer = self.llm_client.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature
        )
        
        # 6. Post-inference grounding audit
        if source_type == "document":
            is_grounded = self.validator.verify_grounding(context_str, answer)
            if not is_grounded:
                answer = self.validator.default_insufficient_msg

        # 7. Append Source Citations
        citation_block = self.format_citations(sources_list)
        if citation_block and self.validator.default_insufficient_msg not in answer:
            answer = f"{answer.strip()}{citation_block}"

        return {
            "query": query,
            "answer": strip_asterisks(answer),
            "retrieved_from": source_type,
            "sources": sources_list
        }

    def _select_document_chunks_for_summary(
        self,
        chunks: List[Document],
        query: str,
        max_chunks: int = 25
    ) -> List[Document]:
        """
        Selects a complete, structurally representative set of chunks for long documents:
        - If len(chunks) <= max_chunks: retains all chunks.
        - If len(chunks) > max_chunks:
          1. Always retains first 10 chunks (covers Introduction, Scope, Definitions, References).
          2. Always retains last 3 chunks (covers Appendices / final sections).
          3. Always retains any chunks matching explicit sections in query (e.g. definitions, references).
          4. Evenly samples across the middle sections to ensure 100% full-document coverage.
        All selected chunks are strictly ordered by chunk_index ascending (natural document order).
        """
        if len(chunks) <= max_chunks:
            return chunks

        q_lower = query.lower()
        selected_indices = set()

        # 1. Beginning: first 10 chunks
        for i in range(min(10, len(chunks))):
            selected_indices.add(i)

        # 2. Ending: last 3 chunks
        for i in range(max(0, len(chunks) - 3), len(chunks)):
            selected_indices.add(i)

        # 3. Query-specific structural keywords
        keywords = []
        if "definition" in q_lower:
            keywords.append("definition")
        if "reference" in q_lower:
            keywords.append("reference")
        if "scope" in q_lower:
            keywords.append("scope")
        if "security" in q_lower:
            keywords.append("security")
        if "requirement" in q_lower:
            keywords.append("requirement")

        for idx, c in enumerate(chunks):
            sec = c.metadata.get("section", "").lower()
            if any(k in sec for k in keywords):
                selected_indices.add(idx)

        # 4. Fill remaining slots with evenly spaced chunks across the document
        remaining_slots = max_chunks - len(selected_indices)
        if remaining_slots > 0:
            step = len(chunks) / (remaining_slots + 1)
            for j in range(1, remaining_slots + 1):
                target_idx = int(j * step)
                if 0 <= target_idx < len(chunks):
                    selected_indices.add(target_idx)

        sorted_indices = sorted(selected_indices)
        return [chunks[i] for i in sorted_indices]

    def _run_summarization_flow(
        self,
        query: str,
        faq_threshold: float = 0.70,
        top_k: int = 4,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Executes the Summarization flow:
        1. Resolves if a specific target document is requested.
        2. If ambiguous -> returns clarification request.
        3. If target document resolved -> document-scoped summarization over complete target document.
        4. If no document specified -> falls back to topic-based retrieval summarization.
        """
        # 1. Check known indexed documents
        known_docs = []
        if hasattr(self.retriever, "get_indexed_documents"):
            try:
                known_docs = self.retriever.get_indexed_documents()
            except Exception:
                known_docs = []

        resolution = self.document_resolver.resolve(query, candidate_docs=known_docs if known_docs else None)

        # Case 1: Ambiguous document name
        if resolution.get("status") == "ambiguous":
            clarification_msg = resolution.get(
                "clarification_message",
                f'I found multiple documents matching "{resolution.get("query_term", "your request")}". Please select the document you want me to summarize.'
            )
            return {
                "query": query,
                "answer": strip_asterisks(clarification_msg),
                "retrieved_from": "clarification",
                "sources": [],
                "intent": "Summarization",
                "ambiguity": resolution
            }

        # Case 2: Document explicitly requested with extension but not found in KB
        if resolution.get("status") == "not_found":
            target_doc = resolution.get("target_document", "the requested document")
            return {
                "query": query,
                "answer": f"I could not find the document '{target_doc}' in the knowledge base. Please check the document name or rephrase your request.",
                "retrieved_from": "summarization",
                "sources": [],
                "intent": "Summarization"
            }

        # Case 3: Specific target document resolved -> Document-Scoped Summarization
        if resolution.get("status") == "resolved":
            target_doc = resolution.get("target_document")
            raw_chunks = []
            if hasattr(self.retriever, "get_document_chunks"):
                raw_chunks = self.retriever.get_document_chunks(target_doc)

            if not raw_chunks:
                return {
                    "query": query,
                    "answer": self.validator.default_insufficient_msg,
                    "retrieved_from": "document_scoped",
                    "sources": [],
                    "intent": "Summarization",
                    "target_document": target_doc
                }

            # Convert to Document objects strictly from target_doc in natural order
            all_doc_chunks = []
            sources_list = []
            for item in raw_chunks:
                meta = item.get("metadata", {})
                c_idx = meta.get("chunk_index", item.get("chunk_index", 0))
                tot_c = meta.get("total_chunks", len(raw_chunks))
                chunk_id = item.get("chunk_id") or meta.get("chunk_id", f"chunk_{c_idx+1}")
                src = meta.get("source") or meta.get("filename") or target_doc
                sec = meta.get("section") or meta.get("category") or "General"

                all_doc_chunks.append(Document(page_content=item.get("content", ""), metadata=meta))
                sources_list.append({
                    "source": src,
                    "filename": src,
                    "category": meta.get("category"),
                    "section": sec,
                    "score": item.get("score", 1.0),
                    "chunk_index": c_idx,
                    "total_chunks": tot_c,
                    "chunk_id": chunk_id,
                    "type": "document"
                })

            # For long documents, select structural representation covering all sections
            doc_chunks_to_summarize = self._select_document_chunks_for_summary(all_doc_chunks, query)

            map_summaries = self.map_summarizer.summarize_chunks(doc_chunks_to_summarize)
            final_summary = self.reduce_summarizer.reduce_summaries(
                map_summaries,
                temperature=temperature
            )

            if not final_summary or not final_summary.strip():
                final_summary = self.validator.default_insufficient_msg

            # Format citations STRICTLY from target document only
            target_sources = [s for s in sources_list if (s.get("source") == target_doc or s.get("filename") == target_doc)]
            citation_block = self.format_citations(target_sources)
            if citation_block and self.validator.default_insufficient_msg not in final_summary:
                final_summary = f"{final_summary.strip()}{citation_block}"

            return {
                "query": query,
                "answer": strip_asterisks(final_summary),
                "retrieved_from": "document_scoped",
                "sources": target_sources,
                "intent": "Summarization",
                "target_document": target_doc
            }

        # Case 4: No specific document specified -> Topic-Based Summarization (existing flow)
        retrieval_res = self.retriever.retrieve(query, faq_threshold=faq_threshold, top_k=top_k)
        results = retrieval_res.get("results", [])
        source_type = retrieval_res.get("retrieved_from", "document")

        is_adequate = True
        if source_type == "document":
            is_adequate = self.validator.check_adequacy(results, score_threshold=0.40)

        if not results or not is_adequate:
            return {
                "query": query,
                "answer": self.validator.default_insufficient_msg,
                "retrieved_from": "summarization",
                "sources": [],
                "intent": "Summarization"
            }

        doc_chunks = []
        sources_list = []
        for item in results:
            content = item["content"]
            meta = item["metadata"]
            doc_chunks.append(Document(page_content=content, metadata=meta))

            src = meta.get("source") or meta.get("filename") or "Doc"
            section = meta.get("section") or meta.get("category") or "General"
            c_idx = meta.get("chunk_index", 0)
            tot_c = meta.get("total_chunks", 1)
            chunk_id = meta.get("chunk_id", f"chunk_{c_idx+1}")

            sources_list.append({
                "source": src,
                "filename": src,
                "category": meta.get("category"),
                "section": section,
                "score": item.get("score", 0.0),
                "chunk_index": c_idx,
                "total_chunks": tot_c,
                "chunk_id": chunk_id,
                "type": "document"
            })

        map_summaries = self.map_summarizer.summarize_chunks(doc_chunks)
        final_summary = self.reduce_summarizer.reduce_summaries(
            map_summaries,
            temperature=temperature
        )

        if not final_summary or not final_summary.strip():
            final_summary = self.validator.default_insufficient_msg

        citation_block = self.format_citations(sources_list)
        if citation_block and self.validator.default_insufficient_msg not in final_summary:
            final_summary = f"{final_summary.strip()}{citation_block}"

        return {
            "query": query,
            "answer": strip_asterisks(final_summary),
            "retrieved_from": "summarization",
            "sources": sources_list,
            "intent": "Summarization"
        }
