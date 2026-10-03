import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("IntentClarifier")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _clean_text(text: str) -> str:
    """Helper to remove all asterisks (*) from string."""
    return text.replace("*", "").strip()


def reconstruct_query(original_query: str, selected_options: List[str]) -> str:
    """
    Reconstructs a comprehensive query combining original_query and all selected intent options.
    Preserves both original intent and clarified sub-intents.
    """
    clean_orig = original_query.strip()
    valid_opts = [opt.strip() for opt in selected_options if opt and opt.strip()]

    if not valid_opts:
        return clean_orig

    combined_opts = "; ".join(valid_opts)
    if combined_opts.lower() == clean_orig.lower():
        return clean_orig

    return f"{clean_orig} - {combined_opts}"


class IntentClarifier:
    """
    Detects ambiguous user queries and generates KB-grounded, verified English clarification options.
    """
    def __init__(self, llm_client: Any, retriever: Optional[Any] = None):
        self.llm_client = llm_client
        self.retriever = retriever

    def check_ambiguity(self, query: str, retriever: Optional[Any] = None) -> Dict[str, Any]:
        """
        Analyzes the query using KB grounding and LLM classification to determine ambiguity.
        Generates and verifies candidate clarification options against the Knowledge Base.
        """
        active_retriever = retriever or self.retriever
        cleaned_query = query.strip()
        logger.info(f"[IntentClarifier] Starting ambiguity check for original_query: '{cleaned_query}'")

        # 1. Initial Knowledge Base Retrieval for Context Grounding
        kb_snippets = []
        kb_context_summary = ""
        initial_results = []
        if active_retriever is not None:
            try:
                init_res = active_retriever.retrieve(cleaned_query, top_k=6)
                initial_results = init_res.get("results", [])
                logger.info(f"[IntentClarifier] Initial KB retrieval returned {len(initial_results)} items from {init_res.get('retrieved_from')}")
                for r in initial_results:
                    meta = r.get("metadata", {})
                    src = meta.get("source") or meta.get("filename") or "Doc"
                    sec = meta.get("section") or meta.get("category") or "General"
                    snippet = r.get("content", "")[:180].replace("\n", " ")
                    kb_snippets.append(f"- [{src} | {sec}]: {snippet}")
                if kb_snippets:
                    kb_context_summary = "\n".join(kb_snippets)
            except Exception as err:
                logger.warning(f"[IntentClarifier] Initial KB retrieval failed: {err}")

        # Short-circuit check for very short terms (<= 1 word)
        words = [w for w in cleaned_query.split() if w.lower() not in {"what", "is", "the", "a", "an", "for", "of", "to", "how", "why", "where"}]
        is_short = len(words) <= 1 and len(cleaned_query) > 0

        raw_candidates = []
        if is_short:
            raw_candidates = self._generate_grounded_options(cleaned_query, kb_context_summary)
        else:
            # LLM Ambiguity Classification Prompt with KB context grounding
            system_prompt = (
                "You are a query analysis assistant for a software project team's database.\n"
                "Analyze the user's search query for ambiguity vs completeness using the provided Knowledge Base context.\n\n"
                "Rule 1: If the query provides sufficient context or asks a clear, specific question, output exactly:\n"
                "STATUS: CLEAR\n\n"
                "Rule 2: If the query is ambiguous, overly brief, vague, or lacks sufficient context, output exactly:\n"
                "STATUS: AMBIGUOUS\n"
                "INTENTS:\n"
                "1. [Specific intent option 1]\n"
                "2. [Specific intent option 2]\n"
                "3. [Specific intent option 3]\n\n"
                "Important Rules:\n"
                "- Ground all generated options strictly in the provided Knowledge Base context when available.\n"
                "- Do NOT invent options for topics not present in the Knowledge Base context.\n"
                "- All generated options MUST be in plain English.\n"
                "- Do NOT use asterisks (*) or bold syntax (**) anywhere in your response."
            )
            user_prompt = f"Query: \"{cleaned_query}\"\n\nKnowledge Base Context:\n{kb_context_summary or 'None available'}"
            try:
                response = self.llm_client.generate_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.0
                )
                if "STATUS: AMBIGUOUS" in response:
                    parsed = re.findall(r'\d+\.\s*(.+)', response)
                    raw_candidates = [_clean_text(opt) for opt in parsed]
            except Exception as err:
                logger.warning(f"[IntentClarifier] Ambiguity classification failed: {err}")
                return {"is_ambiguous": False, "options": [], "original_query": cleaned_query}

        if not raw_candidates:
            return {"is_ambiguous": False, "options": [], "original_query": cleaned_query}

        logger.info(f"[IntentClarifier] Generated raw candidate_intents: {raw_candidates}")

        # 2. Candidate Verification Stage
        verified_options = []
        verification_log = []
        for cand in raw_candidates:
            if not cand:
                continue
            is_valid = True
            if active_retriever is not None:
                try:
                    verify_res = active_retriever.retrieve(cand, top_k=3)
                    results = verify_res.get("results", [])
                    # Verify candidate has matching documents with adequate retrieval score (>= 0.35)
                    if not results:
                        is_valid = False
                    else:
                        best_score = max(r.get("retrieval_score", r.get("score", 0.0)) for r in results)
                        is_valid = best_score >= 0.35
                except Exception as err:
                    logger.warning(f"[IntentClarifier] Verification check failed for '{cand}': {err}")
                    is_valid = True  # Fallback retain if check errors out

            verification_log.append({"candidate": cand, "verified": is_valid})
            if is_valid and cand not in verified_options:
                verified_options.append(cand)

        logger.info(f"[IntentClarifier] Verification results: {verification_log}")
        logger.info(f"[IntentClarifier] Final verified options ({len(verified_options)}): {verified_options}")

        if not verified_options:
            return {
                "is_ambiguous": False,
                "options": [],
                "original_query": cleaned_query,
                "relevant_context": kb_context_summary
            }

        # Keep 1, 2, or up to 3 verified options. Never invent fake options just to reach 3.
        return {
            "is_ambiguous": True,
            "original_query": cleaned_query,
            "options": verified_options[:3],
            "relevant_context": kb_context_summary,
            "verification_log": verification_log
        }

    def _generate_grounded_options(self, query: str, kb_context: str) -> List[str]:
        """
        Helper method to generate KB-grounded candidate clarification options.
        """
        system_prompt = (
            "You are a query expansion assistant for a software project team.\n"
            "Expand the ambiguous search term into 1 to 3 specific, distinct search targets "
            "STRICTLY grounded in the provided Knowledge Base snippets.\n\n"
            "Output format:\n"
            "1. [Option 1]\n"
            "2. [Option 2]\n"
            "3. [Option 3]\n\n"
            "Ensure options are concise, in plain English, and contain NO asterisks (*) or bold syntax."
        )
        user_prompt = f"Ambiguous term: \"{query}\"\n\nKnowledge Base Context:\n{kb_context or 'None'}"
        try:
            response = self.llm_client.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2
            )
            parsed = re.findall(r'\d+\.\s*(.+)', response)
            if parsed:
                return [_clean_text(opt) for opt in parsed]
        except Exception as err:
            logger.warning(f"[IntentClarifier] Grounded option generation failed: {err}")

        # Fallback options if LLM call fails
        clean_q = _clean_text(query)
        return [
            f"Project guidelines regarding {clean_q}",
            f"Requirements specification for {clean_q}",
            f"Troubleshooting and processes for {clean_q}"
        ]
