import re
from typing import Dict, Any, List
from src.llm.client import LLMClient

class IntentClarifier:
    """
    Primary contribution of the project.
    Detects ambiguous user queries and generates specific clarification options.
    """
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def check_ambiguity(self, query: str) -> Dict[str, Any]:
        """
        Analyzes the query using the LLM to determine if it is ambiguous,
        and generates specific candidate intents if so.
        """
        # Trim query
        cleaned_query = query.strip()
        
        # Heuristic shortcut: if the query is extremely short (e.g. 1 word), it is almost always ambiguous
        words = [w for w in cleaned_query.split() if w.lower() not in {"what", "is", "the", "a", "an", "for", "of", "to"}]
        if len(words) <= 1 and len(cleaned_query) > 0:
            # Short-circuit to trigger LLM intent generation
            return self._generate_clarification_options(cleaned_query)

        # Build prompt for LLM classification
        system_prompt = (
            "You are a query analysis assistant for a software project team's database. "
            "Analyze the user's search query for ambiguity, brevity, or lack of context.\n\n"
            "If the query is a clear, specific question or requests specific documentation (e.g., 'What is the total budget for the project?', "
            "'How does onboarding work?', 'Explain the database schema design'), output exactly:\n"
            "STATUS: CLEAR\n\n"
            "If the query is ambiguous, too short, vague, or could refer to multiple different components/features (e.g., 'search order', "
            "'testing', 'onboarding', 'deployment'), output exactly:\n"
            "STATUS: AMBIGUOUS\n"
            "INTENTS:\n"
            "1. [Specific intent option 1]\n"
            "2. [Specific intent option 2]\n"
            "3. [Specific intent option 3]\n\n"
            "Keep the labels 'STATUS:' and 'INTENTS:' exactly as shown. Generate 3 specific, distinct, realistic intent options "
            "relevant to a software team's business specifications, system designs, deployment guides, or onboarding."
        )

        user_prompt = f"Query: \"{cleaned_query}\""
        
        try:
            response = self.llm_client.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0  # Use low temperature for consistent output structure
            )
            
            # Parse response
            if "STATUS: AMBIGUOUS" in response:
                options = re.findall(r'\d+\.\s*(.+)', response)
                if len(options) >= 2:
                    return {
                        "is_ambiguous": True,
                        "options": [opt.strip() for opt in options[:3]]
                    }
            
            return {
                "is_ambiguous": False,
                "options": []
            }
            
        except Exception as e:
            # Fallback in case of LLM failure: treat as clear to avoid blocking RAG
            print(f"[Intent Clarifier Warning] Ambiguity check failed: {e}")
            return {
                "is_ambiguous": False,
                "options": []
            }

    def _generate_clarification_options(self, query: str) -> Dict[str, Any]:
        """
        Helper method to generate clarification options specifically when a query
        is known to be too short.
        """
        system_prompt = (
            "You are a query expansion assistant for a software project team. "
            "The user provided a very short, ambiguous search term. Expand it into 3 specific questions or search targets "
            "relevant to software project documents (requirements, design, testing, onboarding, operations).\n\n"
            "Output format:\n"
            "1. [Option 1]\n"
            "2. [Option 2]\n"
            "3. [Option 3]\n\n"
            "Ensure options are concise and distinct."
        )
        user_prompt = f"Ambiguous term: \"{query}\""
        
        try:
            response = self.llm_client.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3
            )
            options = re.findall(r'\d+\.\s*(.+)', response)
            if len(options) >= 2:
                return {
                    "is_ambiguous": True,
                    "options": [opt.strip() for opt in options[:3]]
                }
        except Exception:
            pass
            
        # Hardcoded fallback if LLM completely fails
        return {
            "is_ambiguous": True,
            "options": [
                f"Project guidelines regarding {query}",
                f"Requirements specification for {query}",
                f"Troubleshooting and processes for {query}"
            ]
        }
