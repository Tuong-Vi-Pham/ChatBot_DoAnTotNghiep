import re
from typing import Dict, Any

class IntentClassifier:
    """
    Lightweight Intent Classifier module.
    Classifies queries into standard intents without extra ML overhead:
    - Greeting
    - Small Talk
    - FAQ
    - Knowledge Search
    - Unknown
    
    Greetings and Small Talk bypass RAG to avoid useless vector searches and reduce latency.
    """
    GREETINGS = {
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening",
        "howdy", "yo", "hi there", "hello there", "xin chào", "xin chao", "chào", "chao",
        "chào bạn", "chao ban", "chào ad", "chao ad", "alo", "halo", "hê lô", "he lo"
    }
    
    SMALL_TALK = {
        "thanks", "thank you", "cảm ơn", "cam on", "cám ơn", "how are you", "hows it going",
        "who are you", "what can you do", "help", "trợ giúp", "bạn là ai", "tất cả bao nhiêu"
    }

    SUMMARIZATION_KEYWORDS = {
        "tóm tắt", "tom tat", "summarize", "summary", "tổng hợp", "tong hop",
        "báo cáo tổng quan", "bao cao tong quan", "khái quát", "khai quat"
    }

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classifies the intent of the incoming user query.
        """
        if not query or not query.strip():
            return {"intent": "Unknown", "bypass_rag": False}

        cleaned = re.sub(r'[^\w\s]', '', query).strip().lower()

        if cleaned in self.GREETINGS:
            return {
                "intent": "Greeting",
                "bypass_rag": True,
                "response": "Hello! I am your SmartLogi assistant. How can I help you with your software project documentation today?"
            }

        if cleaned in self.SMALL_TALK:
            return {
                "intent": "Small Talk",
                "bypass_rag": True,
                "response": "You're welcome! Feel free to ask any question regarding project requirements, deployment, architecture, or FAQs."
            }

        # Check for Summarization Intent
        if any(kw in cleaned for kw in self.SUMMARIZATION_KEYWORDS):
            return {
                "intent": "Summarization",
                "bypass_rag": False
            }

        # Check for search vs unknown
        if len(cleaned.split()) < 1:
            return {
                "intent": "Unknown",
                "bypass_rag": False
            }

        return {
            "intent": "Knowledge Search",
            "bypass_rag": False
        }
