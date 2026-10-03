import re
from typing import Dict

class QueryRewriter:
    """
    Internal Query Rewriter module.
    Automatically cleans, expands abbreviations, corrects domain typos,
    and enriches keyword queries before sending to vector search.
    The rewritten query remains internal and is never displayed to end users.
    """
    ABBREVIATIONS: Dict[str, str] = {
        "pm": "project manager",
        "ba": "business analyst",
        "qa": "quality assurance tester",
        "dev": "developer",
        "devs": "developers",
        "budgt": "budget",
        "smatlogi": "smartlogi",
        "smartlogi": "SmartLogi",
        "env": "environment",
        "doc": "document",
        "docs": "documentation",
        "uat": "user acceptance testing",
        "reqs": "requirements",
        "spec": "specification",
        "specs": "specifications",
        "arch": "architecture",
        "repo": "repository"
    }

    def rewrite(self, query: str) -> str:
        """
        Rewrites and expands the raw input query for enhanced IR recall.
        """
        if not query or not query.strip():
            return query

        cleaned = query.strip()
        tokens = cleaned.split()
        rewritten_tokens = []

        for token in tokens:
            # Strip punctuation for dictionary lookup
            norm_token = re.sub(r'[^\w]', '', token).lower()
            if norm_token in self.ABBREVIATIONS:
                replacement = self.ABBREVIATIONS[norm_token]
                rewritten_tokens.append(replacement)
            else:
                rewritten_tokens.append(token)

        rewritten = " ".join(rewritten_tokens)

        # If query is keyword-only, add domain context hint internally
        if len(tokens) <= 3 and not any(wh in cleaned.lower() for wh in ["what", "who", "where", "how", "when", "why", "which"]):
            rewritten = f"{rewritten} software project technical documentation"

        return rewritten
