import os

# Orchestration & Bounded Loop Safeguards
MAX_AGENT_ITERATIONS: int = int(os.getenv("MAX_AGENT_ITERATIONS", "3"))
MAX_RETRIEVAL_ITERATIONS: int = int(os.getenv("MAX_RETRIEVAL_ITERATIONS", "2"))

# Similarity & Verification Thresholds
FAQ_SIMILARITY_THRESHOLD: float = float(os.getenv("FAQ_SIMILARITY_THRESHOLD", "0.70"))
DOCUMENT_SIMILARITY_THRESHOLD: float = float(os.getenv("DOCUMENT_SIMILARITY_THRESHOLD", "0.35"))
RERANK_SCORE_THRESHOLD: float = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.30"))
