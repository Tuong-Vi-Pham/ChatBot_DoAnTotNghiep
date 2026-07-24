from typing import List
from sentence_transformers import SentenceTransformer

class Embedder:
    """
    Independent embedding module. Wraps Hugging Face sentence-transformers models.
    Supports batch text embedding, query embedding, and easy model replacement.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        # SentenceTransformer handles model downloading and caching automatically
        self.model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts in batch.
        """
        if not texts:
            return []
        embeddings = self.model.encode(
            texts, 
            batch_size=32, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string.
        """
        embedding = self.model.encode(
            query, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        return embedding.tolist()

    def get_dimension(self) -> int:
        """
        Returns the output dimension of the current embedding model.
        """
        return self.model.get_sentence_embedding_dimension()
