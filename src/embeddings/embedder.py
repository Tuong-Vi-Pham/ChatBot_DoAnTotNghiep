import torch
from typing import List
from sentence_transformers import SentenceTransformer

class Embedder:
    """
    Independent embedding module. Wraps Hugging Face sentence-transformers models.
    Supports batch text embedding, query embedding, and easy model replacement.
    Default model is upgraded to BAAI/bge-m3 with MPS/CUDA acceleration and normalized embeddings.
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None):
        self.model_name = model_name
        
        # Determine optimal device: Apple Silicon (MPS) -> CUDA -> CPU
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
                
        self.device = device
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts in batch with normalization.
        """
        if not texts:
            return []
        embeddings = self.model.encode(
            texts, 
            batch_size=16, 
            show_progress_bar=False, 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string with normalization.
        """
        embedding = self.model.encode(
            query, 
            show_progress_bar=False, 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist()

    def get_dimension(self) -> int:
        """
        Returns the output dimension of the current embedding model.
        """
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()
