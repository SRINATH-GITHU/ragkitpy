# ragkitpy/embedder.py
"""
embedder.py — Convert text chunks into vector embeddings using HuggingFace models.
"""

from typing import List
import numpy as np


class HFEmbedder:
    """
    Wrapper around HuggingFace SentenceTransformer models for generating embeddings.

    Args:
        model_name (str): Any SentenceTransformer model from HuggingFace Hub.
                          Default is 'all-MiniLM-L6-v2' — fast, small, great quality.

    Example:
        >>> embedder = HFEmbedder()
        >>> vectors = embedder.embed(["Hello world", "RAG is awesome"])
        >>> print(vectors.shape)  # (2, 384)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Run: pip install sentence-transformers"
            )

        print(f"🤗 Loading embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"✅ Model loaded!")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of strings into embedding vectors.

        Args:
            texts (List[str]): List of text strings to embed

        Returns:
            np.ndarray: 2D array of shape (len(texts), embedding_dim)

        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        # Filter out empty strings
        texts = [t.strip() for t in texts if t.strip()]

        if not texts:
            raise ValueError("All texts were empty after stripping whitespace.")

        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single string. Convenience method for query embedding.

        Args:
            text (str): A single string to embed

        Returns:
            np.ndarray: 1D embedding vector, shape (embedding_dim,)
        """
        if not text or not text.strip():
            raise ValueError("Input text is empty.")

        result = self.embed([text])
        return result[0]

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()

    def __repr__(self):
        return f"HFEmbedder(model='{self.model_name}', dim={self.embedding_dim})"