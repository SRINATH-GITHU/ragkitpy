# ragkitpy/vectorstore.py
"""
vectorstore.py — In-memory FAISS vector store for storing and searching embeddings.
"""

from typing import List, Tuple
import numpy as np


class VectorStore:
    """
    Lightweight in-memory vector store using FAISS for similarity search.

    Args:
        dim (int): Dimension of the embedding vectors

    Example:
        >>> store = VectorStore(dim=384)
        >>> store.add(embeddings, chunks)
        >>> results = store.search(query_embedding, top_k=3)
    """

    def __init__(self, dim: int):
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required. Run: pip install faiss-cpu"
            )

        import faiss
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)   # L2 distance (Euclidean)
        self.chunks: List[str] = []            # Stores original text chunks
        self._faiss = faiss

    def add(self, embeddings: np.ndarray, chunks: List[str]) -> None:
        """
        Add embeddings and their corresponding text chunks to the store.

        Args:
            embeddings (np.ndarray): 2D array of shape (n, dim)
            chunks (List[str]): Original text chunks matching each embedding
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings but {len(chunks)} chunks."
            )

        # FAISS requires float32
        embeddings = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        print(f"✅ Added {len(chunks)} chunks to vector store. Total: {len(self.chunks)}")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        """
        Find the most semantically similar chunks to a query embedding.

        Args:
            query_embedding (np.ndarray): 1D or 2D query vector
            top_k (int): Number of top results to return

        Returns:
            List[str]: Top-k most relevant text chunks
        """
        if len(self.chunks) == 0:
            raise RuntimeError("Vector store is empty. Call add() first.")

        # Ensure correct shape (1, dim)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        top_k = min(top_k, len(self.chunks))  # Can't return more than we have
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append(self.chunks[idx])

        return results

    def search_with_scores(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Same as search() but also returns similarity scores.

        Returns:
            List[Tuple[str, float]]: List of (chunk, distance) pairs
        """
        if len(self.chunks) == 0:
            raise RuntimeError("Vector store is empty. Call add() first.")

        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        top_k = min(top_k, len(self.chunks))
        distances, indices = self.index.search(query_embedding, top_k)

        return [
            (self.chunks[idx], float(dist))
            for idx, dist in zip(indices[0], distances[0])
            if idx < len(self.chunks)
        ]

    @property
    def total_chunks(self) -> int:
        """Return total number of chunks stored."""
        return len(self.chunks)

    def __repr__(self):
        return f"VectorStore(dim={self.dim}, chunks={self.total_chunks})"