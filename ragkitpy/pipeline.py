# ragkitpy/pipeline.py
"""
pipeline.py — End-to-end RAG pipeline. Glues loader, chunker, embedder, and vectorstore.
"""

from typing import List, Optional, Tuple
from ragkitpy.loader import load_file
from ragkitpy.chunker import chunk_text
from ragkitpy.embedder import HFEmbedder
from ragkitpy.vectorstore import VectorStore


class RAGPipeline:
    """
    A lightweight end-to-end RAG (Retrieval-Augmented Generation) pipeline.

    Loads a document → chunks it → embeds chunks → stores in FAISS vector store
    → retrieves relevant chunks for any query.

    Args:
        model_name (str): HuggingFace SentenceTransformer model to use.
                          Default: 'all-MiniLM-L6-v2' (fast + good quality)
        chunk_size (int): Characters per chunk. Default: 500
        overlap (int): Overlapping characters between chunks. Default: 50

    Example:
        >>> from ragkitpy import RAGPipeline
        >>> rag = RAGPipeline()
        >>> rag.load_document("my_file.pdf")
        >>> results = rag.query("What is this document about?")
        >>> for chunk in results:
        ...     print(chunk)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedder = HFEmbedder(model_name)
        self.store: Optional[VectorStore] = None
        self._document_loaded = False
        self._source_path: Optional[str] = None

    def load_document(self, path: str) -> None:
        """
        Load a document, chunk it, embed it, and store in vector index.

        Args:
            path (str): Path to a .txt or .pdf file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is unsupported
        """
        print(f"\n📄 Loading document: {path}")

        # Step 1: Load raw text
        text = load_file(path)
        print(f"✅ Extracted {len(text)} characters")

        # Step 2: Chunk
        chunks = chunk_text(text, self.chunk_size, self.overlap)
        print(f"✅ Split into {len(chunks)} chunks")

        # Step 3: Embed
        print(f"⚙️  Embedding chunks...")
        embeddings = self.embedder.embed(chunks)

        # Step 4: Store
        self.store = VectorStore(dim=embeddings.shape[1])
        self.store.add(embeddings, chunks)

        self._document_loaded = True
        self._source_path = path
        print(f"\n🚀 RAG pipeline ready! You can now call .query()")

    def query(self, question: str, top_k: int = 3) -> List[str]:
        """
        Retrieve the most relevant chunks for a given question.

        Args:
            question (str): Your natural language question
            top_k (int): Number of relevant chunks to return (default: 3)

        Returns:
            List[str]: Top-k most relevant text chunks from the document

        Raises:
            RuntimeError: If no document has been loaded yet
            ValueError: If question is empty
        """
        self._check_ready()

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        query_vector = self.embedder.embed_single(question)
        return self.store.search(query_vector, top_k=top_k)

    def query_with_scores(self, question: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Same as query() but returns chunks with their similarity scores.

        Returns:
            List[Tuple[str, float]]: List of (chunk, distance) pairs.
                                     Lower distance = more similar.
        """
        self._check_ready()

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        query_vector = self.embedder.embed_single(question)
        return self.store.search_with_scores(query_vector, top_k=top_k)

    def _check_ready(self) -> None:
        if not self._document_loaded or self.store is None:
            raise RuntimeError(
                "No document loaded. Call load_document('path/to/file') first."
            )

    @property
    def is_ready(self) -> bool:
        """Returns True if a document has been loaded and pipeline is ready."""
        return self._document_loaded

    @property
    def source(self) -> Optional[str]:
        """Returns path of the loaded document."""
        return self._source_path

    def __repr__(self):
        status = f"source='{self._source_path}'" if self._document_loaded else "no document loaded"
        return f"RAGPipeline(model='{self.embedder.model_name}', {status})"