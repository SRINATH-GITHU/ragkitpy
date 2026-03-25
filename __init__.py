# ragkitpy/__init__.py
"""
ragkitpy — A lightweight RAG pipeline toolkit powered by HuggingFace 🤗

Quick start:
    >>> from ragkitpy import RAGPipeline
    >>> rag = RAGPipeline()
    >>> rag.load_document("my_file.pdf")
    >>> results = rag.query("What is this about?")
"""

from ragkitpy.pipeline import RAGPipeline
from ragkitpy.loader import load_file
from ragkitpy.chunker import chunk_text
from ragkitpy.embedder import HFEmbedder
from ragkitpy.vectorstore import VectorStore

__version__ = "0.1.1"
__author__ = "Srinath Dhumnor"

__all__ = [
    "RAGPipeline",
    "load_file",
    "chunk_text",
    "HFEmbedder",
    "VectorStore",
]