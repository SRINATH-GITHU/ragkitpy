# ragkitpy/chunker.py
"""
chunker.py — Split large text into overlapping chunks for RAG pipelines.
"""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Split text into chunks with optional overlap.

    Args:
        text (str): The full document text
        chunk_size (int): Max characters per chunk (default 500)
        overlap (int): Overlapping characters between chunks (default 50)

    Returns:
        List[str]: List of text chunks

    Example:
        >>> chunks = chunk_text("long document text...", chunk_size=200, overlap=20)
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty or whitespace.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and less than chunk_size.")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_by_sentences(text: str, sentences_per_chunk: int = 5) -> List[str]:
    """
    Alternative chunker — splits text by sentences instead of characters.

    Args:
        text (str): The full document text
        sentences_per_chunk (int): Number of sentences per chunk

    Returns:
        List[str]: List of text chunks
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        if chunk:
            chunks.append(chunk)

    return chunks