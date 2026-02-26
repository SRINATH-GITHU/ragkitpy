# tests/test_vectorstore.py
import pytest
import numpy as np
from ragkitpy.vectorstore import VectorStore


@pytest.fixture
def sample_store():
    store = VectorStore(dim=4)   # Small dim for testing
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    chunks = ["chunk about python", "chunk about RAG", "chunk about embeddings"]
    store.add(embeddings, chunks)
    return store


def test_add_chunks(sample_store):
    assert sample_store.total_chunks == 3


def test_search_returns_results(sample_store):
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    results = sample_store.search(query, top_k=1)
    assert len(results) == 1
    assert results[0] == "chunk about python"   # Most similar to query


def test_search_top_k(sample_store):
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    results = sample_store.search(query, top_k=3)
    assert len(results) == 3


def test_search_with_scores(sample_store):
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    results = sample_store.search_with_scores(query, top_k=2)
    assert len(results) == 2
    assert isinstance(results[0][1], float)    # Score should be float


def test_search_empty_store_raises():
    store = VectorStore(dim=4)
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(RuntimeError):
        store.search(query)


def test_mismatch_raises():
    store = VectorStore(dim=4)
    embeddings = np.zeros((3, 4), dtype=np.float32)
    chunks = ["only one chunk"]               # Mismatch: 3 embeddings, 1 chunk
    with pytest.raises(ValueError):
        store.add(embeddings, chunks)