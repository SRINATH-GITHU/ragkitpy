# tests/test_embedder.py
import pytest
import numpy as np
from ragkitpy.embedder import HFEmbedder


@pytest.fixture(scope="module")
def embedder():
    """Shared embedder instance across tests — avoids reloading model each time."""
    return HFEmbedder()


def test_embed_returns_numpy_array(embedder):
    result = embedder.embed(["Hello world"])
    assert isinstance(result, np.ndarray)


def test_embed_shape(embedder):
    texts = ["First chunk", "Second chunk", "Third chunk"]
    result = embedder.embed(texts)
    assert result.shape[0] == 3       # 3 texts
    assert result.shape[1] == 384     # MiniLM produces 384-dim vectors


def test_embed_single(embedder):
    result = embedder.embed_single("test query")
    assert result.ndim == 1           # Should be 1D
    assert len(result) == 384


def test_embed_empty_list_raises(embedder):
    with pytest.raises(ValueError):
        embedder.embed([])


def test_embedding_dim_property(embedder):
    assert embedder.embedding_dim == 384