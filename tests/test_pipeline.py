# tests/test_pipeline.py
import sys, os
# ensure local package is imported instead of any installed version
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import tempfile
from ragkitpy.pipeline import RAGPipeline


@pytest.fixture(scope="module")
def sample_txt_file():
    """Create a temp .txt file with enough content for RAG testing."""
    content = """
    Artificial intelligence is transforming industries worldwide.
    Machine learning models are trained on large datasets to recognize patterns.
    Natural language processing helps computers understand human language.
    RAG stands for Retrieval Augmented Generation, a powerful technique.
    It combines document retrieval with language model generation.
    Vector databases store embeddings for fast similarity search.
    HuggingFace provides open source models for NLP tasks.
    Sentence transformers convert sentences into dense vector representations.
    FAISS is a library for efficient similarity search developed by Meta.
    Python is the most popular programming language for AI development.
    """ * 5  # Repeat to have enough chunks

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt', delete=False, encoding='utf-8'
    ) as f:
        f.write(content)
        return f.name


@pytest.fixture(scope="module")
def loaded_pipeline(sample_txt_file):
    """Pipeline with document already loaded — shared across tests."""
    rag = RAGPipeline(chunk_size=200, overlap=20)
    rag.load_document(sample_txt_file)
    return rag


def test_pipeline_not_ready_by_default():
    rag = RAGPipeline()
    assert rag.is_ready is False


def test_load_document(loaded_pipeline):
    assert loaded_pipeline.is_ready is True


def test_source_path_set(loaded_pipeline, sample_txt_file):
    assert loaded_pipeline.source == sample_txt_file


def test_query_returns_list(loaded_pipeline):
    results = loaded_pipeline.query("What is RAG?")
    assert isinstance(results, list)
    assert len(results) > 0


def test_query_top_k(loaded_pipeline):
    results = loaded_pipeline.query("What is machine learning?", top_k=2)
    assert len(results) == 2


def test_query_with_scores(loaded_pipeline):
    results = loaded_pipeline.query_with_scores("HuggingFace models", top_k=2)
    assert len(results) == 2
    assert isinstance(results[0], tuple)
    assert isinstance(results[0][1], float)   # Score is float


def test_query_before_load_raises():
    rag = RAGPipeline()
    with pytest.raises(RuntimeError):
        rag.query("anything")


def test_empty_query_raises(loaded_pipeline):
    with pytest.raises(ValueError):
        loaded_pipeline.query("")


def test_repr(loaded_pipeline):
    r = repr(loaded_pipeline)
    assert "RAGPipeline" in r


def test_answer_returns_string(loaded_pipeline):
    # skip if transformers isn't available; optional dependency
    pytest.importorskip("transformers")
    # basic smoke test for generation; should not error and produce text
    ans = loaded_pipeline.answer("What is RAG?")
    assert isinstance(ans, str)
    assert len(ans) > 0


def test_answer_before_load_raises():
    rag = RAGPipeline()
    with pytest.raises(RuntimeError):
        rag.answer("Is anyone there?")


def test_empty_answer_question_raises(loaded_pipeline):
    with pytest.raises(ValueError):
        loaded_pipeline.answer("")