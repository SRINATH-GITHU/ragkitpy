# examples/basic_rag.py
"""
Basic RAG example using ragkitpy.

Usage:
    python examples/basic_rag.py
"""

from ragkitpy.pipeline import RAGPipeline
import tempfile
import os

# --- Create a sample document ---
sample_text = """
Artificial intelligence is revolutionizing how we interact with technology.
RAG (Retrieval Augmented Generation) is a technique that combines document
retrieval with language model generation to produce grounded, accurate answers.

HuggingFace is an open-source platform that provides thousands of pre-trained
models for NLP, computer vision, and more. Sentence Transformers is one of
their most popular libraries for generating text embeddings.

FAISS (Facebook AI Similarity Search) is a library for efficient vector
similarity search, commonly used in RAG pipelines as the vector store.

Python continues to dominate the AI/ML landscape due to its simplicity
and the richness of its ecosystem.
"""

# Save to a temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
    f.write(sample_text)
    tmp_path = f.name

# --- RAG in 5 lines ---
rag = RAGPipeline()
rag.load_document(tmp_path)

questions = [
    "What is RAG?",
    "What is HuggingFace used for?",
    "What library is used for vector search?",
]

for question in questions:
    print(f"\n❓ Question: {question}")
    print("─" * 50)
    results = rag.query(question, top_k=2)
    for i, chunk in enumerate(results, 1):
        print(f"📌 Result {i}: {chunk[:200]}...")

# Cleanup
os.unlink(tmp_path)