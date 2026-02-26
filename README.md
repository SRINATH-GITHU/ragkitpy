# ragkitpy 🤗

[![PyPI version](https://badge.fury.io/py/ragkitpy.svg)](https://badge.fury.io/py/ragkitpy)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, beginner-friendly RAG (Retrieval-Augmented Generation) pipeline
toolkit powered by HuggingFace 🤗 — no OpenAI key required.

---

## 🚀 Why ragkitpy?

Setting up a RAG pipeline from scratch is painful. LangChain and LlamaIndex are
powerful but heavy and complex for beginners. **ragkitpy** gives you a working
RAG pipeline in 5 lines of code using only open-source models.

- ✅ No OpenAI API key needed — runs fully local with HuggingFace
- ✅ Supports PDF and TXT files
- ✅ Built on FAISS for fast vector search
- ✅ Beginner-friendly API
- ✅ Lightweight — minimal dependencies

---

## 📦 Installation
```bash
pip install ragkitpy
```

---

## ⚡ Quick Start
```python
from ragkitpy import RAGPipeline

# Initialize
rag = RAGPipeline()

# Load your document (PDF or TXT)
rag.load_document("my_document.pdf")

# Query it
results = rag.query("What is this document about?")

for i, chunk in enumerate(results, 1):
    print(f"Result {i}: {chunk}")
```

That's it. 5 lines to a working RAG pipeline. 🎯

---

## 🧠 How It Works
```
Your Document (PDF/TXT)
        ↓
   [loader.py]  →  Extract raw text
        ↓
  [chunker.py]  →  Split into overlapping chunks
        ↓
  [embedder.py] →  HuggingFace embeddings (all-MiniLM-L6-v2)
        ↓
[vectorstore.py]→  Store in FAISS index
        ↓
   Your Query   →  Embed query → Search FAISS → Return top-k chunks
```

---

## 📖 Full API Reference

### `RAGPipeline`
```python
from ragkitpy import RAGPipeline

rag = RAGPipeline(
    model_name="all-MiniLM-L6-v2",  # Any SentenceTransformer model
    chunk_size=500,                   # Characters per chunk
    overlap=50                        # Overlap between chunks
)
```

| Method | Description |
|--------|-------------|
| `rag.load_document(path)` | Load a PDF or TXT file |
| `rag.query(question, top_k=3)` | Get top-k relevant chunks |
| `rag.query_with_scores(question, top_k=3)` | Get chunks with similarity scores |
| `rag.is_ready` | Check if document is loaded |
| `rag.source` | Path of loaded document |

---

### Use a different HuggingFace model
```python
# Larger model, better quality
rag = RAGPipeline(model_name="all-mpnet-base-v2")
```

### Get similarity scores
```python
results = rag.query_with_scores("What is machine learning?", top_k=3)
for chunk, score in results:
    print(f"Score: {score:.4f} | {chunk[:100]}")
```

### Use individual modules
```python
from ragkitpy import load_file, chunk_text, HFEmbedder, VectorStore

# Use just the loader
text = load_file("document.pdf")

# Use just the chunker
chunks = chunk_text(text, chunk_size=300, overlap=30)

# Use sentence chunker instead
from ragkitpy.chunker import chunk_by_sentences
chunks = chunk_by_sentences(text, sentences_per_chunk=3)
```

---

## 🗂️ Project Structure
```
ragkitpy/
├── ragkitpy/
│   ├── __init__.py       # Public API
│   ├── loader.py         # PDF & TXT file loading
│   ├── chunker.py        # Text chunking strategies
│   ├── embedder.py       # HuggingFace embeddings wrapper
│   ├── vectorstore.py    # FAISS vector store
│   └── pipeline.py       # End-to-end RAG pipeline
├── tests/                # 24 unit tests
├── examples/             # Working examples
└── pyproject.toml
```

---

## 🔧 Dependencies

| Package | Purpose |
|---------|---------|
| `sentence-transformers` | HuggingFace text embeddings |
| `faiss-cpu` | Vector similarity search |
| `pypdf` | PDF text extraction |
| `numpy` | Array operations |

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.
```bash
git clone https://github.com/YOURUSERNAME/ragkitpy.git
cd ragkitpy
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

Built by **Your Name** as an open-source contribution to the GenAI community.

⭐ Star this repo if it helped you!
```

---

Also write your `LICENSE` file (MIT):
```
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.