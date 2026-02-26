# ragkitpy/loader.py
"""
loader.py — Load text content from TXT and PDF files.
"""

import os


def load_txt(path: str) -> str:
    """Read and return content from a .txt file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path: str) -> str:
    """Extract and return text from a .pdf file."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required for PDF support. Run: pip install pypdf")
    
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def load_file(path: str) -> str:
    """
    Auto-detect file type and load content.
    
    Supported: .txt, .pdf
    
    Args:
        path (str): Path to the file
        
    Returns:
        str: Extracted text content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is not supported
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    ext = os.path.splitext(path)[-1].lower()
    
    loaders = {
        ".txt": load_txt,
        ".pdf": load_pdf,
    }
    
    if ext not in loaders:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported types: {list(loaders.keys())}"
        )
    
    return loaders[ext](path)