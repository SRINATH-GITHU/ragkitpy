# tests/test_loader.py
import pytest
from ragkitpy.loader import load_txt, load_file
import tempfile
import os


def test_load_txt_basic():
    # Create a temp txt file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Hello from ragkitpy!")
        tmp_path = f.name

    result = load_txt(tmp_path)
    assert result == "Hello from ragkitpy!"
    os.unlink(tmp_path)


def test_load_file_txt():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content")
        tmp_path = f.name

    result = load_file(tmp_path)
    assert "Test content" in result
    os.unlink(tmp_path)


def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_file("nonexistent_file.txt")


def test_load_file_unsupported_type():
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        tmp_path = f.name

    with pytest.raises(ValueError):
        load_file(tmp_path)
    os.unlink(tmp_path)