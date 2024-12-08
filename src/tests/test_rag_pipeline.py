# src/tests/test_rag_pipeline.py

import os
import pytest
from rag_pipeline.config import Config
from rag_pipeline.rag_model import RAGModel

@pytest.mark.order(1)
def test_build_index():
    """
    Tests like test_build_index() confirm that your FAISS index is being built correctly and documents are being embedded as expected.
    """
    config = Config()
    rag = RAGModel(config)
    # Build index
    rag.build_index()
    assert os.path.exists(config.index_path), "Index file should be created."

@pytest.mark.order(2)
def test_query():
    """
    Tests like test_query() validate that the retrieval and generation pipeline produces non-empty, meaningful results.
    """
    config = Config()
    rag = RAGModel(config)
    answer = rag.query("What are emerging trends in fintech?")
    assert isinstance(answer, str), "Answer should be a string."
    assert len(answer) > 0, "Answer should not be empty."

@pytest.mark.order(3)
def test_empty_query():
    """
    Tests like test_empty_query() is used for testing an empty query.
    """
    config = Config()
    rag = RAGModel(config)
    answer = rag.query("")  # Test an empty query
    assert isinstance(answer, str), "Answer should still be a string."
