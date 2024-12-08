# src/rag_pipeline/index_builder.py

import numpy as np
import faiss
from .utils import chunk_text
from sentence_transformers import SentenceTransformer

class IndexBuilder:
    """
    IndexBuilder creates a FAISS vector index from documents.

    Attributes:
        embedding_model_name (str): The embedding model name or path.
        doc_chunk_size (int): Size of chunks each document is split into.
        index (faiss.Index): The FAISS index built after embedding the documents.
        chunked_docs (List[str]): List of chunked documents.
    """
    def __init__(self, embedding_model_name: str, doc_chunk_size: int):
        """
        Args:
            embedding_model_name (str): The name/path of the embedding model.
            doc_chunk_size (int): Size of text chunks.
        """
        self.embedding_model_name = embedding_model_name
        self.doc_chunk_size = doc_chunk_size
        self.model = SentenceTransformer(self.embedding_model_name)
        self.index = None
        self.chunked_docs = []

    def build_index(self, documents):
        """
        Build a FAISS index from provided documents.

        Args:
            documents (List[str]): Raw documents.

        Returns:
            faiss.Index: The built FAISS index.
        """
        # Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = chunk_text(doc, self.doc_chunk_size)
            all_chunks.extend(chunks)

        self.chunked_docs = all_chunks

        # Embed chunks
        embeddings = self.model.encode(all_chunks, convert_to_numpy=True)

        # Create FAISS index
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)
        return self.index

    def save_index(self, index_path: str):
        """
        Save FAISS index to the given path.

        Args:
            index_path (str): Path to save the FAISS index file.
        """
        faiss.write_index(self.index, index_path)

    def load_index(self, index_path: str):
        """
        Load FAISS index from the given path.

        Args:
            index_path (str): Path of the saved FAISS index.
        """
        self.index = faiss.read_index(index_path)
        # Note: chunked_docs need to be known - typically you'd store them separately.
        # For simplicity, this example doesn't show persistence of chunked_docs.
        # In a production scenario, you'd save chunked_docs as well.
