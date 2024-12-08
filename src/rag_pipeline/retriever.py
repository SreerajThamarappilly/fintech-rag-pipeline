# src/rag_pipeline/retriever.py

import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    """
    Retriever uses a vector index to find top-k similar documents for a given query.

    Attributes:
        model (SentenceTransformer): Embedding model.
        index (faiss.Index): The loaded FAISS index.
        chunked_docs (List[str]): List of documents' chunks.
        top_k (int): Number of documents to retrieve.
    """
    def __init__(self, embedding_model_name: str, index, chunked_docs, top_k: int):
        """
        Args:
            embedding_model_name (str): Embedding model name or path.
            index (faiss.Index): FAISS index object.
            chunked_docs (List[str]): The chunked documents associated with the index.
            top_k (int): Number of docs to retrieve.
        """
        self.model = SentenceTransformer(embedding_model_name)
        self.index = index
        self.chunked_docs = chunked_docs
        self.top_k = top_k

    def retrieve(self, query: str):
        """
        Retrieve top-k similar documents for the query.

        Args:
            query (str): User's query string.

        Returns:
            List[str]: Top-k retrieved documents.
        """
        query_vec = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vec, self.top_k)
        # I contains indices of top_k documents
        retrieved = [self.chunked_docs[i] for i in I[0]]
        return retrieved
