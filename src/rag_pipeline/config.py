# src/rag_pipeline/config.py

import os
from dotenv import load_dotenv

class Config:
    """
    Config class that loads environment variables and holds configuration values.

    Attributes:
        embedding_model (str): Name or path to the embedding model.
        generation_model (str): Name or path to the generation model.
        top_k (int): Number of documents to retrieve.
        doc_chunk_size (int): Size of text chunks.
        data_dir (str): Directory of the data.
        index_path (str): Path to save/load the FAISS index.
    """
    def __init__(self):
        load_dotenv()
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.generation_model = os.getenv("GENERATION_MODEL", "distilgpt2")
        self.top_k = int(os.getenv("TOP_K", 3))
        self.doc_chunk_size = int(os.getenv("DOC_CHUNK_SIZE", 512))
        self.data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "docs")
        self.index_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "faiss_index.bin")
