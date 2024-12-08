# src/rag_pipeline/rag_model.py

from .data_loader import DataLoader
from .index_builder import IndexBuilder
from .retriever import Retriever
from .generator import Generator
from .config import Config
import os
import pickle

class RAGModel:
    """
    RAGModel provides a facade over the entire retrieval and generation process.

    Methods:
        build_index(): Build the FAISS index from documents.
        query(question: str): Given a user question, retrieve relevant docs and generate an answer.
    """
    def __init__(self, config: Config):
        """
        Args:
            config (Config): Configuration object.
        """
        self.config = config
        self.data_loader = DataLoader(self.config.data_dir)

        self.index_builder = IndexBuilder(self.config.embedding_model, self.config.doc_chunk_size)
        self.generator = Generator(self.config.generation_model)

        self.chunked_docs_path = os.path.join(self.config.data_dir, "chunked_docs.pkl")

        # Load index if exists
        if os.path.exists(self.config.index_path) and os.path.exists(self.chunked_docs_path):
            self.index_builder.load_index(self.config.index_path)
            with open(self.chunked_docs_path, "rb") as f:
                self.index_builder.chunked_docs = pickle.load(f)

    def build_index(self):
        """
        Build or rebuild the FAISS index from the source documents.
        """
        docs = self.data_loader.load_documents()
        self.index_builder.build_index(docs)
        self.index_builder.save_index(self.config.index_path)

        # Save chunked docs
        with open(self.chunked_docs_path, "wb") as f:
            pickle.dump(self.index_builder.chunked_docs, f)

    def query(self, question: str):
        """
        Retrieve top-k docs relevant to the question and generate an answer.

        Args:
            question (str): The user's query.

        Returns:
            str: The generated answer.
        """
        if self.index_builder.index is None:
            raise ValueError("Index not built. Please build the index first.")

        retriever = Retriever(self.config.embedding_model, self.index_builder.index, 
                              self.index_builder.chunked_docs, self.config.top_k)
        retrieved_docs = retriever.retrieve(question)
        context = "\n\n".join(retrieved_docs)
        answer = self.generator.generate_answer(context, question)
        return answer
