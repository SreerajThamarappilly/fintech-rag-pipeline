# src/rag_pipeline/data_loader.py

import os
import glob

class DataLoader:
    """
    DataLoader is responsible for loading and reading raw documents from a specified directory.

    Methods:
        load_documents(): Loads text documents from the data directory and returns a list of strings.
    """
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir (str): Directory containing the documents.
        """
        self.data_dir = data_dir

    def load_documents(self):
        """
        Load all .txt files from the data directory and return their contents as a list of strings.

        Returns:
            List[str]: A list of document texts.
        """
        file_paths = glob.glob(os.path.join(self.data_dir, "*.txt"))
        documents = []
        for fp in file_paths:
            with open(fp, "r", encoding="utf-8") as f:
                documents.append(f.read().strip())
        return documents
