# src/main.py

import argparse
from rag_pipeline.config import Config
from rag_pipeline.rag_model import RAGModel

def main():
    """
    Main entry point for the RAG pipeline application.

    This script can build/update the FAISS index or run a query against the index.
    """
    parser = argparse.ArgumentParser(description="Fintech RAG Pipeline")
    parser.add_argument("--action", type=str, choices=["build_index", "query"], required=True,
                        help="Action to perform: build_index or query")
    parser.add_argument("--question", type=str, help="User query when action=query")

    args = parser.parse_args()
    config = Config()
    rag = RAGModel(config)

    if args.action == "build_index":
        print("Building or updating the index...")
        rag.build_index()
        print("Index built successfully.")
    elif args.action == "query":
        if not args.question:
            print("Please provide a --question for the query action.")
            return
        print(f"Querying: {args.question}")
        answer = rag.query(args.question)
        print("Generated Answer:\n", answer)

if __name__ == "__main__":
    main()
