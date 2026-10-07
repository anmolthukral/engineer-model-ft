#!/usr/bin/env python3
"""Test query script for RAG pipeline.

Usage:
    python -m rag.query "your question here"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import search_chunks
from embedder import get_embedder


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m rag.query '<question>' [--top-k N] [--source SOURCE]")
        sys.exit(1)

    # Parse args
    top_k = 5
    source_filter = None
    question_parts = []

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--top-k" and i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1])
            i += 2
        elif arg == "--source" and i + 1 < len(sys.argv):
            source_filter = sys.argv[i + 1]
            i += 2
        else:
            question_parts.append(arg)
            i += 1

    question = " ".join(question_parts)

    print(f"Question: {question}")
    print(f"Top-K: {top_k}")
    if source_filter:
        print(f"Source filter: {source_filter}")

    embedder = get_embedder()
    query_vector = embedder.embed(question)

    results = search_chunks(query_vector, top_k=top_k, source_filter=source_filter)

    if not results:
        print("\nNo results found.")
        return

    print(f"\nFound {len(results)} relevant chunks:\n")
    for j, chunk in enumerate(results, 1):
        print(f"--- Result {j} ---")
        print(f"Source: {chunk['source']}")
        print(f"File: {chunk['file_path']}")
        print(f"Heading: {chunk['heading_trail']}")
        print(f"Tokens: {chunk.get('token_count', 'N/A')}")
        print(f"Content:\n{chunk['content'][:500]}...")
        print()


if __name__ == "__main__":
    main()