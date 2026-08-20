#!/usr/bin/env python3
"""Retrieve-then-generate RAG query script: embed the question, pull the
top-5 most relevant doc chunks from pgvector, and ask the tone-fine-tuned
model to answer using them as context."""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.db import get_connection

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
EMBED_MODEL = "nomic-embed-text"
TONE_MODEL = "engineering-tone"
TOP_K = 5


def embed(text: str) -> list[float]:
    resp = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "input": text})
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    vector = embed(question)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT source, file_path, heading_trail, content
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (vector, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {"source": r[0], "file_path": r[1], "heading_trail": r[2], "content": r[3]}
        for r in rows
    ]


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = "\n\n".join(
        f"[{c['source']} — {c['heading_trail']}]\n{c['content']}" for c in chunks
    )
    return (
        f"### User:\nUse the following documentation excerpts to answer the question. "
        f"If the excerpts don't cover it, say so rather than guessing.\n\n"
        f"{context_blocks}\n\nQuestion: {question}\n### Assistant:\n"
    )


def ask(question: str) -> tuple[str, list[dict]]:
    chunks = retrieve(question)
    prompt = build_prompt(question, chunks)
    # raw=True is required here: build_prompt() already includes the full
    # "### User:/### Assistant:" markers itself. Without raw=True, Ollama
    # would additionally apply Modelfile.tone's own TEMPLATE on top of this
    # already-formatted string, double-wrapping the markers and producing
    # a garbled prompt the model was never trained on.
    resp = requests.post(
        OLLAMA_GENERATE_URL,
        json={"model": TONE_MODEL, "prompt": prompt, "stream": False, "raw": True},
    )
    resp.raise_for_status()
    return resp.json()["response"], chunks


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 rag/query.py '<question>'")
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    answer, chunks = ask(question)
    print(answer)
    print("\n--- sources ---")
    for c in chunks:
        print(f"  {c['source']}: {c['file_path']} ({c['heading_trail']})")


if __name__ == "__main__":
    main()
