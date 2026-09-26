"""Runpod Serverless handler: retrieve-then-generate over the tone fine-tuned
model, mirroring rag/query.py's ask() so the endpoint behaves identically to
the local CLI/Streamlit flow. Talks to the Ollama server baked into this same
container image instead of localhost-only dev Ollama."""
import os

import psycopg
import requests
import runpod

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
TONE_MODEL = "engineering-tone"
TOP_K = 5


def get_connection() -> psycopg.Connection:
    return psycopg.connect(os.environ["RAG_DATABASE_URL"])


def embed(text: str) -> list[float]:
    resp = requests.post(f"{OLLAMA_URL}/api/embed", json={"model": EMBED_MODEL, "input": text})
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    vector = embed(question)
    conn = get_connection()
    try:
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
    finally:
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


def handler(event):
    question = (event.get("input") or {}).get("question", "").strip()
    if not question:
        return {"error": "input.question is required"}

    chunks = retrieve(question)
    prompt = build_prompt(question, chunks)

    # raw=True: prompt already has the "### User:/### Assistant:" markers —
    # see the matching note in rag/query.py's ask().
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": TONE_MODEL, "prompt": prompt, "stream": False, "raw": True},
        timeout=120,
    )
    resp.raise_for_status()
    answer = resp.json()["response"]

    return {
        "response": answer,
        "sources": [
            {"source": c["source"], "file_path": c["file_path"], "heading_trail": c["heading_trail"]}
            for c in chunks
        ],
    }


runpod.serverless.start({"handler": handler})
