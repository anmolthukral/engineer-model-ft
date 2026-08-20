#!/usr/bin/env python3
"""Read rag/corpus.jsonl, chunk each document, embed each chunk via
Ollama's local nomic-embed-text model, and store into pgvector. Uses
ON CONFLICT DO NOTHING against the (source, file_path, heading_trail)
unique constraint so re-running after an interruption is safe."""
import json
import sys
from pathlib import Path

import requests
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.chunking import chunk_markdown
from rag.db import get_connection

CORPUS_FILE = Path("./rag/corpus.jsonl")
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


def embed(text: str) -> list[float]:
    resp = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "input": text})
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def main():
    tokenizer = AutoTokenizer.from_pretrained("./mlx_model/base_model")
    conn = get_connection()
    cur = conn.cursor()

    total_docs, total_chunks, inserted = 0, 0, 0
    with open(CORPUS_FILE) as f:
        for line in f:
            doc = json.loads(line)
            total_docs += 1
            chunks = chunk_markdown(
                doc["raw_markdown"], source=doc["source"], file_path=doc["file_path"],
                tokenizer=tokenizer, min_tokens=400, max_tokens=700,
            )
            for chunk in chunks:
                total_chunks += 1
                vector = embed(chunk["content"])
                cur.execute(
                    """
                    INSERT INTO chunks (source, file_path, heading_trail, content, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (source, file_path, heading_trail) DO NOTHING
                    """,
                    (chunk["source"], chunk["file_path"], chunk["heading_trail"],
                     chunk["content"], vector),
                )
                if cur.rowcount > 0:
                    inserted += 1
            if total_docs % 200 == 0:
                conn.commit()
                print(f"  ...{total_docs} docs processed, {inserted} chunks inserted so far")

    conn.commit()
    cur.close()
    conn.close()
    print(f"\nDone. {total_docs} documents -> {total_chunks} chunks -> {inserted} newly inserted rows.")


if __name__ == "__main__":
    main()
