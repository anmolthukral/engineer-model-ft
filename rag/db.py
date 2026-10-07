"""PostgreSQL + pgvector database operations for RAG pipeline."""
import os
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row

from embedder import Embedder


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    file_path TEXT NOT NULL,
    heading_trail TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(%(dim)s) NOT NULL,
    token_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source, file_path, heading_trail)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_source_path_idx
    ON chunks (source, file_path);
"""


def get_dsn() -> str:
    """Get database connection string from environment."""
    dsn = os.environ.get("RAG_DATABASE_URL")
    if not dsn:
        raise RuntimeError("RAG_DATABASE_URL environment variable not set")
    return dsn


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = psycopg.connect(get_dsn(), row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(embedder: Embedder) -> None:
    """Initialize database schema with correct vector dimension."""
    dim = embedder.dimension
    statements = [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id BIGSERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            file_path TEXT NOT NULL,
            heading_trail TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR({dim}) NOT NULL,
            token_count INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (source, file_path, heading_trail)
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
            ON chunks USING hnsw (embedding vector_cosine_ops);
        """,
        """
        CREATE INDEX IF NOT EXISTS chunks_source_path_idx
            ON chunks (source, file_path);
        """,
    ]
    with get_connection() as conn:
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)


def insert_chunks(chunks: list[dict], embedder: Embedder) -> int:
    """Insert chunks with embeddings. Returns count of newly inserted rows."""
    if not chunks:
        return 0

    texts = [c["content"] for c in chunks]
    vectors = embedder.embed_batch(texts)

    inserted = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            for chunk, vector in zip(chunks, vectors):
                cur.execute(
                    """
                    INSERT INTO chunks (source, file_path, heading_trail, content, embedding, token_count)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source, file_path, heading_trail) DO NOTHING
                    """,
                    (
                        chunk["source"],
                        chunk["file_path"],
                        chunk["heading_trail"],
                        chunk["content"],
                        vector,
                        chunk.get("token_count"),
                    ),
                )
                if cur.rowcount > 0:
                    inserted += 1
    return inserted


def search_chunks(
    query_vector: list[float],
    top_k: int = 5,
    source_filter: str | None = None,
) -> list[dict]:
    """Search for similar chunks using cosine similarity."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            if source_filter:
                cur.execute(
                    """
                    SELECT source, file_path, heading_trail, content, token_count
                    FROM chunks
                    WHERE source = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (source_filter, query_vector, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT source, file_path, heading_trail, content, token_count
                    FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_vector, top_k),
                )
            return [dict(row) for row in cur.fetchall()]


def get_stats() -> dict[str, Any]:
    """Get database statistics."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as total FROM chunks")
            total = cur.fetchone()["total"]
            cur.execute(
                "SELECT source, COUNT(*) as count FROM chunks GROUP BY source"
            )
            by_source = {row["source"]: row["count"] for row in cur.fetchall()}
            cur.execute("SELECT pg_size_pretty(pg_total_relation_size('chunks')) as size")
            size = cur.fetchone()["size"]
    return {"total_chunks": total, "by_source": by_source, "table_size": size}