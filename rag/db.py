"""Shared Postgres connection helper for the RAG pipeline."""
import os
import psycopg


def get_connection() -> psycopg.Connection:
    dsn = os.environ.get("RAG_DATABASE_URL", "dbname=engineer_rag")
    return psycopg.connect(dsn)
