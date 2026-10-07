#!/usr/bin/env python3
"""RAG ingestion pipeline for Engineer Playbook content.

Reads markdown/MDX files from configured source directories,
chunks with heading hierarchy preserved, generates embeddings,
and stores in PostgreSQL with pgvector.

Usage:
    python -m rag.ingest                    # Run full ingestion
    python -m rag.ingest --source blogs     # Ingest only blogs
    python -m rag.ingest --dry-run          # Show what would be processed
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunker import chunk_markdown, read_markdown_files
from db import get_dsn, init_db, insert_chunks, get_stats
from embedder import get_embedder


DEFAULT_SOURCES = [
    Path("/Users/anmolthukral/projects/megamind/tutorials/app"),
    Path("/Users/anmolthukral/projects/megamind/tutorials/src"),
    Path("/Users/anmolthukral/projects/megamind/blogs/content"),
]


def ingest(sources: list[Path], dry_run: bool = False) -> None:
    """Run the ingestion pipeline."""
    print(f"Connecting to database: {get_dsn()}")
    embedder = get_embedder()
    print(f"Using embedder: {embedder.__class__.__name__} (dim={embedder.dimension})")

    if not dry_run:
        init_db(embedder)
        print("Database schema initialized")

    documents = read_markdown_files(sources)
    print(f"Found {len(documents)} markdown files")

    if dry_run:
        for doc in documents:
            chunks = chunk_markdown(
                doc["raw_markdown"],
                source=doc["source"],
                file_path=doc["file_path"],
            )
            print(f"  {doc['source']}/{doc['file_path']}: {len(chunks)} chunks")
        return

    total_chunks = 0
    total_inserted = 0

    for i, doc in enumerate(documents, 1):
        chunks = chunk_markdown(
            doc["raw_markdown"],
            source=doc["source"],
            file_path=doc["file_path"],
        )
        inserted = insert_chunks(chunks, embedder)
        total_chunks += len(chunks)
        total_inserted += inserted

        if i % 10 == 0 or i == len(documents):
            print(f"  [{i}/{len(documents)}] {doc['source']}/{doc['file_path']}: "
                  f"{len(chunks)} chunks, {inserted} new")

    print(f"\nDone. {len(documents)} documents -> {total_chunks} chunks -> {total_inserted} newly inserted")

    stats = get_stats()
    print(f"Database: {stats['total_chunks']} total chunks, {stats['table_size']}")
    for src, count in stats['by_source'].items():
        print(f"  {src}: {count} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG ingestion pipeline")
    parser.add_argument(
        "--source",
        choices=["tutorials", "blogs", "all"],
        default="all",
        help="Which source to ingest (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without writing to database",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        type=Path,
        help="Custom paths to ingest (overrides --source)",
    )
    args = parser.parse_args()

    if args.paths:
        sources = args.paths
    elif args.source == "tutorials":
        sources = DEFAULT_SOURCES[:2]
    elif args.source == "blogs":
        sources = DEFAULT_SOURCES[2:]
    else:
        sources = DEFAULT_SOURCES

    ingest(sources, dry_run=args.dry_run)


if __name__ == "__main__":
    main()