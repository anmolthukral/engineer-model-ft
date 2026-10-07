"""Markdown chunker with heading hierarchy preservation for RAG ingestion.

Splits markdown/MDX documents into chunks bounded by headings,
preserving the full heading trail (e.g. "Getting Started > Installation > Requirements")
for citation and context.
"""
import re
from pathlib import Path
from typing import Any

try:
    import tiktoken
except ImportError:
    tiktoken = None

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


class Tokenizer:
    """Wrapper for tiktoken encoder with fallback."""

    def __init__(self, model: str = "cl100k_base"):
        if tiktoken:
            self.encoder = tiktoken.get_encoding(model)
        else:
            self.encoder = None

    def encode(self, text: str) -> list[int]:
        if self.encoder:
            return self.encoder.encode(text)
        # Fallback: rough word-based tokenization
        return text.split()

    def decode(self, tokens: list[int]) -> str:
        if self.encoder:
            return self.encoder.decode(tokens)
        return " ".join(str(t) for t in tokens)


def _split_into_sections(markdown_text: str) -> list[tuple[int, str, str]]:
    """Split markdown into sections by headings. Returns (level, title, body)."""
    matches = list(HEADING_RE.finditer(markdown_text))
    sections = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        body = markdown_text[start:end].strip()
        sections.append((level, title, body))
    if not sections:
        sections = [(1, "", markdown_text.strip())]
    return sections


def _heading_trail(stack: list[tuple[int, str]], title: str) -> str:
    """Build heading trail from stack."""
    return " > ".join([t for _, t in stack] + ([title] if title else []))


def _split_oversized(text: str, tokenizer: Tokenizer, max_tokens: int) -> list[str]:
    """Split oversized text by paragraph, then by words if needed."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    pieces, current, current_tokens = [], [], 0

    for p in paragraphs:
        p_tokens = len(tokenizer.encode(p))

        if p_tokens > max_tokens:
            if current:
                pieces.append("\n\n".join(current))
                current, current_tokens = [], 0

            words = p.split()
            word_buffer, word_tokens = [], 0
            for word in words:
                word_token_count = len(tokenizer.encode(word))
                if word_buffer and word_tokens + word_token_count > max_tokens:
                    pieces.append(" ".join(word_buffer))
                    word_buffer, word_tokens = [], 0
                word_buffer.append(word)
                word_tokens += word_token_count
            if word_buffer:
                pieces.append(" ".join(word_buffer))
        else:
            if current and current_tokens + p_tokens > max_tokens:
                pieces.append("\n\n".join(current))
                current, current_tokens = [], 0
            current.append(p)
            current_tokens += p_tokens

    if current:
        pieces.append("\n\n".join(current))
    return pieces or [text]


def chunk_markdown(
    markdown_text: str,
    source: str,
    file_path: str,
    tokenizer: Tokenizer | None = None,
    min_tokens: int = 400,
    max_tokens: int = 700,
) -> list[dict[str, Any]]:
    """Chunk markdown text preserving heading hierarchy.

    Args:
        markdown_text: Raw markdown/MDX content
        source: Source identifier (e.g., "tutorials", "blogs")
        file_path: Relative path of the source file
        tokenizer: Tokenizer instance (creates default if None)
        min_tokens: Minimum tokens per chunk before flush
        max_tokens: Maximum tokens per chunk

    Returns:
        List of chunk dicts with keys: source, file_path, heading_trail, content, token_count
    """
    if tokenizer is None:
        tokenizer = Tokenizer()

    sections = _split_into_sections(markdown_text)

    # Build heading trail per section using a stack keyed by heading level
    stack = []
    trailed_sections = []
    for level, title, body in sections:
        while stack and stack[-1][0] >= level:
            stack.pop()
        if title:
            stack.append((level, title))
        trailed_sections.append((_heading_trail(stack, "" if title else ""), title, body))

    chunks = []
    buffer_trail: str | None = None
    buffer_parts: list[str] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer_trail, buffer_parts, buffer_tokens
        if buffer_parts:
            content = "\n\n".join(buffer_parts).strip()
            if content:
                chunks.append({
                    "source": source,
                    "file_path": file_path,
                    "heading_trail": buffer_trail or "(untitled)",
                    "content": content,
                    "token_count": len(tokenizer.encode(content)),
                })
        buffer_trail, buffer_parts, buffer_tokens = None, [], 0

    for trail, title, body in trailed_sections:
        if not body:
            continue
        body_tokens = len(tokenizer.encode(body))

        if body_tokens > max_tokens:
            flush()
            pieces = _split_oversized(body, tokenizer, max_tokens)
            for i, piece in enumerate(pieces):
                if len(pieces) > 1:
                    part_trail = f"{trail or '(untitled)'} (part {i+1}/{len(pieces)})"
                else:
                    part_trail = trail or "(untitled)"

                chunks.append({
                    "source": source,
                    "file_path": file_path,
                    "heading_trail": part_trail,
                    "content": piece.strip(),
                    "token_count": len(tokenizer.encode(piece)),
                })
            continue

        if buffer_tokens + body_tokens > max_tokens and buffer_tokens >= min_tokens:
            flush()

        if buffer_trail is None:
            buffer_trail = trail
        buffer_parts.append(body)
        buffer_tokens += body_tokens

        if buffer_tokens >= min_tokens:
            flush()

    flush()
    return chunks


def read_markdown_files(root_paths: list[Path]) -> list[dict[str, Any]]:
    """Read all .md and .mdx files from given root paths.

    Returns list of dicts with: source, file_path, raw_markdown
    """
    documents = []
    for root in root_paths:
        if not root.exists():
            continue
        source = root.name
        for ext in ("*.md", "*.mdx"):
            for file_path in root.rglob(ext):
                try:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if not text.strip():
                    continue
                rel_path = str(file_path.relative_to(root))
                documents.append({
                    "source": source,
                    "file_path": rel_path,
                    "raw_markdown": text,
                })
    return documents