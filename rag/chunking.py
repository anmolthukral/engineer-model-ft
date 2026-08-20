"""Heading-bounded markdown chunker for the RAG ingestion pipeline.

Splits a markdown document into sections by heading (# through ######),
merges consecutive small sections up to min_tokens, and splits any
section larger than max_tokens by paragraph. Every chunk carries the
"heading trail" (e.g. "Examples > Flattening one level") it came from,
for citation and debugging.
"""
import re

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _split_into_sections(markdown_text: str):
    """Return list of (level, title, body) walking the document in order."""
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
        # No headings at all: treat the whole doc as one untitled section.
        sections = [(1, "", markdown_text.strip())]
    return sections


def _heading_trail(stack, title):
    return " > ".join([t for _, t in stack] + ([title] if title else []))


def _split_oversized(text: str, tokenizer, max_tokens: int):
    """Split a too-large block of text by paragraph, greedily filling up
    to max_tokens per piece. If a single paragraph exceeds max_tokens,
    split it further by words."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    pieces, current, current_tokens = [], [], 0
    for p in paragraphs:
        p_tokens = len(tokenizer.encode(p))

        # If a single paragraph exceeds max_tokens, split it by words
        if p_tokens > max_tokens:
            # Flush current buffer first
            if current:
                pieces.append("\n\n".join(current))
                current, current_tokens = [], 0

            # Split oversized paragraph by words
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
            # Normal paragraph that fits within max_tokens
            if current and current_tokens + p_tokens > max_tokens:
                pieces.append("\n\n".join(current))
                current, current_tokens = [], 0
            current.append(p)
            current_tokens += p_tokens

    if current:
        pieces.append("\n\n".join(current))
    return pieces or [text]


def chunk_markdown(markdown_text: str, source: str, file_path: str, tokenizer,
                    min_tokens: int = 400, max_tokens: int = 700) -> list[dict]:
    sections = _split_into_sections(markdown_text)

    # Build heading trail per section using a stack keyed by heading level.
    stack = []
    trailed_sections = []
    for level, title, body in sections:
        while stack and stack[-1][0] >= level:
            stack.pop()
        if title:
            stack.append((level, title))
        trailed_sections.append((_heading_trail(stack, "" if title else ""), title, body))

    chunks = []
    buffer_trail, buffer_parts, buffer_tokens = None, [], 0

    def flush():
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
            flush()  # oversized section starts its own chunk(s), don't merge with buffer
            pieces = _split_oversized(body, tokenizer, max_tokens)
            for i, piece in enumerate(pieces):
                # Append part index only if this section was split into multiple pieces
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
