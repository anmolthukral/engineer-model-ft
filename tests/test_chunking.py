import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.chunking import chunk_markdown


class FakeTokenizer:
    """Deterministic stand-in so the test doesn't need to load the real
    tokenizer: token count is just whitespace-split word count."""
    def encode(self, text):
        return text.split()


FIXTURE_MD = """# Array.prototype.flat()

Some intro text about flattening arrays with a handful of words to pad this out to a reasonable size for the test fixture.

## Syntax

flat() takes an optional depth argument. Here is more filler text describing the syntax in a bit more detail so this section has real content.

## Examples

### Flattening one level

Example text here.

### Flattening infinitely

More example text here describing the infinite flattening behavior in some more detail.
"""


def test_splits_by_heading_and_tracks_trail():
    chunks = chunk_markdown(FIXTURE_MD, source="mdn", file_path="array/flat.md",
                             tokenizer=FakeTokenizer(), min_tokens=1, max_tokens=1000)
    assert len(chunks) >= 1
    for c in chunks:
        assert c["source"] == "mdn"
        assert c["file_path"] == "array/flat.md"
        assert c["content"].strip() != ""
        assert c["token_count"] == len(c["content"].split())
        assert isinstance(c["heading_trail"], str) and c["heading_trail"] != ""


def test_small_sections_get_merged_up_to_min_tokens():
    chunks = chunk_markdown(FIXTURE_MD, source="mdn", file_path="array/flat.md",
                             tokenizer=FakeTokenizer(), min_tokens=40, max_tokens=1000)
    # With a high min_tokens, the small subsections should merge into fewer,
    # larger chunks rather than staying as tiny fragments.
    assert all(c["token_count"] >= 5 for c in chunks)  # no tiny leftover fragments
    assert len(chunks) < 5  # fewer chunks than raw heading count (5 headings)


def test_oversized_sections_get_split_by_max_tokens():
    long_section = "# Big Section\n\n" + " ".join(f"word{i}" for i in range(500))
    chunks = chunk_markdown(long_section, source="mdn", file_path="big.md",
                             tokenizer=FakeTokenizer(), min_tokens=1, max_tokens=100)
    assert len(chunks) > 1
    for c in chunks:
        assert c["token_count"] <= 100


def test_heading_trail_reflects_nesting():
    chunks = chunk_markdown(FIXTURE_MD, source="mdn", file_path="array/flat.md",
                             tokenizer=FakeTokenizer(), min_tokens=1, max_tokens=1000)
    trails = [c["heading_trail"] for c in chunks]
    # At least one chunk should carry a nested trail like "Examples > Flattening one level"
    assert any(">" in t for t in trails)


def test_multipiece_splits_get_distinct_trailing_indices():
    """Verify that when a single oversized section splits into multiple pieces,
    each piece gets a unique heading_trail (with part index) to avoid collisions
    in downstream UNIQUE(source, file_path, heading_trail) constraints."""
    long_section = "# Big Section\n\n" + " ".join(f"word{i}" for i in range(500))
    chunks = chunk_markdown(long_section, source="mdn", file_path="big.md",
                             tokenizer=FakeTokenizer(), min_tokens=1, max_tokens=100)

    # Must have multiple chunks from the split
    assert len(chunks) > 1

    # All chunks from this section must have distinct heading_trail values
    trails = [c["heading_trail"] for c in chunks]
    assert len(trails) == len(set(trails)), f"Duplicate heading_trail values: {trails}"

    # Each trail should indicate its part number (except if only one piece)
    for trail in trails:
        assert "part" in trail.lower() or len(chunks) == 1, \
            f"Expected part number in trail '{trail}' for multi-piece split"
