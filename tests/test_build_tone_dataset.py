import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_tone_dataset import TONE_EXAMPLES, build_dataset


def test_tone_examples_are_well_formed():
    assert len(TONE_EXAMPLES) >= 30
    for ex in TONE_EXAMPLES:
        assert set(ex.keys()) == {"text"}
        assert "### User:" in ex["text"]
        assert "### Assistant:" in ex["text"]
        # User section must come before Assistant section
        assert ex["text"].index("### User:") < ex["text"].index("### Assistant:")


def test_tone_examples_cover_varied_topics():
    # A tone-only dataset must not be dominated by one subject (that would
    # re-teach technical facts instead of voice). No single word from this
    # set of topic markers should appear in more than a third of examples.
    topic_markers = ["react", "javascript", "typescript", "component", "hook"]
    n = len(TONE_EXAMPLES)
    for marker in topic_markers:
        count = sum(1 for ex in TONE_EXAMPLES if marker in ex["text"].lower())
        assert count <= n // 3, f"'{marker}' appears in too many examples ({count}/{n})"


def test_build_dataset_writes_three_splits(tmp_path):
    build_dataset(str(tmp_path))
    train = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    valid = [json.loads(l) for l in (tmp_path / "valid.jsonl").read_text().splitlines()]
    test = [json.loads(l) for l in (tmp_path / "test.jsonl").read_text().splitlines()]

    total = len(train) + len(valid) + len(test)
    assert total == len(TONE_EXAMPLES)
    assert len(valid) >= 1
    assert len(test) >= 1
    # every written example has the same shape as TONE_EXAMPLES
    for ex in train + valid + test:
        assert set(ex.keys()) == {"text"}
