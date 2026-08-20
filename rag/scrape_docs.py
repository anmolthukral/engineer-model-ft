#!/usr/bin/env python3
"""Pull markdown doc source directly from the upstream GitHub repos rather
than scraping rendered HTML — cleaner text, no nav/ad/footer noise, no
robots.txt/rate-limit concerns, and much faster. MDN scope is limited to
the JS/Web-API/CSS/HTML subtrees (the full repo covers far more than this
project needs)."""
import json
import subprocess
from pathlib import Path

SCRATCH_DIR = Path("./rag/_scratch")
OUTPUT_FILE = Path("./rag/corpus.jsonl")

REPOS = [
    {
        "name": "mdn",
        "url": "https://github.com/mdn/content",
        "content_paths": [
            "files/en-us/web/javascript",
            "files/en-us/web/api",
            "files/en-us/web/css",
            "files/en-us/web/html",
        ],
    },
    {
        "name": "react.dev",
        "url": "https://github.com/reactjs/react.dev",
        "content_paths": ["src/content"],
    },
    {
        "name": "nextjs",
        "url": "https://github.com/vercel/next.js",
        "content_paths": ["docs"],
    },
    {
        "name": "ts-handbook",
        "url": "https://github.com/microsoft/TypeScript-Website",
        "content_paths": ["packages/documentation/copy/en"],
    },
]


def shallow_clone(repo: dict) -> Path:
    dest = SCRATCH_DIR / repo["name"]
    if dest.exists():
        print(f"  [skip clone] {repo['name']} already present at {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Cloning {repo['url']} (depth 1)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", repo["url"], str(dest)],
        check=True,
    )
    return dest


def collect_markdown_files(repo_dir: Path, content_paths: list[str]) -> list[Path]:
    files = []
    for rel_path in content_paths:
        base = repo_dir / rel_path
        if not base.exists():
            print(f"  [warn] {base} does not exist, skipping")
            continue
        files.extend(base.rglob("*.md"))
        files.extend(base.rglob("*.mdx"))
    return files


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(OUTPUT_FILE, "w") as out:
        for repo in REPOS:
            print(f"\n{repo['name']}:")
            repo_dir = shallow_clone(repo)
            md_files = collect_markdown_files(repo_dir, repo["content_paths"])
            print(f"  found {len(md_files)} markdown files")
            for md_file in md_files:
                try:
                    text = md_file.read_text(encoding="utf-8", errors="ignore")
                except Exception as e:
                    print(f"  [warn] could not read {md_file}: {e}")
                    continue
                if not text.strip():
                    continue
                rel_path = str(md_file.relative_to(repo_dir))
                out.write(json.dumps({
                    "source": repo["name"],
                    "file_path": rel_path,
                    "raw_markdown": text,
                }) + "\n")
                total += 1
    print(f"\nWrote {total} documents to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
