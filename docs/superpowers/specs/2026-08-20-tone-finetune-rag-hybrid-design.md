# Tone Fine-Tune + Docs RAG Hybrid — Design Spec

Date: 2026-08-20
Status: Approved for planning

## Goal

Split the assistant's two jobs cleanly:

- **Tone/personality/teaching style** comes from a small, focused LoRA fine-tune of Qwen2.5-Coder-1.5B.
- **Technical accuracy** comes from retrieval-augmented generation (RAG) over real documentation (MDN, react.dev, Next.js docs, TypeScript Handbook), stored in Postgres + pgvector.

This replaces the current `engineering-model-finetuned` model, which was trained on 688 technical Q&A pairs and mixes both concerns — it works for tone but has no way to stay accurate or current on facts, and a 1.5B model fine-tuned on a few hundred examples doesn't reliably memorize technical detail anyway (see prior debugging session: training was fixed for stability, but factual accuracy was never the fine-tune's strength to begin with).

## Non-goals (this iteration)

- AWS/GCP docs (deferred — no open-source markdown source like the other four, would need real HTML scraping with rate-limiting/robots.txt handling; separate future iteration)
- Re-ranking, multi-hop retrieval, or query rewriting — plain top-k cosine similarity is enough at this scale
- A long-running backend service, auth, or multi-user support — this stays a single local Streamlit app for personal use
- Multimodal input — the base model and fine-tune are text-only, no image/audio handling
- Automatic doc-corpus refresh/updates — this is a one-time ingestion; re-running the ingestion script manually is fine for now
- Redistribution of scraped doc content — this is a personal local retrieval store, not a published dataset

## Part 1 — Tone-only fine-tune

### Cleanup (full wipe of the current fine-tune)

Delete, in this order:
- `ollama rm engineering-model-finetuned`
- `mlx_model/adapter_v4/`
- `merged_model/`
- `engineering-model-v4.gguf`
- `dataset/`, `clean_data/` (superseded by the new tone-only dataset)

Keep: `mlx_model/base_model/` (MLX-converted base model, used for training), `qwen2.5-coder-1.5b-hf/` (HF-format base model, used by `merge_lora.py` as the merge target), `llama.cpp/` (GGUF conversion tooling), all pipeline scripts (`finetune_1.5b.py`, `merge_lora.py`, `build_dataset.py` gets replaced — see below). `qwen2.5-coder-1.5b-base.gguf` is unrelated to this pipeline (only used by the separate `Modelfile.base`) — leave it alone either way.

### New dataset: `build_tone_dataset.py` (replaces `build_dataset.py`)

~80-120 hand-crafted examples in the same `### User:\n...\n### Assistant:\n...` format the pipeline already uses. Critically, examples span **varied subjects** — everyday questions, general how-tos, simple math/science, a few coding basics — not exclusively JS/React. The point isn't teaching facts, it's teaching a consistent voice across arbitrary topics: warm, encouraging, breaks things into numbered steps, checks understanding, avoids jargon dumps. Keeping any technical content in the examples deliberately simple/well-known (not obscure API details) avoids the model re-learning to assert unverified technical specifics — that job now belongs to RAG.

Length filtering reuses the same token-based approach from the current `build_dataset.py` (hand-written examples will be short by construction, but keep the guard for safety).

### Training

Reuse the exact hyperparameters already validated as stable on this machine:
```
mlx_lm.lora --model ./mlx_model/base_model --train \
  --data ./dataset --adapter-path ./mlx_model/adapter_tone \
  --iters 300 --learning-rate 1e-5 --batch-size 2 \
  --num-layers 16 --grad-checkpoint --fine-tune-type lora
```
(Iteration count lowered from 500 to 300 since the dataset is smaller — 300 iterations at batch-size 2 covers the ~100-example set several times over without needing as long a run. Adjust if validation loss hasn't plateaued.)

This machine has 18GB total RAM; `--grad-checkpoint` and `--batch-size 2` are required, not optional — the earlier investigation found batch-size 4 without checkpointing pushed peak memory to 30-57GB, causing silent Metal corruption (`inf`/`nan` losses). Peak memory with the current safe config measured ~5.3GB.

### Export & deploy

`merge_lora.py` (adapter path updated to `adapter_tone`) → `llama.cpp/convert_hf_to_gguf.py` → `engineering-tone-v1.gguf` → new `Modelfile.tone` → `ollama create engineering-tone -f Modelfile.tone`.

### Validation

Manual spot-check: ask 5-6 questions across different subjects (not just JS) and confirm the tone is consistent (encouraging, step-by-step) regardless of topic, and that the model isn't confidently asserting specific technical facts it should instead be getting from RAG context.

## Part 2 — Docs RAG system

### Ingestion — pull markdown from source repos, not HTML scraping

All four target doc sets publish their actual content as markdown/MDX in public GitHub repos:

| Source | Repo | Content path |
|---|---|---|
| MDN (JS/Web APIs/CSS/HTML) | `mdn/content` | `files/en-us/` |
| react.dev | `reactjs/react.dev` | `src/content/` |
| Next.js docs | `vercel/next.js` | `docs/` |
| TypeScript Handbook | `microsoft/TypeScript-Website` | `packages/documentation/copy/en/` |

Pulling markdown directly (shallow git clone or GitHub API archive download) avoids HTML boilerplate (nav/footer/ads), avoids scraping fragility and robots.txt/rate-limit concerns entirely, and is far faster than rendering pages. `scrape_docs.py` clones each repo (shallow, `--depth 1`) into a scratch directory, walks the relevant content path, and hands markdown files to the chunker. Non-English MDN locales are skipped (`en-us` only).

### Chunking

Split each markdown file by heading boundaries (`##`/`###`), merging adjacent small sections and splitting oversized ones, targeting roughly 400-700 tokens per chunk (measured with the same tokenizer used for training, for consistency) — small enough to keep retrieved context focused, large enough to keep a heading's content coherent. Store the source repo, file path, and heading trail with every chunk for citation/debugging.

### Embedding

`nomic-embed-text` pulled via `ollama pull nomic-embed-text` (768-dim), called through Ollama's embedding API for every chunk during ingestion, and for the user's query at retrieval time.

### Storage — Postgres + pgvector

```
brew install pgvector
createdb engineer_rag
psql engineer_rag -c "CREATE EXTENSION vector;"
```
Schema:
```sql
CREATE TABLE chunks (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,       -- e.g. 'mdn', 'react.dev', 'nextjs', 'ts-handbook'
    file_path TEXT NOT NULL,
    heading_trail TEXT,
    content TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL
);
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
```
HNSW over IVFFlat since the corpus is static after ingestion (no need to optimize for build speed over many updates) and HNSW gives better query recall at this scale.

### Query pipeline — `rag_query.py`

1. Embed the user's question via `nomic-embed-text`.
2. `SELECT content, source, file_path FROM chunks ORDER BY embedding <=> $1 LIMIT 5;` (cosine distance, top-5).
3. Build a prompt: retrieved chunks as labeled context blocks, followed by the user's question, using the `### User:/### Assistant:` format the tone model was trained on.
4. Send to `engineering-tone` via the Ollama API, stream the response.
5. Print the answer; optionally print which sources were used (for debugging/trust).

### Validation

- Retrieval spot-check: run a handful of known-answer queries (e.g. "how does useEffect cleanup work") and confirm the top results are actually the relevant MDN/react.dev sections.
- End-to-end spot-check: same queries through the full pipeline, confirm the answer is both factually grounded (matches retrieved content) and in the trained tutor tone.

## Part 3 — Minimal chat UI

Plain `ollama run`/Ollama's own interfaces bypass the RAG retrieval step entirely (they know nothing about the pgvector lookup), so a UI needs to sit on top of the RAG pipeline (`rag/query.py`'s retrieve-then-generate logic), not on top of raw Ollama. A single-file Streamlit app (`chat_app.py`) provides a real chat-bubble interface (`st.chat_message`, `st.chat_input`) with in-session conversation history, calling the same `retrieve()`/`build_prompt()` logic `rag/query.py` already exposes. Run locally with `streamlit run chat_app.py`; no separate backend server, no auth, single local user.

## Open risks

- **Memory**: ingestion (embedding potentially tens of thousands of chunks) and training both run against the same 18GB budget already found to be tight. Embedding via Ollama's small model should be lightweight per-call; if ingestion is slow, batch it in a resumable script rather than one long run.
- **pgvector extension**: not yet installed on this Postgres instance — first step of implementation, not assumed to already work.
- **Corpus staleness**: docs will drift out of date after the one-time ingestion; acceptable for this iteration per non-goals.
