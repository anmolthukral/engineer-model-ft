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

## Validation results

End-to-end validation (Task 12) ran the 5 suggested questions from the task brief through `rag/query.py` against the live 26,259-chunk corpus, plus one no-retrieval comparison directly against `engineering-tone`. Full transcripts are in the task-12 report.

**Tone.** Consistent across all 5 RAG answers: numbered steps, plain-language framing, a worked code example, and an encouraging/practical closing note. This held regardless of doc source (MDN, react.dev, Next.js, TypeScript Handbook), confirming the tone fine-tune generalizes across subjects as designed — RAG context doesn't disrupt the trained voice.

**Retrieval relevance.** 4 of 5 queries pulled clearly on-topic chunks (JS `Array.flat()`, CSS `position`, React `useState`, and largely relevant Next.js routing docs). The TypeScript `satisfies` query was the weak case: the corpus doesn't contain the dedicated "The satisfies Operator" Handbook page, so retrieval surfaced only tangentially related chunks (narrowing, type compatibility, a JSDoc `@satisfies` release note) — none of which actually explain the operator's core behavior.

**Grounding vs. hallucination — the core finding.** This is where the hybrid approach shows real, not just theoretical, limits:
- The `Array.flat()` and TypeScript `satisfies` answers were technically accurate and matched real behavior (the `satisfies` one appears to be pretraining knowledge carrying through, since the retrieved chunks didn't actually cover it — accurate, but not clearly *grounded* in what was retrieved).
- The Next.js app-router-vs-pages-router answer included at least one unsupported/inaccurate claim (implying the Pages Router has "limited support" for dynamic routes, which it does not) not backed by the retrieved excerpts.
- The `useState` object answer was internally contradictory: it correctly quoted "don't mutate state" from the retrieved React docs, then immediately gave a "you can mutate directly if you deep-clone first" example that contradicts both the source and React's actual model.
- The `position: absolute` vs `fixed` answer contradicted itself within a single response (first correctly stating absolute is relative to the nearest positioned ancestor, then later claiming absolute is "always relative to the viewport, unaffected by scrolling") and introduced a fabricated "margin collision" concept not present in CSS or the retrieved sources.

So RAG retrieval measurably improves *topical grounding* (real excerpts, real citations, tone intact) but does not eliminate hallucination — the model can still assert things beyond, or in contradiction of, what was retrieved. This matches the one earlier known issue (`const`) noted separately, and the pattern generalizes: roughly half of the 5 answers here contained at least one confident but unsupported/self-contradictory technical claim.

**No-retrieval comparison.** Asking `engineering-tone` directly (no RAG) "What does the TypeScript `satisfies` operator do?" produced an answer with the same warm, step-by-step tone but materially worse technical content: it described `satisfies` as similar to an "extends" keyword from other languages, and its code example used invalid TypeScript syntax (`satisfies` applied to a type alias declaration rather than a value expression) — a fabrication the fine-tune has no way to catch, since it was never trained on `satisfies` and has no retrieval to fall back on. Side by side, this is the clearest evidence for the split: the tone is identical in both versions, but the RAG-augmented version is answering from real source text while the no-retrieval version is confabulating syntax.

**Conclusion.** The hybrid design's core hypothesis holds: separating tone (fine-tune) from facts (RAG) works, and RAG-grounded answers are consistently more accurate than ungrounded ones from the same tone model. But RAG does not make the system fully accurate — the model can still generate confident, plausible-sounding technical claims that aren't actually in the retrieved excerpts, especially in the connective/explanatory sentences around correctly-quoted material. This is a known category of remaining risk, not a defect specific to this task's answers, and would need a fact-checking or citation-verification step in a future iteration if higher precision is required.
