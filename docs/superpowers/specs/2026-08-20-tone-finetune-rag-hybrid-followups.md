# Follow-up work: Tone Fine-Tune + Docs RAG Hybrid

The plan at `docs/superpowers/plans/2026-08-20-tone-finetune-rag-hybrid.md` shipped and was merged to `main` by explicit user decision, ahead of resolving all findings from the final whole-branch review. This doc preserves those findings (the plan's per-task execution ledger lived in gitignored scratch space and was deleted when the worktree was cleaned up — this is the durable record going forward).

## Critical (recommended to fix first)

**Pre-heading content loss in `rag/chunking.py`.** `_split_into_sections` builds sections only from `HEADING_RE` matches; any text before the first heading match is silently dropped. MDN files have no in-body H1 (title lives in frontmatter) and put the canonical one-sentence definition immediately after frontmatter, before the first `##` — so that sentence is lost on essentially every MDN reference page.

Measured impact (verified live against the real corpus):

| source | docs | words total | words dropped | % |
|---|---|---|---|---|
| mdn | 10,927 | 4,542,377 | 650,768 | **14.3%** |
| nextjs | 451 | 376,219 | 30,932 | 8.2% |
| react.dev | 223 | 481,435 | 13,751 | 2.9% |
| ts-handbook | 133 | 272,100 | 6,103 | 2.2% |

No page is entirely missing (every doc has ≥1 chunk), so this is degraded recall, not a hole — but it removes the highest-retrieval-value sentence (the definition) from most reference pages, and plausibly contributes to the grounding failures documented below.

**Fix shape:** in `_split_into_sections`, emit a leading untitled section for any non-empty text before the first heading match (after stripping YAML frontmatter), same pattern as the existing `"(untitled)"` fallback. Add a regression test, then re-run `rag/ingest.py` to recover the content (same procedure already used once for the heading_trail collision fix).

## Important

1. **pgvector-on-PG16 build is undocumented and fragile.** The extension is built from source directly into `/opt/homebrew/opt/postgresql@16`'s keg (`vector.dylib` + `extension/vector*`), because Homebrew's `pgvector` formula only ships bottles for postgresql@17/18. A routine `brew upgrade postgresql@16` will wipe this and break `engineer_rag` with `could not open extension control file`. **Fix:** add the rebuild recipe as a comment in `rag/schema.sql`:
   ```
   # If pgvector stops working after a postgresql@16 upgrade, rebuild it:
   git clone --depth 1 --branch v0.8.6 https://github.com/pgvector/pgvector.git /tmp/pgvector-build
   make -C /tmp/pgvector-build PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config
   make -C /tmp/pgvector-build PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config install
   ```

2. **Spec's Training section is stale/wrong.** `docs/superpowers/specs/2026-08-20-tone-finetune-rag-hybrid-design.md` still shows `--iters 300` with no `scale` parameter — this is the exact config that produced unreliable/corrupted output. The actual working config is `lora_config_tone.yaml` (`scale: 4.0`, `iters: 150`) plus `Modelfile.tone`'s `temperature 0.3` (also load-bearing for stability). Update the spec's Training section to match, with a one-line note on why (LoRA scale=20 default was too aggressive for a 24-example dataset).

3. **Stale-duplicate cleanup after re-ingestion has no reproducible path.** Re-running `rag/ingest.py` without truncating first leaves stale rows (old pre-fix `heading_trail` values coexisting with new ones) whenever `heading_trail` formatting changes. The cleanup used once:
   ```sql
   DELETE FROM chunks c1
   WHERE EXISTS (
       SELECT 1 FROM chunks c2
       WHERE c2.source = c1.source AND c2.file_path = c1.file_path
         AND c2.heading_trail LIKE c1.heading_trail || ' (part %'
   );
   ```
   **Fix:** add a `--reset` flag to `rag/ingest.py` that truncates `chunks` first, or check in this query as a documented maintenance script.

4. **The `satisfies` retrieval diagnosis in the spec's Validation results is factually wrong.** It claims the corpus lacks the dedicated TypeScript Handbook page on `satisfies`. It doesn't — `ts-handbook/release-notes/TypeScript 4.9.md` ("The `satisfies` Operator", both parts) is in the corpus, confirmed present in the database. The real issue is retrieval ranking: that chunk doesn't make the top-5 for the query "What does the TypeScript satisfies operator do?" This is a **retrieval-ranking failure, not a coverage gap** — the fix is reranking or hybrid keyword+vector search, not more scraping. This also means the "clearest evidence" cited for the hybrid design's value (the no-retrieval vs. RAG comparison on this exact query) rests on a query where retrieval demonstrably underperformed, weakening that specific piece of evidence even though the overall RAG-vs-no-RAG conclusion still holds elsewhere.

5. **No `num_ctx` cap; large prompts likely exceed the model's context window.** `engineering-tone` runs at Ollama's default `num_ctx=4096`. Measured Task-12 prompts ranged 986–2,916 tokens; a top-5 retrieval of large chunks (max chunk measured at 1,084 tokens) can reach ~5,600 tokens, over the window. Ollama truncates from the front, meaning the `build_prompt()` instruction ("stick to the sources, say so if they don't cover it") is the first thing dropped — plausibly a direct contributor to the hallucination-style failures found in Task 12. The adapter was also trained at `max_seq_length: 2048`, so prompts this long are already outside its training distribution. **Fix:** add `PARAMETER num_ctx 8192` to `Modelfile.tone`, and/or cap total context tokens assembled in `rag/query.py`'s `build_prompt()`.

6. **`chat_app.py` duplicates `ask()`'s logic instead of calling it**, contradicting the plan's own stated intent ("no duplicated RAG code"). It re-implements the `requests.post(..., raw=True)` call verbatim instead of calling `rag.query.ask()`. Collapse to `answer, chunks = ask(question)`.

7. **`requirements-rag.txt` is missing `requests` and `transformers`**, both directly imported by `rag/ingest.py`/`rag/query.py`/`chat_app.py`. They happen to be present transitively (via streamlit, via the MLX toolchain), so nothing is broken today, but the file doesn't describe the real dependency set. Add both.

## Minor (lower priority, batch with the above when convenient)

- `embed()` + `OLLAMA_EMBED_URL` + `EMBED_MODEL` are duplicated verbatim between `rag/ingest.py` and `rag/query.py` — extract to `rag/embedding.py`.
- `chunk_markdown` can emit chunks over the nominal 700-token max when a sub-700-token section merges with a following section before the 700 check fires (measured real max: 1,084 tokens). No downstream harm (embedding model handles up to 2048), but the invariant isn't actually enforced/tested at the intended default settings.
- `_heading_trail(stack, "" if title else "")` in `rag/chunking.py` — the conditional always evaluates to `""`; dead code.
- `HEADING_RE` matches `#`-prefixed lines inside fenced code blocks (e.g. bash comments, stack traces) — measured at 70 of 89,173 headings (0.08%) across 18 docs. Negligible but real.
- Chunk content/heading trails carry raw MDX anchors and MDN macros (`{{jsxref("Array")}}`, `{/*slug*/}`) into the retrieved context shown to the model and displayed as sources — cosmetic noise, not incorrect.
- No retry around `embed()` in `rag/ingest.py` — a single transient HTTP failure aborts a multi-hour ingestion run (mitigated by idempotent inserts + resumability, but still costly to hit).
- `rag/ingest.py` is cwd-dependent (relative paths); `rag/query.py` isn't. Anchor both to `Path(__file__).resolve().parent.parent`.
- Cast inconsistency: `rag/query.py` uses `%s::vector` (needed for the `ORDER BY ... <=>` expression context), `rag/ingest.py` uses bare `%s` (works via assignment-cast into the typed column). Both are correct as-is, but worth a comment explaining why they differ.
- Streamlit (`chat_app.py`) binds to all interfaces by default — fine per the "no auth" non-goal, but reachable by anyone on the same LAN. `streamlit run chat_app.py --server.address localhost` restricts to local-only.
- No README — a fresh clone of the public GitHub repo has no path to a working system, since every runtime artifact (GGUF, adapter, corpus, DB) is gitignored by design.

## On the residual hallucination limitation (not something to "fix" here)

Task 12's validation found 3 of 5 spot-checked RAG answers contained technical claims that were self-contradictory or unsupported by/contradicting their own retrieved sources. This is a property of running a 1.5B model in a plain top-5 RAG loop, not a bug — documenting it precisely (with verbatim quotes, in the spec's "Validation results" section) is real value delivered by this project, not a gap in it. Recommended low-cost mitigation: add a caption in `chat_app.py`'s UI ("Answers are generated from the excerpts below and may contain errors — check the sources") so the limitation is visible where a user actually encounters it, since the "Sources" expander currently reads as more authoritative than the measured accuracy supports.
