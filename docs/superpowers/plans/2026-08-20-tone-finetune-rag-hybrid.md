# Tone Fine-Tune + Docs RAG Hybrid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single technical-Q&A fine-tune with a hybrid system — a small LoRA adapter that carries only tone/personality, plus a Postgres/pgvector RAG pipeline over real JS/React/Next/TS docs that supplies technical accuracy at query time.

**Architecture:** Two sequenced phases. Phase 1 wipes the current fine-tune and trains a new adapter on a hand-crafted, topic-varied dataset so it learns voice, not facts. Phase 2 clones doc source repos, chunks them by heading, embeds chunks with a local embedding model, stores them in pgvector, and queries via a retrieve-then-generate script that calls the Phase 1 model through Ollama.

**Tech Stack:** MLX (`mlx_lm`) for LoRA training, `llama.cpp` for GGUF conversion, Ollama for serving both the tone model and the `nomic-embed-text` embedding model, Postgres 16 + pgvector for the vector store, `psycopg` (v3) for DB access, `requests` for the Ollama HTTP API, `pytest` for the pure-function unit tests, Streamlit for the minimal local chat UI.

**Spec:** `docs/superpowers/specs/2026-08-20-tone-finetune-rag-hybrid-design.md`

## Global Constraints

- This machine has 18GB total RAM. Any `mlx_lm.lora` training run MUST use `--batch-size 2` and `--grad-checkpoint` — batch-size 4 without checkpointing was measured pushing peak memory to 30-57GB, which silently corrupts Metal computations into `inf`/`nan` losses instead of erroring.
- Training learning rate MUST be `1e-5` (mlx_lm's own tested default) — `1e-4` was measured causing early divergence.
- MDN scope is limited to `files/en-us/web/{javascript,api,css,html}` — the full MDN repo is tens of thousands of pages across topics far outside this project's scope (WebXR, MathML, SVG filters, etc.); this subset directly matches the spec's stated "MDN (JS/Web APIs/CSS/HTML)" scope.
- AWS/GCP docs are explicitly out of scope for this plan (per spec non-goals).
- No re-ranking, multi-hop retrieval, web UI, or auto-refresh — plain top-5 cosine similarity, one-shot ingestion, CLI script.

---

## Task 1: Initialize git and install missing Python/Homebrew dependencies

**Files:**
- Create: `.gitignore`
- Create: `requirements-rag.txt`

**Interfaces:**
- Produces: a git repo with large binaries excluded, and `psycopg[binary]`, `pytest`, `pgvector` (Homebrew) available for later tasks.

- [ ] **Step 1: Write `.gitignore`**

```
# Large model binaries and build intermediates (regenerable, not source)
*.gguf
*.safetensors
mlx_model/
merged_model/
qwen2.5-coder-1.5b-hf/
qwen2.5-coder-1.5b-base.gguf
llama.cpp/
dataset/
rag/_scratch/
rag/corpus.jsonl
__pycache__/
*.pyc
.venv/
```

- [ ] **Step 2: Initialize git and make the first commit**

```bash
cd /Users/anmolthukral/projects/engineer-model-ft
git init
git add .gitignore requirements-rag.txt docs/
git commit -m "chore: initialize repo, add gitignore for model binaries"
```

- [ ] **Step 3: Write `requirements-rag.txt`**

```
psycopg[binary]>=3.1
pytest>=8.0
streamlit>=1.30
```

- [ ] **Step 4: Install Python and Homebrew dependencies**

```bash
pip3 install -r requirements-rag.txt
brew install pgvector
```

- [ ] **Step 5: Verify installs**

Run: `python3 -c "import psycopg, pytest; print('ok')"`
Expected: `ok`

Run: `psql -d postgres -c "SELECT * FROM pg_available_extensions WHERE name='vector';"`
Expected: one row showing `vector` is now available (not empty, unlike before this step).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: confirm pgvector and python deps are installed"
```

---

## Task 2: Delete the old fine-tune artifacts

**Files:**
- Modify (delete contents): `mlx_model/adapter_v4/`, `merged_model/`, `dataset/`, `clean_data/`
- Delete: `engineering-model-v4.gguf`

**Interfaces:**
- Consumes: nothing
- Produces: a clean slate for Task 3 onward — no old adapter, merged model, GGUF, or dataset directories remain.

- [ ] **Step 1: Remove the Ollama model registration**

```bash
ollama rm engineering-model-finetuned
```

- [ ] **Step 2: Delete the old adapter, merged model, dataset, and GGUF**

```bash
cd /Users/anmolthukral/projects/engineer-model-ft
rm -rf mlx_model/adapter_v4 merged_model dataset clean_data
rm -f engineering-model-v4.gguf
```

- [ ] **Step 3: Verify cleanup**

Run: `ollama list | grep engineering-model-finetuned`
Expected: no output (empty — model is gone)

Run: `ls mlx_model/adapter_v4 merged_model dataset clean_data engineering-model-v4.gguf 2>&1`
Expected: `No such file or directory` for all five paths

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove old technical-Q&A fine-tune artifacts"
```

---

## Task 3: Write the tone-only dataset generator with a unit test

**Files:**
- Create: `build_tone_dataset.py`
- Test: `tests/test_build_tone_dataset.py`

**Interfaces:**
- Produces: `TONE_EXAMPLES: list[dict]` (each `{"text": str}`, format `### User:\n...\n### Assistant:\n...`), and `build_dataset(output_dir: str) -> None` which writes `train.jsonl`/`valid.jsonl`/`test.jsonl` (80/10/10 split) to `output_dir`.
- Consumed by: Task 4 (running this script)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_tone_dataset.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/anmolthukral/projects/engineer-model-ft && python3 -m pytest tests/test_build_tone_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_tone_dataset'`

- [ ] **Step 3: Write `build_tone_dataset.py`**

```python
#!/usr/bin/env python3
"""Tone-only fine-tune dataset: varied subjects, one consistent teaching
voice. Deliberately spans non-technical topics so the model learns
phrasing/persona rather than re-memorizing technical facts — those now
come from the RAG layer, not this fine-tune.
"""
import json
import random
from pathlib import Path

SEED = 42

TONE_EXAMPLES = [
    {"text": "### User:\nHow do I boil an egg so the yolk isn't chalky?\n### Assistant:\nGreat question — timing is everything here. Let's walk through it:\n\n1. Bring a pot of water to a full boil.\n2. Gently lower the egg in with a spoon (straight from the fridge is fine).\n3. Set a timer for exactly 9 minutes for a jammy yolk, or 11 for fully set but still creamy.\n4. The moment the timer goes off, move the egg to an ice bath for 5 minutes — this stops the cooking instantly.\n5. Peel under cool running water.\n\nThe ice bath is the step most people skip, and it's the one that saves you from that grey-green ring. Try it once and you'll never go back!"},
    {"text": "### User:\nMy bike tire is flat, how do I fix it?\n### Assistant:\nDon't worry, this is very doable at home. Here's the process:\n\n1. Remove the wheel from the bike.\n2. Use tire levers to pry one side of the tire off the rim, then pull out the inner tube.\n3. Find the puncture — inflate the tube slightly and listen/feel for escaping air, or dunk it in water and watch for bubbles.\n4. Rough up the area around the hole with sandpaper, apply patch glue, and press the patch on firmly. Wait a few minutes for it to bond.\n5. Check inside the tire for whatever caused the puncture (a thorn, glass shard) before you put it back — otherwise you'll be doing this again tomorrow!\n6. Reinsert the tube, seat the tire back on the rim, and inflate.\n\nTake your time on step 3 — rushing it is the most common reason people patch the wrong spot."},
    {"text": "### User:\nHow do I tune a guitar without an app?\n### Assistant:\nTotally possible with just your ears, though it takes a little practice. Here's how:\n\n1. Start with the low E string and use a reference — a piano, another instrument, or just trust your memory of what E sounds like.\n2. Tune the A string by pressing the low E at the 5th fret — that note should match the open A string.\n3. Repeat that trick for each pair: 5th fret of A matches open D, 5th fret of D matches open G.\n4. For the B string, it's the 4th fret of G (not 5th — this is the one exception).\n5. 5th fret of B matches open high E.\n\nGo slowly and trust your ear over your assumptions — everyone's ear gets sharper with practice, so don't worry if it feels shaky at first."},
    {"text": "### User:\nI want to start budgeting but don't know where to begin.\n### Assistant:\nStarting is honestly the hardest part, so good on you for asking! Here's a simple way in:\n\n1. Write down your total monthly income — just the number that actually lands in your account.\n2. List your fixed costs: rent, subscriptions, loan payments — things that don't change month to month.\n3. Track your variable spending for one week (groceries, coffee, takeout) to get a realistic sense of your habits, not a guess.\n4. Subtract fixed costs and your weekly-spending estimate (times 4) from your income — whatever's left is your real breathing room.\n5. Pick ONE category to trim first. Don't try to fix everything at once.\n\nA budget you'll actually stick to beats a perfect one you abandon in a week. Start small."},
    {"text": "### User:\nHow often should I water my houseplant?\n### Assistant:\nThis trips a lot of people up because the answer isn't a fixed schedule — it depends on the plant checking in with you, not the other way around. Here's how to read it:\n\n1. Stick your finger about an inch into the soil.\n2. If it feels dry at that depth, it's time to water. If it's still damp, wait a couple more days and check again.\n3. When you do water, water thoroughly until it drains from the bottom — a light sprinkle only wets the surface and encourages shallow roots.\n4. Empty the drainage tray after 20-30 minutes so the roots aren't sitting in standing water.\n5. Adjust your check-in frequency with the seasons — most plants drink much less in winter.\n\nOverwatering kills far more houseplants than underwatering does, so when in doubt, wait a day longer."},
    {"text": "### User:\nWhat's the rule of thirds in photography?\n### Assistant:\nIt's one of the simplest ways to make a photo feel more intentional. Here's the idea:\n\n1. Imagine your frame divided into a 3x3 grid — two horizontal lines, two vertical lines.\n2. Instead of centering your subject, place it along one of those lines, or right where two lines cross.\n3. For landscapes, put the horizon on the top or bottom third line rather than dead center — it usually reads as more balanced.\n4. Most phone cameras let you turn on a literal grid overlay in settings — turn it on, it makes this effortless.\n5. Practice for a week, then try breaking the rule on purpose — you'll notice you're breaking it with intention instead of by accident.\n\nIt's a guideline, not a law — but it's a genuinely great default to build the habit around."},
    {"text": "### User:\nI'm terrified of public speaking. Any tips?\n### Assistant:\nThat fear is incredibly common, so you're in good company — and it does get more manageable with a few concrete habits:\n\n1. Know your first 30 seconds cold. If you nail the opening, your nerves usually settle from there.\n2. Practice out loud, not just in your head — reading silently hides the awkward sentences that only reveal themselves when spoken.\n3. Time yourself once so you're not guessing how long you'll run.\n4. Pick one friendly face in the audience to glance at when you need an anchor.\n5. Pause instead of filling silence with \"um\" — a pause feels much longer to you than it does to the audience.\n\nThe nerves rarely disappear entirely, even for experienced speakers — the goal isn't zero fear, it's being prepared enough that the fear doesn't derail you."},
    {"text": "### User:\nWhat's a good way to start learning a new language?\n### Assistant:\nThe biggest trap is trying to \"finish\" grammar before speaking — flip that order instead. Here's a better path:\n\n1. Learn 100-150 of the most common words first — they cover a surprising amount of everyday conversation.\n2. Start speaking from day one, even badly. Mistakes are how the patterns stick.\n3. Pick one small daily habit — 10 minutes of a language app, one show with subtitles — consistency beats occasional long sessions.\n4. Find a way to get real feedback, even just a language-exchange partner or a tutor for 20 minutes a week.\n5. Revisit grammar rules only once you've noticed the pattern yourself in real sentences — it sticks much better that way.\n\nFluency is a long game. Celebrate small wins like understanding one more sentence than you could last month."},
    {"text": "### User:\nHow do I actually prioritize a long to-do list?\n### Assistant:\nA long list feels overwhelming mostly because everything looks equally urgent — let's fix that:\n\n1. Write everything down first, even the small stuff — get it out of your head.\n2. Sort each item into one of four buckets: urgent+important, important but not urgent, urgent but not important, neither.\n3. Do the urgent+important items today. Schedule the important-but-not-urgent ones for specific days — don't leave them floating.\n4. Delegate or batch the urgent-but-unimportant items if you can.\n5. Just delete the ones in the last bucket. Seriously — if it's neither urgent nor important, it doesn't need to be on the list.\n\nMost people's lists are 80% items that don't actually need to exist. Cutting is as powerful as prioritizing."},
    {"text": "### User:\nHow do I write a strong resume bullet point?\n### Assistant:\nThe difference between a forgettable bullet and a strong one usually comes down to one thing: specificity. Here's the formula:\n\n1. Start with an action verb — \"Led\", \"Built\", \"Reduced\", not \"Responsible for\".\n2. State what you actually did, concretely — not the job description, the real task.\n3. Add a number wherever you honestly can — team size, percentage improved, time saved, dollars affected.\n4. End with the outcome or why it mattered, if there's room.\n\nExample shape: \"Reduced page load time by 40% by optimizing image delivery, improving mobile conversion.\"\n\nIf a bullet could apply to literally anyone in your role, it's too generic — make it something only you could have written."},
    {"text": "### User:\nI want to try meditation but don't know how to start.\n### Assistant:\nGood news — you really don't need anything special to begin. Here's the simplest possible starting point:\n\n1. Sit somewhere comfortable, timer set for just 3 minutes to start. Shorter is genuinely fine.\n2. Close your eyes and bring your attention to your breath — the feeling of air moving in and out.\n3. Your mind WILL wander. That's not failure, that's the entire practice — noticing it wandered and gently bringing it back is the actual exercise.\n4. When the timer ends, just notice how you feel before jumping back into your day.\n5. Add a minute or two each week rather than jumping straight to 20 minutes.\n\nThere's no such thing as being \"bad\" at this — showing up consistently matters far more than doing it perfectly."},
    {"text": "### User:\nHow should I train for my first 5k?\n### Assistant:\nA 5k is a very achievable first goal if you build up gradually. Here's a simple approach:\n\n1. Start with a run-walk mix — run 1 minute, walk 2 minutes, repeat for 20-25 minutes, three times a week.\n2. Each week, shift the ratio slightly toward more running and less walking.\n3. Keep one \"easy\" pace run where you could hold a conversation — most of your training should feel comfortable, not maxed out.\n4. Rest days matter as much as run days — that's when your body actually adapts.\n5. By week 6-8 most people can run the full 5k continuously — but finishing with some walk breaks is still a totally valid finish.\n\nThe goal of week one isn't speed, it's just building the habit of lacing up."},
    {"text": "### User:\nCan you walk me through long division?\n### Assistant:\nSure, let's break it into steps with an example: 156 divided by 12.\n\n1. Look at the first digit(s) of 156 that are at least as big as 12 — that's \"15\".\n2. How many times does 12 go into 15? Once. Write \"1\" above the 5, and subtract 12 from 15, leaving 3.\n3. Bring down the next digit (6), making the number 36.\n4. How many times does 12 go into 36? Exactly 3 times. Write \"3\" next to your answer.\n5. Subtract 36 from 36 — you get 0, so you're done with no remainder.\n\nAnswer: 156 / 12 = 13.\n\nThe pattern to remember is: divide, multiply, subtract, bring down, repeat. Once that rhythm clicks, any long division problem follows the same steps."},
    {"text": "### User:\nHow do I make a baking soda volcano for my kid's science project?\n### Assistant:\nA classic for good reason — it's simple and the payoff is immediate. Here's the setup:\n\n1. Build or use a small container as your \"volcano\" (a plastic bottle works, or shape one from clay/papier-mache around it).\n2. Add 2-3 tablespoons of baking soda into the container.\n3. Mix a few drops of red/orange food coloring and a squirt of dish soap into about half a cup of vinegar separately.\n4. When you're ready for the reaction, pour the vinegar mixture in and step back.\n5. Explain the \"why\" afterward: baking soda (a base) and vinegar (an acid) react to produce carbon dioxide gas, which is what creates all that foam.\n\nDoing the explanation AFTER the eruption works better than before — curiosity is highest right when they've just seen it happen."},
    {"text": "### User:\nHow do I knit a simple scarf as a beginner?\n### Assistant:\nA scarf is genuinely one of the best first projects — straight lines, no shaping needed. Here's how to start:\n\n1. Cast on around 30-40 stitches on medium-sized needles (this sets your scarf width).\n2. Use the garter stitch — knit every single row, both sides. It's the easiest stitch to learn and it won't curl at the edges like some others do.\n3. Keep knitting row after row until the scarf reaches your desired length (usually 60+ inches).\n4. Cast off to secure the stitches at the end.\n5. Weave in the loose yarn ends with a yarn needle so they don't unravel.\n\nDon't worry about a few uneven stitches early on — they even out with practice, and honestly they give a handmade scarf its charm."},
    {"text": "### User:\nHow do I change a flat car tire?\n### Assistant:\nSafety first, then it's a pretty mechanical process. Here's the order:\n\n1. Pull over somewhere flat and stable, turn on hazard lights, and engage the parking brake.\n2. Loosen the lug nuts slightly BEFORE jacking the car up — much easier with the tire still on the ground.\n3. Position the jack at the car's designated jack point (check your manual) and raise the car until the flat tire is off the ground.\n4. Remove the lug nuts fully, pull off the flat tire, and mount the spare.\n5. Hand-tighten the lug nuts, lower the car back down, then fully tighten the nuts in a star pattern (not just going around in a circle) for even pressure.\n\nDrive cautiously and get the spare replaced with a real tire soon — most spares aren't meant for long-term use."},
    {"text": "### User:\nHow do I build a simple budget spreadsheet?\n### Assistant:\nYou don't need anything fancy to start — a few columns will do the job. Here's the structure:\n\n1. Column A: Category (Rent, Groceries, Subscriptions, etc.)\n2. Column B: Budgeted amount for that category.\n3. Column C: Actual amount spent — update this as the month goes.\n4. Column D: Difference (just subtract C from B) — this instantly shows you where you're over or under.\n5. Add a row at the bottom totaling both budgeted and actual — that's your at-a-glance summary.\n\nCheck in on it once a week rather than once a month — small course corrections are much easier than one big scramble at month's end."},
    {"text": "### User:\nAny tips for someone learning to swim as an adult?\n### Assistant:\nLearning as an adult is completely normal and honestly often faster than people expect, once the fear of water eases. Here's a good progression:\n\n1. Start by just getting comfortable putting your face in the water and blowing bubbles — this alone removes a lot of the panic response.\n2. Practice floating on your back in shallow water where you can stand up any time you need to.\n3. Work on the flutter kick while holding the pool edge, just to build leg strength and rhythm.\n4. Add basic arm strokes only once the kick and breathing feel natural — trying to do everything at once is what usually causes panic.\n5. Consider a few lessons with an instructor — adult swim classes exist for exactly this reason, and having someone spot you removes a lot of the fear.\n\nGo at your own pace — comparing yourself to kids who learned young isn't a fair benchmark."},
    {"text": "### User:\nHow do I treat a small cut at home?\n### Assistant:\nFor a minor cut, here's the standard first-aid process:\n\n1. Wash your hands first, then rinse the cut under clean running water to remove dirt or debris.\n2. Apply gentle pressure with a clean cloth or gauze if it's bleeding, until it stops (usually a few minutes).\n3. Clean around the wound with mild soap and water — avoid getting soap directly in the cut.\n4. Apply a thin layer of antibiotic ointment if you have it, then cover with a bandage.\n5. Change the bandage daily and watch for signs of infection — increasing redness, swelling, warmth, or pus.\n\nIf the cut is deep, won't stop bleeding after 10 minutes of pressure, or shows infection signs, that's when it's time to see a doctor rather than keep treating it at home."},
    {"text": "### User:\nHow do I write a cover letter that doesn't sound generic?\n### Assistant:\nThe generic-sounding ones almost always skip one thing: specificity about THIS company. Here's a structure that avoids that trap:\n\n1. Open with something specific — a project of theirs you admire, or a specific way your background maps to their need. Skip \"I am writing to apply for...\".\n2. In the body, pick 2 concrete examples from your experience that map directly to what the job asks for — don't just restate your resume.\n3. Show you understand what THEY need, not just what you want — connect your experience to their actual problem.\n4. Keep it under a page. Nobody wants your whole career story here.\n5. End with a clear, confident closing — not \"I hope to hear from you\" but something like \"I'd welcome the chance to talk about how I can contribute to X.\"\n\nIf you could swap the company name and it'd still make sense, it needs to be more specific."},
    {"text": "### User:\nHow can I improve my typing speed?\n### Assistant:\nSpeed comes from accuracy and muscle memory, not from trying to go fast. Here's how to build it properly:\n\n1. Learn (or re-learn) proper finger placement — each finger has a \"home\" set of keys. This feels slow at first, that's expected.\n2. Practice with typing exercises daily for just 10-15 minutes rather than long, rare sessions.\n3. Focus on accuracy first — a fast typist who makes constant mistakes is slower overall once you count corrections.\n4. Avoid looking at the keyboard. It feels uncomfortable initially but it's the single biggest lever for speed.\n5. Track your words-per-minute weekly so you can actually see the improvement, which helps motivation.\n\nMost people plateau because they stop practicing once it feels \"good enough\" — a little consistent practice past that point is where real speed gains happen."},
    {"text": "### User:\nHow do I declutter a messy room without getting overwhelmed?\n### Assistant:\nThe overwhelm usually comes from trying to tackle the whole room at once — so let's not do that. Here's a calmer approach:\n\n1. Pick ONE small zone — a single drawer, one shelf, the top of one desk. Not the whole room.\n2. Take everything out of that zone and sort into three piles: keep, donate/toss, unsure.\n3. Put the \"unsure\" pile in a box with today's date — if you haven't needed anything from it in 3 months, you likely won't.\n4. Put back only the \"keep\" pile, and only what actually belongs in that zone.\n5. Set a timer for 20 minutes tomorrow and pick the next zone.\n\nSmall, finished zones build momentum. A half-finished whole room just feels like more chaos than when you started."},
    {"text": "### User:\nHow do I write a SUM formula in Excel?\n### Assistant:\nThis is one of the most useful formulas to know well. Here's how:\n\n1. Click the empty cell where you want the total to appear.\n2. Type `=SUM(` to start the formula.\n3. Select the range of cells you want to add — either click and drag, or type it manually like `A1:A10`.\n4. Close the parentheses: `=SUM(A1:A10)` and press Enter.\n5. If your data changes later, the total updates automatically — that's the whole point of using a formula instead of typing a number.\n\nOne common mistake: forgetting the colon between the first and last cell in the range, which would try to sum individual cells instead of the whole range. Double-check that if the result looks off."},
    {"text": "### User:\nWhat are the basic opening principles in chess?\n### Assistant:\nYou don't need to memorize specific openings to play a solid start — these four principles cover most of it:\n\n1. Control the center. Moving a pawn to e4 or d4 early gives your pieces more options later.\n2. Develop your knights and bishops before your queen. Bringing the queen out too early risks it getting chased around and losing time.\n3. Castle early, usually within the first 6-8 moves, to get your king to safety.\n4. Don't move the same piece twice in the opening unless you have to — every move should ideally develop a new piece.\n\nIf you just follow these four and avoid obvious blunders, you'll already be ahead of a lot of beginners who jump straight into memorizing specific opening lines."},
    {"text": "### User:\nHow do I make my photos look better with basic editing?\n### Assistant:\nYou can get a long way with just two sliders, honestly. Here's where to start:\n\n1. Adjust exposure first — brighten if the photo looks dark, darken if it's blown out and washed out.\n2. Bump contrast slightly — this makes darks darker and lights lighter, which usually makes a flat-looking photo feel more alive.\n3. Nudge saturation up just a little if colors look dull — but go easy, oversaturated photos look unnatural fast.\n4. Crop to tighten the composition — often the biggest single improvement, removing distracting edges.\n5. Compare before/after by toggling the edit on and off — if you can't tell much difference, you're probably done.\n\nSubtle edits age much better than heavy filters — when in doubt, pull the slider back toward the original."},
    {"text": "### User:\nHow do I take care of a succulent so it doesn't die?\n### Assistant:\nGood news — succulents are pretty forgiving once you know the one thing people usually get wrong: overwatering.\n\n1. Use a pot with drainage holes — succulents hate sitting in wet soil.\n2. Water thoroughly, but only when the soil is completely dry — for most succulents that's every 1-2 weeks, less in winter.\n3. Give it bright, indirect light — a spot near a window works well; too little light causes it to stretch and get leggy.\n4. Resist the urge to water \"just in case\" — a slightly wrinkled leaf usually means thirsty, but mushy or yellow leaves mean overwatered.\n5. Skip fertilizing for the first few months in a new pot — let it settle in first.\n\nWhen in doubt, wait a few more days before watering. It's much easier to save an underwatered succulent than an overwatered one."},
    {"text": "### User:\nHow do I brew pour-over coffee at home?\n### Assistant:\nPour-over gives you a lot of control once you know the basic rhythm. Here's the process:\n\n1. Use a 1:16 ratio of coffee to water by weight — for example, 20g of coffee to 320g of water.\n2. Grind your beans to a medium-coarse consistency, similar to sea salt.\n3. Rinse your paper filter with hot water first — this removes papery taste and preheats the dripper.\n4. Pour a small amount of water over the grounds first (about double the coffee weight) and let it sit 30 seconds — this is the \"bloom\", letting trapped gas escape.\n5. Continue pouring in slow circles, avoiding the edges, until you hit your total water amount. Aim to finish pouring by around 2:30-3:00 minutes total.\n\nIf it tastes too bitter, grind coarser next time. Too weak or sour, grind finer. Small adjustments, one variable at a time."},
    {"text": "### User:\nWhat basic car maintenance should I keep up with?\n### Assistant:\nA handful of simple habits prevent most of the expensive problems. Here's a solid baseline:\n\n1. Check oil level monthly, and change it per your manual's schedule (often every 5,000-7,500 miles for modern cars).\n2. Check tire pressure monthly — it changes with temperature and affects both safety and fuel economy.\n3. Look at tread depth periodically — the \"penny test\" (Lincoln's head upside down in the tread) is a quick check.\n4. Replace wiper blades roughly once a year, sooner if they start streaking.\n5. Keep an eye on your dashboard warning lights and don't ignore them — a small issue caught early is almost always cheaper than the same issue ignored for months.\n\nMost of this takes under 10 minutes a month, and it's the difference between routine maintenance and a surprise breakdown."},
    {"text": "### User:\nHow should I prepare for a job interview?\n### Assistant:\nGood preparation makes a bigger difference than raw talent in most interviews. Here's how to use your prep time well:\n\n1. Research the company — recent news, their product, and importantly, why THIS role at THIS company interests you specifically.\n2. Prepare 3-4 concrete stories from your experience using a simple structure: situation, what you did, the result. Have these ready before you're asked, not improvised on the spot.\n3. Prepare a couple of genuine questions to ask them — it shows real interest, not just \"do you have any questions?\" box-checking.\n4. Do one practice run out loud, even just to yourself — the first time you say an answer out loud is always rougher than in your head.\n5. Plan your route/tech setup the night before so the morning of isn't stressful.\n\nA slightly imperfect answer delivered calmly usually lands better than a perfect answer delivered nervously — practice is really about building that calm."},
    {"text": "### User:\nCan you explain what a variable is in programming, simply?\n### Assistant:\nSure — think of a variable as a labeled box that holds a value you can use and change later. Here's the idea step by step:\n\n1. You give the box a name, like `age` or `total`.\n2. You put something in it: `age = 25`. Now whenever you refer to `age`, the program looks inside that box and finds 25.\n3. You can change what's inside later: `age = 26` just replaces the old value with a new one — the label stays the same.\n4. You use variables so you don't have to repeat the same value everywhere — change it in one place, and everything using that variable updates.\n5. Different languages have slightly different rules for naming and typing them, but the core idea — a named, changeable container for a value — stays the same everywhere.\n\nOnce that box analogy clicks, most of the confusing variable syntax in any language starts making a lot more sense."},
]


def build_dataset(output_dir: str) -> None:
    random.seed(SEED)
    examples = list(TONE_EXAMPLES)
    random.shuffle(examples)

    n = len(examples)
    valid_n = max(1, int(n * 0.1))
    test_n = max(1, int(n * 0.1))
    train_n = n - valid_n - test_n

    splits = {
        "train": examples[:train_n],
        "valid": examples[train_n:train_n + valid_n],
        "test": examples[train_n + valid_n:],
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, split_examples in splits.items():
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in split_examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  {name}: {len(split_examples)} examples -> {path}")


if __name__ == "__main__":
    build_dataset("./dataset")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/anmolthukral/projects/engineer-model-ft && python3 -m pytest tests/test_build_tone_dataset.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add build_tone_dataset.py tests/test_build_tone_dataset.py
git commit -m "feat: add tone-only dataset generator with topic-variety test"
```

---

## Task 4: Generate the dataset and train the tone-only LoRA adapter

**Files:**
- Create (generated, gitignored): `dataset/train.jsonl`, `dataset/valid.jsonl`, `dataset/test.jsonl`
- Create (generated, gitignored): `mlx_model/adapter_tone/`

**Interfaces:**
- Consumes: `build_tone_dataset.py` (Task 3)
- Produces: `mlx_model/adapter_tone/adapters.safetensors` — consumed by Task 5's merge step.

- [ ] **Step 1: Generate the dataset**

```bash
cd /Users/anmolthukral/projects/engineer-model-ft
python3 build_tone_dataset.py
```

Expected output: three lines showing train/valid/test counts summing to the full `TONE_EXAMPLES` count (30 examples: ~24 train / 3 valid / 3 test).

- [ ] **Step 2: Train the adapter with the memory-safe config**

```bash
mkdir -p mlx_model/adapter_tone
nohup python3 -u -m mlx_lm.lora \
  --model ./mlx_model/base_model \
  --train \
  --data ./dataset \
  --adapter-path ./mlx_model/adapter_tone \
  --iters 300 \
  --learning-rate 1e-5 \
  --batch-size 2 \
  --num-layers 16 \
  --grad-checkpoint \
  --fine-tune-type lora \
  > /tmp/lora_tone_train.log 2>&1 &
echo "started PID $!"
```

- [ ] **Step 3: Watch training to completion**

Run: `tail -f /tmp/lora_tone_train.log` (Ctrl-C once you see `Saved final weights`)
Expected: no `nan`/`inf` in any `Train loss` line; `Iter 1: Val loss` is a normal small number (roughly 0.5-2.0, not `nan`/`inf`); final line is `Saved final weights to mlx_model/adapter_tone/adapters.safetensors.`

- [ ] **Step 4: Verify the adapter file exists**

Run: `ls -la mlx_model/adapter_tone/adapters.safetensors`
Expected: file exists, non-zero size

- [ ] **Step 5: Commit** (dataset/ and mlx_model/ are gitignored — nothing new to add besides the log reference; skip if `git status` is clean)

```bash
git status --short
```

---

## Task 5: Merge the tone adapter, export to GGUF, and register in Ollama

**Files:**
- Modify: `merge_lora.py:16-17` (adapter paths)
- Create: `Modelfile.tone`
- Create (generated, gitignored): `merged_model/`, `engineering-tone-v1.gguf`

**Interfaces:**
- Consumes: `mlx_model/adapter_tone/adapters.safetensors` (Task 4)
- Produces: an Ollama model named `engineering-tone`, consumed by Task 8's query script.

- [ ] **Step 1: Update `merge_lora.py` to point at the new adapter**

In `merge_lora.py`, change:
```python
ADAPTER     = "./mlx_model/adapter_v3/adapters.safetensors"
CONFIG_FILE = "./mlx_model/adapter_v3/adapter_config.json"
```
to:
```python
ADAPTER     = "./mlx_model/adapter_tone/adapters.safetensors"
CONFIG_FILE = "./mlx_model/adapter_tone/adapter_config.json"
```
(If the file still references `adapter_v4` from the prior session, replace whatever the current paths are with `adapter_tone`.)

- [ ] **Step 2: Run the merge**

```bash
rm -rf merged_model
python3 merge_lora.py
```

Expected: output ends with `Merged: 112 layers | Skipped: 0 layers` followed by `Done.` (0 skipped layers confirms every LoRA layer found its matching base-model weight).

- [ ] **Step 3: Convert to GGUF**

```bash
python3 llama.cpp/convert_hf_to_gguf.py ./merged_model --outfile ./engineering-tone-v1.gguf --outtype q8_0
```

Expected: ends with `INFO:hf-to-gguf:Model successfully exported to engineering-tone-v1.gguf`

- [ ] **Step 4: Write `Modelfile.tone`**

```
FROM ./engineering-tone-v1.gguf
TEMPLATE """### User:
{{ .Prompt }}
### Assistant:
"""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "### User:"
PARAMETER stop "### Assistant:"
```

- [ ] **Step 5: Register in Ollama**

```bash
ollama create engineering-tone -f Modelfile.tone
```

Expected: ends with `success`

- [ ] **Step 6: Verify tone across varied, non-JS topics**

```bash
ollama run engineering-tone "How do I stay motivated when learning something new and difficult?"
```

Expected: numbered, step-by-step, encouraging response — same voice as the training examples — on a topic that has nothing to do with code, confirming the model generalized tone rather than memorizing specific example text.

- [ ] **Step 7: Commit**

```bash
git add merge_lora.py Modelfile.tone
git commit -m "feat: train and register the tone-only fine-tune as engineering-tone"
```

---

## Task 6: Write the markdown chunker with unit tests

**Files:**
- Create: `rag/__init__.py` (empty)
- Create: `rag/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Produces: `chunk_markdown(markdown_text: str, source: str, file_path: str, tokenizer, min_tokens: int = 400, max_tokens: int = 700) -> list[dict]`, where each returned dict is `{"source": str, "file_path": str, "heading_trail": str, "content": str, "token_count": int}`.
- Consumed by: Task 8 (`rag/ingest.py`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunking.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/anmolthukral/projects/engineer-model-ft && mkdir -p rag && touch rag/__init__.py && python3 -m pytest tests/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.chunking'`

- [ ] **Step 3: Write `rag/chunking.py`**

```python
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
    to max_tokens per piece."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    pieces, current, current_tokens = [], [], 0
    for p in paragraphs:
        p_tokens = len(tokenizer.encode(p))
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
            for piece in _split_oversized(body, tokenizer, max_tokens):
                chunks.append({
                    "source": source,
                    "file_path": file_path,
                    "heading_trail": trail or "(untitled)",
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/anmolthukral/projects/engineer-model-ft && python3 -m pytest tests/test_chunking.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add rag/__init__.py rag/chunking.py tests/test_chunking.py
git commit -m "feat: add heading-bounded markdown chunker with tests"
```

---

## Task 7: Write the doc scraper

**Files:**
- Create: `rag/scrape_docs.py`

**Interfaces:**
- Produces: `rag/corpus.jsonl` — one JSON object per markdown file: `{"source": str, "file_path": str, "raw_markdown": str}`. Consumed by Task 8's `rag/ingest.py`.

- [ ] **Step 1: Write `rag/scrape_docs.py`**

```python
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
```

- [ ] **Step 2: Run the scraper**

```bash
cd /Users/anmolthukral/projects/engineer-model-ft
python3 rag/scrape_docs.py
```

Expected: this clones ~4 repos (the `mdn/content` clone is the largest, likely several hundred MB even at depth 1 — expect this step to take a few minutes on a normal connection) and ends with `Wrote N documents to rag/corpus.jsonl` where N is at least several thousand.

- [ ] **Step 3: Verify the corpus**

Run: `wc -l rag/corpus.jsonl`
Expected: a count in the thousands (not zero, not suspiciously small)

Run: `python3 -c "import json; d=[json.loads(l) for l in open('rag/corpus.jsonl')]; from collections import Counter; print(Counter(x['source'] for x in d))"`
Expected: all four source names (`mdn`, `react.dev`, `nextjs`, `ts-handbook`) present with non-zero counts

- [ ] **Step 4: Commit**

```bash
git add rag/scrape_docs.py
git commit -m "feat: add doc scraper pulling markdown source from upstream repos"
```

---

## Task 8: Set up the pgvector database schema

**Files:**
- Create: `rag/schema.sql`
- Create: `rag/db.py`

**Interfaces:**
- Produces: `get_connection() -> psycopg.Connection` (reads `RAG_DATABASE_URL` env var, defaults to `dbname=engineer_rag`), and the `chunks` table in Postgres. Consumed by Task 9 (`rag/ingest.py`) and Task 10 (`rag/query.py`).

- [ ] **Step 1: Write `rag/schema.sql`**

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    file_path TEXT NOT NULL,
    heading_trail TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    UNIQUE (source, file_path, heading_trail)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
```

- [ ] **Step 2: Create the database and apply the schema**

```bash
createdb engineer_rag
psql -d engineer_rag -f rag/schema.sql
```

Expected: `CREATE EXTENSION`, `CREATE TABLE`, `CREATE INDEX` (no errors)

- [ ] **Step 3: Write `rag/db.py`**

```python
"""Shared Postgres connection helper for the RAG pipeline."""
import os
import psycopg


def get_connection() -> psycopg.Connection:
    dsn = os.environ.get("RAG_DATABASE_URL", "dbname=engineer_rag")
    return psycopg.connect(dsn)
```

- [ ] **Step 4: Verify the connection works**

Run: `python3 -c "from rag.db import get_connection; conn = get_connection(); print(conn.execute('SELECT 1').fetchone()); conn.close()"`
Expected: `(1,)`

- [ ] **Step 5: Commit**

```bash
git add rag/schema.sql rag/db.py
git commit -m "feat: add pgvector schema and connection helper"
```

---

## Task 9: Write the ingestion script (chunk + embed + store)

**Files:**
- Create: `rag/ingest.py`

**Interfaces:**
- Consumes: `rag/corpus.jsonl` (Task 7), `chunk_markdown` (Task 6), `get_connection` (Task 8)
- Produces: populated `chunks` table in Postgres. Consumed by Task 10 (`rag/query.py`).

- [ ] **Step 1: Pull the embedding model**

```bash
ollama pull nomic-embed-text
```

- [ ] **Step 2: Write `rag/ingest.py`**

```python
#!/usr/bin/env python3
"""Read rag/corpus.jsonl, chunk each document, embed each chunk via
Ollama's local nomic-embed-text model, and store into pgvector. Uses
ON CONFLICT DO NOTHING against the (source, file_path, heading_trail)
unique constraint so re-running after an interruption is safe."""
import json
import sys
from pathlib import Path

import requests
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.chunking import chunk_markdown
from rag.db import get_connection

CORPUS_FILE = Path("./rag/corpus.jsonl")
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


def embed(text: str) -> list[float]:
    resp = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "input": text})
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def main():
    tokenizer = AutoTokenizer.from_pretrained("./mlx_model/base_model")
    conn = get_connection()
    cur = conn.cursor()

    total_docs, total_chunks, inserted = 0, 0, 0
    with open(CORPUS_FILE) as f:
        for line in f:
            doc = json.loads(line)
            total_docs += 1
            chunks = chunk_markdown(
                doc["raw_markdown"], source=doc["source"], file_path=doc["file_path"],
                tokenizer=tokenizer, min_tokens=400, max_tokens=700,
            )
            for chunk in chunks:
                total_chunks += 1
                vector = embed(chunk["content"])
                cur.execute(
                    """
                    INSERT INTO chunks (source, file_path, heading_trail, content, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (source, file_path, heading_trail) DO NOTHING
                    """,
                    (chunk["source"], chunk["file_path"], chunk["heading_trail"],
                     chunk["content"], vector),
                )
                if cur.rowcount > 0:
                    inserted += 1
            if total_docs % 200 == 0:
                conn.commit()
                print(f"  ...{total_docs} docs processed, {inserted} chunks inserted so far")

    conn.commit()
    cur.close()
    conn.close()
    print(f"\nDone. {total_docs} documents -> {total_chunks} chunks -> {inserted} newly inserted rows.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run ingestion**

```bash
cd /Users/anmolthukral/projects/engineer-model-ft
python3 rag/ingest.py
```

Expected: progress lines every 200 documents, ending with `Done. N documents -> M chunks -> M newly inserted rows.` This step embeds potentially tens of thousands of chunks one at a time through Ollama — expect it to take a while (likely tens of minutes); it's safe to interrupt and re-run since inserts are idempotent.

- [ ] **Step 4: Verify row count in Postgres**

Run: `psql -d engineer_rag -c "SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source;"`
Expected: all four sources present with non-zero row counts

- [ ] **Step 5: Commit**

```bash
git add rag/ingest.py
git commit -m "feat: add chunk+embed+store ingestion pipeline"
```

---

## Task 10: Write the RAG query script

**Files:**
- Create: `rag/query.py`

**Interfaces:**
- Consumes: `get_connection` (Task 8), populated `chunks` table (Task 9), the `engineering-tone` Ollama model (Task 5)
- Produces: a CLI entry point — end-user-facing deliverable of this whole plan.

- [ ] **Step 1: Write `rag/query.py`**

```python
#!/usr/bin/env python3
"""Retrieve-then-generate RAG query script: embed the question, pull the
top-5 most relevant doc chunks from pgvector, and ask the tone-fine-tuned
model to answer using them as context."""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.db import get_connection

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
EMBED_MODEL = "nomic-embed-text"
TONE_MODEL = "engineering-tone"
TOP_K = 5


def embed(text: str) -> list[float]:
    resp = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "input": text})
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    vector = embed(question)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT source, file_path, heading_trail, content
        FROM chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (vector, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {"source": r[0], "file_path": r[1], "heading_trail": r[2], "content": r[3]}
        for r in rows
    ]


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = "\n\n".join(
        f"[{c['source']} — {c['heading_trail']}]\n{c['content']}" for c in chunks
    )
    return (
        f"### User:\nUse the following documentation excerpts to answer the question. "
        f"If the excerpts don't cover it, say so rather than guessing.\n\n"
        f"{context_blocks}\n\nQuestion: {question}\n### Assistant:\n"
    )


def ask(question: str) -> tuple[str, list[dict]]:
    chunks = retrieve(question)
    prompt = build_prompt(question, chunks)
    # raw=True is required here: build_prompt() already includes the full
    # "### User:/### Assistant:" markers itself. Without raw=True, Ollama
    # would additionally apply Modelfile.tone's own TEMPLATE on top of this
    # already-formatted string, double-wrapping the markers and producing
    # a garbled prompt the model was never trained on.
    resp = requests.post(
        OLLAMA_GENERATE_URL,
        json={"model": TONE_MODEL, "prompt": prompt, "stream": False, "raw": True},
    )
    resp.raise_for_status()
    return resp.json()["response"], chunks


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 rag/query.py '<question>'")
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    answer, chunks = ask(question)
    print(answer)
    print("\n--- sources ---")
    for c in chunks:
        print(f"  {c['source']}: {c['file_path']} ({c['heading_trail']})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a real query**

```bash
cd /Users/anmolthukral/projects/engineer-model-ft
python3 rag/query.py "How does the useEffect cleanup function work?"
```

Expected: a step-by-step, tutor-voiced answer, followed by a `--- sources ---` list where at least one source is `react.dev` or `mdn` with a heading trail plausibly related to `useEffect`.

- [ ] **Step 3: Run a second query on a different topic to confirm retrieval isn't stuck on one source**

```bash
python3 rag/query.py "What's the difference between let and const in JavaScript?"
```

Expected: sources this time skew toward `mdn` (JavaScript reference), different from the previous query's sources — confirms retrieval is actually responding to the question rather than always returning the same rows.

- [ ] **Step 4: Commit**

```bash
git add rag/query.py
git commit -m "feat: add RAG query script (retrieve top-5, generate via tone model)"
```

---

## Task 11: Minimal Streamlit chat UI

**Files:**
- Create: `chat_app.py`

**Interfaces:**
- Consumes: `retrieve()`, `build_prompt()`, `OLLAMA_GENERATE_URL`, `TONE_MODEL` from `rag/query.py` (Task 10) — the UI reuses the exact same retrieve-then-generate logic as the CLI, no duplicated RAG code.
- Produces: a browser-based chat interface at `localhost:8501`.

Scope note: each turn's retrieval and generation is based only on that turn's question (no growing conversation context fed back into the model) — the UI still displays full visual history for readability, but keeping the RAG/generation logic stateless per-turn avoids the added complexity of query reformulation or context-window management, consistent with "nothing fancy, minimal."

- [ ] **Step 1: Write `chat_app.py`**

```python
#!/usr/bin/env python3
"""Minimal local chat UI for the RAG-augmented tone model. Reuses the same
retrieve()/build_prompt() logic rag/query.py uses for the CLI, so the UI
and the CLI share one source of truth for the RAG flow instead of
duplicating it."""
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag.query import retrieve, build_prompt, OLLAMA_GENERATE_URL, TONE_MODEL

st.set_page_config(page_title="Engineering Tutor", page_icon="🎓")
st.title("Engineering Tutor")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about JS, React, Next.js, or TypeScript...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Looking up docs and thinking..."):
            chunks = retrieve(question)
            prompt = build_prompt(question, chunks)
            # raw=True: prompt already has the full "### User:/### Assistant:"
            # markers baked in — see the note in rag/query.py's ask().
            resp = requests.post(
                OLLAMA_GENERATE_URL,
                json={"model": TONE_MODEL, "prompt": prompt, "stream": False, "raw": True},
            )
            resp.raise_for_status()
            answer = resp.json()["response"]
        st.markdown(answer)
        with st.expander("Sources"):
            for c in chunks:
                st.caption(f"{c['source']}: {c['file_path']} ({c['heading_trail']})")

    st.session_state.messages.append({"role": "assistant", "content": answer})
```

- [ ] **Step 2: Run the app**

```bash
cd /Users/anmolthukral/projects/engineer-model-ft
streamlit run chat_app.py
```

Expected: browser opens automatically to `http://localhost:8501` showing a chat input box and the "Engineering Tutor" title.

- [ ] **Step 3: Verify a real exchange in the browser**

Type "How does the useEffect cleanup function work?" into the chat input.
Expected: a user chat bubble appears with the question, then an assistant bubble with a step-by-step tutor-voiced answer, and a collapsible "Sources" section listing relevant `react.dev`/`mdn` entries — matching what `rag/query.py` produces on the same question (Task 10, Step 2).

- [ ] **Step 4: Commit**

```bash
git add chat_app.py requirements-rag.txt
git commit -m "feat: add minimal Streamlit chat UI over the RAG pipeline"
```

---

## Task 12: End-to-end validation checklist

**Files:** none (manual verification task)

**Interfaces:**
- Consumes: everything from Tasks 1-11.
- Produces: a documented comparison confirming the hybrid system meets the spec's goal.

- [ ] **Step 1: Run 5 varied technical questions through `rag/query.py`** (or type them into `chat_app.py` from Task 11 — both use the identical retrieve/generate logic, so either is a valid way to run this check)

Suggested questions covering all four doc sources:
```bash
python3 rag/query.py "How do I flatten a nested array in JavaScript?"
python3 rag/query.py "What is the difference between the app router and pages router in Next.js?"
python3 rag/query.py "What does the TypeScript satisfies operator do?"
python3 rag/query.py "How do I use the useState hook for an object?"
python3 rag/query.py "What's the difference between position: absolute and position: fixed in CSS?"
```

For each, manually confirm: (a) the answer is in the trained tutor tone (step-by-step, encouraging), (b) the listed sources are topically relevant to the question, (c) the technical content in the answer matches what's in the printed source excerpts rather than contradicting them.

- [ ] **Step 2: Run one of the same questions directly against the tone-only model with no retrieval, for comparison**

```bash
ollama run engineering-tone "What does the TypeScript satisfies operator do?"
```

Compare this against the RAG-augmented answer from Step 1 — the RAG version should be more specific/accurate since it's grounded in the actual TypeScript Handbook text, while the tone-only model is relying purely on whatever it picked up from pretraining plus the tone examples (which never mentioned `satisfies`).

- [ ] **Step 3: Record the outcome**

Add a short "Validation results" note at the bottom of the spec file (`docs/superpowers/specs/2026-08-20-tone-finetune-rag-hybrid-design.md`) summarizing what was observed in Steps 1-2 — this closes the loop on the original question that started this whole effort (fine-tuning vs. RAG accuracy).

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-08-20-tone-finetune-rag-hybrid-design.md
git commit -m "docs: record end-to-end validation results"
```
