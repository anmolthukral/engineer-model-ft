#!/usr/bin/env python3
"""
Download, filter, and combine the best public datasets for fine-tuning
an engineering teacher model focused on React, JavaScript, and frontend dev.

Datasets used:
  1. cfahlgren1/react-code-instructions  — React-specific code instructions (MIT)
  2. sahil2801/CodeAlpaca-20k            — General coding Q&A (filter JS/React)
  3. ali-alkhars/interviews              — SWE interview Q&A (React, JS, Angular)
  4. axay/javascript-dataset             — JavaScript focused Q&A
  5. ./clean_data (local)                — Our hand-crafted teacher-style examples

Output: ./dataset/train.jsonl, valid.jsonl, test.jsonl
"""

import json
import random
import re
from pathlib import Path

from datasets import load_dataset

SEED = 42
random.seed(SEED)

JS_REACT_KEYWORDS = re.compile(
    r"\b(javascript|typescript|react|jsx|tsx|frontend|html|css|"
    r"node|npm|webpack|vite|nextjs|next\.js|vue|angular|redux|"
    r"hooks?|component|async|await|promise|closure|prototype|"
    r"dom|event\s*loop|es6|es2015|arrow\s*function|destructur|"
    r"spread|rest\s*param|module|import|export|fetch|api|"
    r"useState|useEffect|useCallback|useMemo|context|prop)\b",
    re.IGNORECASE,
)

# mlx_model/adapter_v3 was trained with max_seq_length=2048 tokens (~4 chars/token
# for code). Examples longer than this get force-truncated mid-token during training
# (some source examples are already truncated mid-tag before we even see them), and
# their outsized loss under mask_prompt=false destabilizes training. Cap well below
# the token limit so nothing gets silently cut.
MAX_EXAMPLE_CHARS = 6000

def is_js_react(text: str) -> bool:
    return bool(JS_REACT_KEYWORDS.search(text))

def is_reasonable_length(user: str, assistant: str) -> bool:
    return len(user) + len(assistant) <= MAX_EXAMPLE_CHARS

def to_training_example(user: str, assistant: str) -> dict:
    return {"text": f"### User:\n{user.strip()}\n### Assistant:\n{assistant.strip()}"}

def load_local_clean_data(path="./clean_data/train.jsonl") -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"  [skip] {path} not found")
        return []
    examples = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    print(f"  local clean_data: {len(examples)} examples")
    return examples

def load_react_code_instructions(max_samples=300) -> list[dict]:
    """cfahlgren1/react-code-instructions — React app code generation examples."""
    print("\nLoading cfahlgren1/react-code-instructions...")
    try:
        ds = load_dataset("cfahlgren1/react-code-instructions", split="train")
        examples = []
        for row in ds:
            # Dataset has messages in conversations format or separate columns
            # Try common column names
            user = row.get("user") or row.get("prompt") or row.get("instruction") or ""
            assistant = row.get("assistant") or row.get("response") or row.get("output") or ""

            # Some versions use a 'messages' list
            if not user and "messages" in row:
                msgs = row["messages"]
                user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
                asst_msgs = [m["content"] for m in msgs if m.get("role") == "assistant"]
                user = user_msgs[0] if user_msgs else ""
                assistant = asst_msgs[0] if asst_msgs else ""

            if user and assistant and len(assistant) > 50 and is_reasonable_length(user, assistant):
                examples.append(to_training_example(user, assistant))

        sampled = random.sample(examples, min(max_samples, len(examples)))
        print(f"  ✅ {len(sampled)} React examples")
        return sampled
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return []

def load_code_alpaca(max_samples=300) -> list[dict]:
    """sahil2801/CodeAlpaca-20k — filter for JS/React/frontend content."""
    print("\nLoading sahil2801/CodeAlpaca-20k...")
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        js_examples = []
        for row in ds:
            instruction = row.get("instruction", "")
            inp = row.get("input", "")
            output = row.get("output", "")
            user = f"{instruction}\n\n{inp}".strip() if inp else instruction
            if is_js_react(user + output) and len(output) > 80 and is_reasonable_length(user, output):
                js_examples.append(to_training_example(user, output))

        sampled = random.sample(js_examples, min(max_samples, len(js_examples)))
        print(f"  ✅ {len(sampled)} JS/React examples (from {len(js_examples)} matches)")
        return sampled
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return []

def load_interviews(max_samples=200) -> list[dict]:
    """ali-alkhars/interviews — SWE interview Q&A covering React, JS, Angular."""
    print("\nLoading ali-alkhars/interviews...")
    try:
        ds = load_dataset("ali-alkhars/interviews", split="train")
        examples = []
        for row in ds:
            # Try common column names for this dataset
            question = (
                row.get("question") or row.get("instruction") or
                row.get("input") or row.get("prompt") or ""
            )
            answer = (
                row.get("answer") or row.get("output") or
                row.get("response") or row.get("text") or ""
            )
            if question and answer and is_js_react(question + answer) and len(answer) > 60 and is_reasonable_length(question, answer):
                examples.append(to_training_example(question, answer))

        sampled = random.sample(examples, min(max_samples, len(examples)))
        print(f"  ✅ {len(sampled)} interview examples")
        return sampled
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return []

def load_javascript_dataset(max_samples=200) -> list[dict]:
    """axay/javascript-dataset — JavaScript Q&A."""
    print("\nLoading axay/javascript-dataset...")
    try:
        ds = load_dataset("axay/javascript-dataset", split="train")
        examples = []
        for row in ds:
            question = (
                row.get("question") or row.get("instruction") or
                row.get("input") or row.get("prompt") or ""
            )
            answer = (
                row.get("answer") or row.get("output") or
                row.get("response") or ""
            )
            if question and answer and len(answer) > 60 and is_reasonable_length(question, answer):
                examples.append(to_training_example(question, answer))

        sampled = random.sample(examples, min(max_samples, len(examples)))
        print(f"  ✅ {len(sampled)} JS examples")
        return sampled
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return []

def load_hf_code_alpaca_h4(max_samples=200) -> list[dict]:
    """HuggingFaceH4/CodeAlpaca_20K — another well-curated coding instruction set."""
    print("\nLoading HuggingFaceH4/CodeAlpaca_20K...")
    try:
        ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")
        js_examples = []
        for row in ds:
            prompt = row.get("prompt", "")
            completion = row.get("completion", "")
            if is_js_react(prompt + completion) and len(completion) > 80 and is_reasonable_length(prompt, completion):
                js_examples.append(to_training_example(prompt, completion))

        sampled = random.sample(js_examples, min(max_samples, len(js_examples)))
        print(f"  ✅ {len(sampled)} JS/React examples")
        return sampled
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return []

def write_splits(all_examples: list[dict], output_dir: str, batch_size: int = 4):
    """Split into train/valid/test ensuring valid and test have >= batch_size examples."""
    random.shuffle(all_examples)
    n = len(all_examples)

    # Guarantee at least batch_size in valid and test
    min_eval = batch_size
    remaining = n - 2 * min_eval
    train_n = max(min_eval, int(remaining * 0.9))
    valid_n = min_eval + max(0, (n - train_n - min_eval) // 2)
    test_n = n - train_n - valid_n

    splits = {
        "train": all_examples[:train_n],
        "valid": all_examples[train_n:train_n + valid_n],
        "test": all_examples[train_n + valid_n:],
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, examples in splits.items():
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  {name}: {len(examples)} examples → {path}")

    print(f"\n  Total: {n} examples")

def main():
    print("=" * 60)
    print("Building engineering-teacher fine-tune dataset")
    print("=" * 60)

    all_examples = []

    # 1. Our hand-crafted teacher-style examples (highest weight — keep all)
    all_examples += load_local_clean_data("./clean_data/train.jsonl")
    all_examples += load_local_clean_data("./clean_data/valid.jsonl")
    all_examples += load_local_clean_data("./clean_data/test.jsonl")

    # 2. Public datasets
    all_examples += load_react_code_instructions(max_samples=300)
    all_examples += load_code_alpaca(max_samples=300)
    all_examples += load_interviews(max_samples=200)
    all_examples += load_javascript_dataset(max_samples=200)
    all_examples += load_hf_code_alpaca_h4(max_samples=200)

    # Deduplicate by text
    seen = set()
    unique = []
    for ex in all_examples:
        key = ex["text"][:120]
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    # Token-based length filter. MAX_EXAMPLE_CHARS is a cheap proxy applied per-loader,
    # but char count is a poor stand-in for token count: dense/minified code and CJK
    # text can run as low as ~2.2 chars/token (vs. the ~4 assumed), so examples that
    # passed the char cap can still exceed max_seq_length=2048 once tokenized. Those
    # get force-truncated mid-response during training and destabilize the batch.
    # Drop rather than truncate — a truncated code example just teaches the model to
    # produce broken code.
    print("\nApplying token-based length filter...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("./mlx_model/base_model")
    # This machine has 18GB total unified memory. mlx_lm's LoRA trainer was
    # measured hitting 30-57GB of peak allocation (severe swap/overcommit)
    # on batches with ~1000+ token sequences, which silently corrupts
    # Metal computations (inf/nan) instead of erroring cleanly. Capping
    # tightly here (combined with a smaller batch size and gradient
    # checkpointing at train time) keeps peak memory within physical RAM.
    MAX_TOKENS = 768
    before = len(unique)
    unique = [ex for ex in unique if len(tokenizer(ex["text"])["input_ids"]) <= MAX_TOKENS]
    print(f"  Dropped {before - len(unique)} examples exceeding {MAX_TOKENS} tokens")

    print(f"\n📊 Total unique examples: {len(unique)}")
    print("Writing dataset splits...")
    write_splits(unique, "./dataset", batch_size=4)

    print("\n✅ Done! Next step:")
    print("   mlx_lm.lora \\")
    print("     --model ./mlx_model/base_model \\")
    print("     --train \\")
    print("     --data ./dataset \\")
    print("     --adapter-path ./mlx_model/adapter_v4 \\")
    print("     --iters 500 \\")
    print("     --learning-rate 1e-5 \\")  # mlx_lm's own default; 1e-4 diverges
    print("     --batch-size 2 \\")  # this machine has 18GB RAM; batch 4 overruns it
    print("     --num-layers 16 \\")
    print("     --grad-checkpoint \\")  # required to keep peak memory in budget
    print("     --fine-tune-type lora")

if __name__ == "__main__":
    main()
