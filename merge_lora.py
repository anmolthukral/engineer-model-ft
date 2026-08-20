#!/usr/bin/env python3
"""
Merge MLX LoRA adapter directly into the original HuggingFace model weights.
Bypasses mlx_lm.fuse entirely — produces a clean HF model ready for GGUF conversion.

MLX LoRA forward: output = x @ W.T + scale * (x @ lora_a) @ lora_b
Merge formula:    W_merged = W + scale * lora_b.T @ lora_a.T
"""

import json
import torch
import safetensors.torch as st
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL  = "./qwen2.5-coder-1.5b-hf"
ADAPTER     = "./mlx_model/adapter_tone/adapters.safetensors"
CONFIG_FILE = "./mlx_model/adapter_tone/adapter_config.json"
OUTPUT_DIR  = "./merged_model"

def main():
    # Load adapter config
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    scale = cfg["lora_parameters"]["scale"]
    print(f"LoRA scale: {scale}")

    # Load adapter weights (MLX safetensors)
    print("Loading adapter weights...")
    adapter = st.load_file(ADAPTER)
    print(f"  {len(adapter)} adapter tensors")

    # Load base model
    print(f"Loading base model from {BASE_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Build a map of adapter keys → base model parameter names
    # MLX key:  model.layers.N.self_attn.q_proj.lora_a
    # HF key:   model.layers.N.self_attn.q_proj.weight
    lora_pairs = {}
    for key in adapter:
        if key.endswith(".lora_a"):
            base_key = key[: -len(".lora_a")]
            lora_pairs[base_key] = {"lora_a": key, "lora_b": base_key + ".lora_b"}

    print(f"  Found {len(lora_pairs)} LoRA layer pairs")

    # Merge: W_merged = W + scale * lora_b.T @ lora_a.T
    merged = 0
    skipped = 0
    state_dict = dict(model.named_parameters())

    for base_key, keys in lora_pairs.items():
        hf_key = base_key + ".weight"
        if hf_key not in state_dict:
            skipped += 1
            continue

        lora_a = adapter[keys["lora_a"]].float()  # (in, rank)
        lora_b = adapter[keys["lora_b"]].float()  # (rank, out)
        delta = scale * (lora_b.T @ lora_a.T)    # (out, in)

        param = state_dict[hf_key]
        if param.shape != delta.shape:
            print(f"  Shape mismatch {hf_key}: {param.shape} vs {delta.shape} — skipping")
            skipped += 1
            continue

        param.data += delta
        merged += 1

    print(f"  Merged: {merged} layers | Skipped: {skipped} layers")

    # Save merged model in HF format
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
