#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-Coder-1.5B on MacBook M3 Pro using MLX (Apple Silicon native)
Safe, efficient, won't harm your Mac - uses unified memory, Metal GPU acceleration.
"""

import os
import json
import argparse
from pathlib import Path

# Check MLX availability
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    from mlx_lm.tuner import train
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Installing MLX...")
    os.system("python3 -m pip install mlx mlx-lm datasets huggingface_hub -q")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    from mlx_lm.tuner import train
    MLX_AVAILABLE = True

from datasets import load_dataset
from huggingface_hub import snapshot_download


def check_system():
    """Verify we're on Apple Silicon with enough memory."""
    import platform
    import psutil
    
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    if platform.processor() != 'arm':
        print("⚠️  Warning: Not on Apple Silicon. MLX works best on M-series chips.")
    
    mem_gb = psutil.virtual_memory().total / (1024**3)
    if mem_gb < 16:
        print(f"⚠️  Warning: Only {mem_gb:.1f}GB RAM. May need smaller batch size.")
    
    return True


def prepare_data(data_source, output_dir, num_samples=1000):
    """Prepare training data in MLX format (separate train/valid/test jsonl files)."""
    print(f"\n📥 Preparing training data from {data_source}...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all examples
    if data_source == "angular":
        # Use your Angular datasets from HF
        datasets_to_use = [
            "rsh-raj/angular-commits",
            "rsh-raj/angular-cli-commits",
        ]
        all_examples = []
        for ds_name in datasets_to_use:
            try:
                ds = load_dataset(ds_name, split=f"train[:{num_samples//len(datasets_to_use)}]")
                for ex in ds:
                    if "message" in ex and "diff" in ex:
                        text = f"<|im_start|>user\n{ex['message']}<|im_end|>\n<|im_start|>assistant\n{ex['diff']}<|im_end|>"
                        all_examples.append({"text": text})
            except Exception as e:
                print(f"  Could not load {ds_name}: {e}")
        
        if not all_examples:
            print("  Falling back to synthetic data...")
            return prepare_data("synthetic", output_dir, num_samples)
    
    elif data_source == "interviews":
        # Startup interviews - instruction/response format (clean data)
        all_examples = []
        
        print("  Loading Glavin001/startup-interviews...")
        try:
            ds = load_dataset("Glavin001/startup-interviews", split=f"train[:{num_samples}]")
            for ex in ds:
                if "instruction" in ex and "output" in ex:
                    user_text = ex["instruction"]
                    if "input" in ex and ex["input"]:
                        user_text += f"\n\n{ex['input']}"
                    text = f"### User:\n{user_text}\n### Assistant:\n{ex['output']}"
                    all_examples.append({"text": text})
            print(f"  ✅ Loaded {len(all_examples)} startup interview examples")
        except Exception as e:
            print(f"  Could not load startup-interviews: {e}")
        
        if not all_examples:
            print("  Falling back to synthetic data...")
            return prepare_data("synthetic", output_dir, num_samples)

    elif data_source == "mixed":
        # Mix coding synthetic data with startup interviews
        all_examples = []
        
        # 1. Synthetic coding examples (70%)
        coding_samples = int(num_samples * 0.7)
        print(f"  Generating {coding_samples} synthetic coding examples...")
        coding_examples = [
            {"text": "### User:\nWrite a Python function for binary search\n### Assistant:\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```"},
            {"text": "### User:\nCreate a React hook for debouncing\n### Assistant:\n```typescript\nimport { useState, useEffect } from 'react';\n\nfunction useDebounce<T>(value: T, delay: number): T {\n  const [debouncedValue, setDebouncedValue] = useState<T>(value);\n\n  useEffect(() => {\n    const handler = setTimeout(() => setDebouncedValue(value), delay);\n    return () => clearTimeout(handler);\n  }, [value, delay]);\n\n  return debouncedValue;\n}\n```"},
            {"text": "### User:\nExplain async/await in JavaScript with example\n### Assistant:\nAsync/await is syntactic sugar over Promises that makes asynchronous code look synchronous:\n\n```javascript\nasync function fetchUserData(userId) {\n  try {\n    const response = await fetch('/api/users/' + userId);\n    if (!response.ok) throw new Error('User not found');\n    const user = await response.json();\n    return user;\n  } catch (error) {\n    console.error('Failed to fetch user:', error);\n    throw error;\n  }\n}\n```\n\nThe await pauses execution until the Promise resolves, but doesn't block the main thread."},
            {"text": "### User:\nWrite a Python decorator for retry with exponential backoff\n### Assistant:\n```python\nimport time\nimport random\nfrom functools import wraps\nfrom typing import Callable, Any\n\ndef retry_with_backoff(\n    max_retries: int = 3,\n    base_delay: float = 1.0,\n    max_delay: float = 60.0,\n    exponential_base: float = 2.0,\n    jitter: bool = True\n):\n    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:\n        @wraps(func)\n        def wrapper(*args, **kwargs) -> Any:\n            for attempt in range(max_retries + 1):\n                try:\n                    return func(*args, **kwargs)\n                except Exception as e:\n                    if attempt == max_retries:\n                        raise\n                    delay = min(base_delay * (exponential_base ** attempt), max_delay)\n                    if jitter:\n                        delay *= (0.5 + random.random())\n                    time.sleep(delay)\n            return wrapper\n        return decorator\n    return decorator\n```"},
            {"text": "### User:\nImplement a thread-safe singleton in Python\n### Assistant:\n```python\nimport threading\nfrom typing import TypeVar, Optional\n\nT = TypeVar('T', bound='Singleton')\n\nclass Singleton:\n    _instance: Optional[T] = None\n    _lock = threading.Lock()\n    \n    def __new__(cls: type[T]) -> T:\n        if cls._instance is None:\n            with cls._lock:\n                if cls._instance is None:\n                    cls._instance = super().__new__(cls)\n        return cls._instance\n    \n    def __init__(self):\n        if not hasattr(self, '_initialized'):\n            self._initialized = True\n```"},
            {"text": "### User:\nWrite a TypeScript type for deep partial\n### Assistant:\n```typescript\n// Makes all properties (including nested) optional\ntype DeepPartial<T> = {\n  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];\n};\n\n// Usage\ninterface User {\n  name: string;\n  address: {\n    city: string;\n    zip: string;\n  };\n}\n\ntype PartialUser = DeepPartial<User>;\n// Allows: { address: { city: 'NYC' } }\n```"},
        ]
        coding_examples = (coding_examples * (coding_samples // len(coding_examples) + 1))[:coding_samples]
        all_examples.extend(coding_examples)
        
        # 2. Startup interviews (30%)
        interview_samples = num_samples - coding_samples
        print(f"  Loading {interview_samples} startup interview examples...")
        try:
            ds = load_dataset("Glavin001/startup-interviews", split=f"train[:{interview_samples}]")
            for ex in ds:
                if "instruction" in ex and "output" in ex:
                    user_text = ex["instruction"]
                    if "input" in ex and ex["input"]:
                        user_text += f"\n\n{ex['input']}"
                    text = f"### User:\n{user_text}\n### Assistant:\n{ex['output']}"
                    all_examples.append({"text": text})
            print(f"  ✅ Loaded {len(ds)} startup interview examples")
        except Exception as e:
            print(f"  Could not load startup-interviews: {e}")
        
        if not all_examples:
            print("  Falling back to synthetic data...")
            return prepare_data("synthetic", output_dir, num_samples)
    
    elif data_source == "synthetic":
        # High-quality synthetic coding examples
        all_examples = [
            {"text": "### User:\nWrite a Python function for binary search\n### Assistant:\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```"},
            {"text": "### User:\nCreate a React hook for debouncing\n### Assistant:\n```typescript\nimport { useState, useEffect } from 'react';\n\nfunction useDebounce<T>(value: T, delay: number): T {\n  const [debouncedValue, setDebouncedValue] = useState<T>(value);\n\n  useEffect(() => {\n    const handler = setTimeout(() => setDebouncedValue(value), delay);\n    return () => clearTimeout(handler);\n  }, [value, delay]);\n\n  return debouncedValue;\n}\n```"},
            {"text": "### User:\nExplain async/await in JavaScript with example\n### Assistant:\nAsync/await is syntactic sugar over Promises that makes asynchronous code look synchronous:\n\n```javascript\nasync function fetchUserData(userId) {\n  try {\n    const response = await fetch(\`/api/users/\${userId}\`);\n    if (!response.ok) throw new Error('User not found');\n    const user = await response.json();\n    return user;\n  } catch (error) {\n    console.error('Failed to fetch user:', error);\n    throw error;\n  }\n}\n```\n\nThe \`await\` pauses execution until the Promise resolves, but doesn't block the main thread."},
            {"text": "### User:\nWrite a Python decorator for retry with exponential backoff\n### Assistant:\n```python\nimport time\nimport random\nfrom functools import wraps\nfrom typing import Callable, Any\n\ndef retry_with_backoff(\n    max_retries: int = 3,\n    base_delay: float = 1.0,\n    max_delay: float = 60.0,\n    exponential_base: float = 2.0,\n    jitter: bool = True\n):\n    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:\n        @wraps(func)\n        def wrapper(*args, **kwargs) -> Any:\n            for attempt in range(max_retries + 1):\n                try:\n                    return func(*args, **kwargs)\n                except Exception as e:\n                    if attempt == max_retries:\n                        raise\n                    delay = min(base_delay * (exponential_base ** attempt), max_delay)\n                    if jitter:\n                        delay *= (0.5 + random.random())\n                    time.sleep(delay)\n            return wrapper\n        return decorator\n    return decorator\n```"},
            {"text": "### User:\nImplement a thread-safe singleton in Python\n### Assistant:\n```python\nimport threading\nfrom typing import TypeVar, Optional\n\nT = TypeVar('T', bound='Singleton')\n\nclass Singleton:\n    _instance: Optional[T] = None\n    _lock = threading.Lock()\n    \n    def __new__(cls: type[T]) -> T:\n        if cls._instance is None:\n            with cls._lock:\n                if cls._instance is None:\n                    cls._instance = super().__new__(cls)\n        return cls._instance\n    \n    def __init__(self):\n        if not hasattr(self, '_initialized'):\n            self._initialized = True\n            # Initialize here\n```"},
            {"text": "### User:\nWrite a TypeScript type for deep partial\n### Assistant:\n```typescript\n// Makes all properties (including nested) optional\ntype DeepPartial<T> = {\n  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];\n};\n\n// Usage\ninterface User {\n  name: string;\n  address: {\n    city: string;\n    zip: string;\n  };\n}\n\ntype PartialUser = DeepPartial<User>;\n// Allows: { address: { city: 'NYC' } }\n```"},
        ]
        # Replicate to reach num_samples
        all_examples = (all_examples * (num_samples // len(all_examples) + 1))[:num_samples]
    
    else:
        raise ValueError(f"Unknown data source: {data_source}")
    
    # Split: 80% train, 10% valid, 10% test
    n = len(all_examples)
    train_n = int(n * 0.8)
    valid_n = int(n * 0.1)
    
    train_examples = all_examples[:train_n]
    valid_examples = all_examples[train_n:train_n + valid_n]
    test_examples = all_examples[train_n + valid_n:]
    
    # Write separate files
    for name, examples in [("train", train_examples), ("valid", valid_examples), ("test", test_examples)]:
        path = output_dir / f"{name}.jsonl"
        with open(path, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')
        print(f"  ✅ {name}: {len(examples)} examples → {path}")
    
    return output_dir


def convert_model(model_name, output_dir):
    """Convert HF model to MLX format."""
    print(f"\n🔄 Converting {model_name} to MLX format...")
    
    output_path = Path(output_dir)
    if (output_path / "tokenizer.json").exists():
        print(f"  Already converted at {output_path}")
        return output_path
    
    from mlx_lm.convert import convert
    convert(model_name, str(output_path))
    print(f"  ✅ Converted to {output_path}")
    return output_path


def train_lora(model_path, data_dir, adapter_path, config):
    """Train LoRA adapter using MLX."""
    print(f"\n🏋️  Starting LoRA fine-tuning...")
    print(f"  Model: {model_path}")
    print(f"  Data: {data_dir}")
    print(f"  Adapter output: {adapter_path}")
    print(f"  Config: {config}")
    
    # MLX training command
    import subprocess
    cmd = [
        "python3", "-m", "mlx_lm.lora",
        "--model", str(model_path),
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(adapter_path),
        "--iters", str(config["iters"]),
        "--learning-rate", str(config["lr"]),
        "--batch-size", str(config["batch_size"]),
        "--num-layers", str(config["lora_layers"]),
        "--fine-tune-type", "lora",
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ❌ Training failed:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    print(f"  ✅ Training complete! Adapter saved to {adapter_path}")
    return True


def test_model(model_path, adapter_path, prompts):
    """Test the fine-tuned model."""
    print(f"\n🧪 Testing model...")
    
    if adapter_path and Path(adapter_path).exists():
        model, tokenizer = load(str(model_path), adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(str(model_path))
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt[:80]}...")
        
        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9
        )
        print(f"Response: {response[:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-Coder-1.5B on Mac with MLX")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B", help="HF model name")
    parser.add_argument("--data", choices=["angular", "synthetic", "interviews", "mixed"], default="angular", help="Training data source")
    parser.add_argument("--samples", type=int, default=500, help="Number of training samples")
    parser.add_argument("--iters", type=int, default=200, help="Training iterations")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-layers", type=int, default=16, help="Number of LoRA layers")
    parser.add_argument("--output-dir", default="./mlx_model", help="Output directory")
    parser.add_argument("--test-only", action="store_true", help="Only test existing adapter")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🍎 MLX Fine-tuning on MacBook M3 Pro")
    print("=" * 60)
    
    # Safety check
    check_system()
    
    # Paths
    base_dir = Path(args.output_dir)
    model_dir = base_dir / "base_model"
    data_dir = base_dir / "data"
    adapter_path = base_dir / "adapter"
    
    # Config
    config = {
        "iters": args.iters,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "lora_layers": args.lora_layers,
    }
    
    if not args.test_only:
        # 1. Prepare data
        prepare_data(args.data, data_dir, args.samples)
        
        # 2. Convert model to MLX
        convert_model(args.model, model_dir)
        
        # 3. Train LoRA
        success = train_lora(model_dir, data_dir, adapter_path, config)
        if not success:
            return 1
    
    # 4. Test
    test_prompts = [
        "### User:\nWrite a Python function to flatten a nested list\n### Assistant:\n",
        "### User:\nCreate a TypeScript interface for a paginated API response\n### Assistant:\n",
        "### User:\nExplain the difference between map and flatMap in RxJS\n### Assistant:\n",
    ]
    test_model(model_dir, adapter_path, test_prompts)
    
    print("\n" + "=" * 60)
    print("✅ Done! Your fine-tuned model is ready.")
    print(f"   Base model: {model_dir}")
    print(f"   Adapter: {adapter_path}")
    print("\nTo use later:")
    print(f"  python -m mlx_lm.generate --model {model_dir} --adapter-path {adapter_path} --prompt 'Your prompt'")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())