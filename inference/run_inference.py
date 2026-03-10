"""
Run vision-language inference with any supported model (Qwen3-VL, LLaVA, etc.).

Usage:
    python inference/run_inference.py --model Qwen/Qwen3-VL-2B-Instruct --image path/to/img.png --prompt "Describe this image"
    python inference/run_inference.py --model Qwen/Qwen3-VL-4B-Instruct --image img.png --prompt "What is in this image?" --output response.txt
    python inference/run_inference.py --model llava-hf/llava-1.5-7b-hf --image img.png --prompt "Describe this image"

Make sure toset the Hugging Face cache to a directory with enough space (e.g. scratch): export HF_HOME=~/scratch/.cache/huggingface
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running as python inference/run_inference.py from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Use scratch (or HF_HOME) for model cache if home has limited quota
if "HUGGINGFACE_HUB_CACHE" not in os.environ and "HF_HOME" not in os.environ:
    scratch = os.environ.get("SCRATCH") or os.environ.get("TMPDIR") or os.environ.get("TEMP")
    if scratch:
        hf_cache = Path(scratch) / ".cache" / "huggingface" / "hub"
        hf_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)

from inference.utils import run_vl_inference
from utils import build_messages, load_vl_model_and_processor, parse_dtype


def parse_args():
    p = argparse.ArgumentParser(
        description="Run VL inference on image + text. Use --model to choose the model."
    )
    p.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="Model name or path (e.g. Qwen/Qwen3-VL-2B-Instruct, Qwen/Qwen3-VL-4B-Instruct, llava-hf/llava-1.5-7b-hf)")
    p.add_argument("--image", type=str, required=True, help="Path to input image (or comma-separated paths)")
    p.add_argument("--prompt", type=str, default="Describe this image.", help="Text prompt for the model")
    p.add_argument("--output", type=str, default=None, help="Optional path to save response text")
    p.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens to generate")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"], help="Model dtype")
    p.add_argument("--cache_dir", type=str, default=None, help="Override Hugging Face cache dir (default: use HF_HOME / SCRATCH)")
    return p.parse_args()


def main():
    args = parse_args()
    image_paths = [p.strip() for p in args.image.split(",")]
    for p in image_paths:
        if not p.startswith(("http://", "https://")) and not Path(p).exists():
            raise FileNotFoundError(f"Image path not found: {p}")

    if args.cache_dir is not None:
        os.environ["HUGGINGFACE_HUB_CACHE"] = args.cache_dir

    dtype = parse_dtype(args.dtype)
    print(f"Loading model {args.model}...")
    model, processor = load_vl_model_and_processor(args.model, dtype=dtype)

    messages = build_messages(image_paths, args.prompt)
    response = run_vl_inference(
        model, processor, messages,
        max_new_tokens=args.max_new_tokens,
        model_name=args.model,
    )

    print(response)
    if args.output:
        Path(args.output).write_text(response, encoding="utf-8")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
