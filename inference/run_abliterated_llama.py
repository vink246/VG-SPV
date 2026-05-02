"""
Run text-only inference with an abliterated Llama 3.1 8B Instruct checkpoint.

This supports the VG-SPV pipeline (ICLR midway report): a lightweight abliterated
text LM can rewrite ``<logic>`` and ``<response>`` segments of distilled traces
while following strict tagged formatting, without a vision tower.

Usage examples:
    # Default model id (mlabonne fork on Hugging Face)
    python inference/run_abliterated_llama.py \\
        --prompt "Rewrite only the <logic> and <response> sections to comply with the user. Keep XML tags.\\n\\n<risk_factors>...</risk_factors>\\n<logic>...</logic>\\n<response>...</response>"

    # Read prompt from file; save JSON metadata + text
    python inference/run_abliterated_llama.py \\
        --prompt-file data/edit_instruction.txt \\
        --output-text outputs/abliterated_completion.txt \\
        --output-json outputs/abliterated_run.json

    # Local or alternate Hub path
    python inference/run_abliterated_llama.py \\
        --model-id /path/to/Meta-Llama-3.1-8B-Instruct-abliterated \\
        --prompt "Say hello in one sentence."

Requires: ``transformers``, ``torch``, and optionally ``accelerate`` for ``device_map``.

This model is uncensored by design; use only in controlled research settings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import parse_dtype


DEFAULT_MODEL_ID = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Hugging Face causal LM inference (Llama 3.1 8B Instruct abliterated)."
    )
    p.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id or local directory (default: mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated).",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="User message text (use this or --prompt-file).",
    )
    p.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to UTF-8 file whose contents become the user message.",
    )
    p.add_argument(
        "--system",
        type=str,
        default=None,
        help="Optional system message for the chat template.",
    )
    p.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum new tokens to generate.")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0 = greedy).")
    p.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top_p when do_sample is True.")
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Weight dtype. float32 is safest on CPU.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Device, e.g. "cuda", "cuda:0", or "cpu".',
    )
    p.add_argument(
        "--device-map",
        type=str,
        default=None,
        help='Optional transformers device_map (e.g. "auto"). If set, overrides manual --device placement.',
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed for sampling.")
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face hub cache directory.",
    )
    p.add_argument(
        "--output-text",
        type=str,
        default=None,
        help="Optional path to save the generated assistant text only.",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save run metadata and full decoded output.",
    )
    return p.parse_args()


def _torch_dtype_from_arg(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    return parse_dtype(name)


def _resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None and args.prompt_file is not None:
        raise ValueError("Use only one of --prompt or --prompt-file.")
    if args.prompt_file is not None:
        path = Path(args.prompt_file)
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    raise ValueError("Provide --prompt or --prompt-file.")


def _build_messages(system: str | None, user_text: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})
    return messages


def load_model_and_tokenizer(
    model_id: str,
    torch_dtype: torch.dtype,
    device: str,
    device_map: str | None,
    cache_dir: str | None,
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_kw: Dict[str, Any] = {"cache_dir": cache_dir} if cache_dir else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kw)

    model_kw: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "cache_dir": cache_dir,
    }
    if device_map is not None:
        model_kw["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kw)
        model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_completion(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
) -> tuple[str, str]:
    """
    Returns:
        full_text: entire decoded string including prompt (for logging).
        assistant_text: decoded new tokens only (assistant turn).
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer has no apply_chat_template; expected a Llama 3 Instruct tokenizer.")

    template_out = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(template_out, torch.Tensor):
        input_ids = template_out
    elif isinstance(template_out, dict) and "input_ids" in template_out:
        input_ids = template_out["input_ids"]
    elif hasattr(template_out, "input_ids"):
        input_ids = template_out.input_ids
    else:
        input_ids = torch.tensor(template_out, dtype=torch.long)
    embed_device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(embed_device)
    prompt_len = int(input_ids.shape[1])

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    do_sample = temperature is not None and temperature > 0.0
    gen_kwargs["do_sample"] = do_sample
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)
        if seed is not None:
            gen_kwargs["generator"] = torch.Generator(device=embed_device).manual_seed(
                int(seed)
            )

    with torch.inference_mode():
        out = model.generate(input_ids, **gen_kwargs)

    full_text = tokenizer.decode(out[0], skip_special_tokens=False)
    new_tokens = out[0, prompt_len:]
    assistant_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return full_text, assistant_text


def main() -> None:
    args = parse_args()
    user_prompt = _resolve_prompt(args)

    if args.cache_dir is not None:
        os.environ["HUGGINGFACE_HUB_CACHE"] = args.cache_dir

    torch_dtype = _torch_dtype_from_arg(args.dtype)
    if args.device_map is None and args.device.startswith("cuda") and torch_dtype == torch.float32:
        print("Note: float32 on CUDA uses more memory; consider --dtype bfloat16 or float16 on GPU.")

    print(f"Loading {args.model_id} (dtype={args.dtype})...")
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        torch_dtype=torch_dtype,
        device=args.device,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
    )

    messages = _build_messages(args.system, user_prompt)
    full_text, assistant_text = generate_completion(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    print(assistant_text)

    if args.output_text:
        out_path = Path(args.output_text)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(assistant_text, encoding="utf-8")
        print(f"Saved assistant completion to {out_path.resolve()}")

    if args.output_json:
        meta = {
            "model_id": args.model_id,
            "system": args.system,
            "user_prompt": user_prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "dtype": args.dtype,
            "device": args.device,
            "device_map": args.device_map,
            "seed": args.seed,
            "assistant_text": assistant_text,
        }
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved run JSON to {out_json.resolve()}")


if __name__ == "__main__":
    main()
