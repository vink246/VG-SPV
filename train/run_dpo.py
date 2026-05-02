"""
Launch DPO training with any supported VL model (TinyLLaVA, LLaVA, etc.) using VGSPVTrainer.

Uses TRL DPOTrainer pipeline; loss is overridable in train/dpo_trainer.py (VG-fDPO).
"""

import argparse
import sys
from pathlib import Path

# Allow running as python train/run_dpo.py from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from transformers import BitsAndBytesConfig
from trl import DPOConfig

from train.dataset_adapter import DEFAULT_PROMPT_INSTRUCTION, load_dpo_dataset
from train.dpo_trainer import VGSPVTrainer
from vlm import load_vlm, load_vlm_with_optional_lora


def parse_args():
    p = argparse.ArgumentParser(description="Run DPO training with TinyLLaVA (VG-SPV).")
    p.add_argument(
        "--model_name",
        type=str,
        default="tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B",
        help="Model name or path (e.g. tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B, llava-hf/llava-1.5-7b-hf)",
    )
    p.add_argument("--data_path", type=str, default=None, help="Path to DPO dataset: CSV (image, perturbed_image, chosen_reasoning_trace, rejected_reasoning_trace) or dataset dir/HF name. Default: minimal example.")
    p.add_argument("--output_dir", type=str, default="outputs/dpo", help="Training output directory")
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-7)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    p.add_argument("--alpha_vdpo", type=float, default=0.1, help="Weight for V-DPO term in total loss")
    p.add_argument("--vdpo_margin_m", type=float, default=0.0, help="Margin m for V-DPO hinge")
    p.add_argument("--alpha_format", type=float, default=12.0, help="Scaling for format-fail fallback loss")
    p.add_argument(
        "--grounding_mode",
        type=str,
        default="semantic",
        choices=["semantic", "spatial"],
        help="How to compute s: semantic cosine or spatial IoU.",
    )
    p.add_argument("--alpha_sem", type=float, default=1.0, help="Coefficient for semantic scaler s_sem")
    p.add_argument("--alpha_iou", type=float, default=1.0, help="Coefficient for spatial scaler s_sp")
    p.add_argument(
        "--s_fallback_value",
        type=float,
        default=1.0,
        help="Fallback s when scaler inputs are missing/invalid (unless strict mode is enabled).",
    )
    p.add_argument(
        "--strict_scaler_inputs",
        action="store_true",
        help="Raise if scaler inputs are missing/invalid instead of using s_fallback_value.",
    )
    p.add_argument("--ref_8bit", action="store_true", help="Load reference model in 8-bit for memory savings")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 (recommended for Ampere+)")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--prompt_instruction", type=str, default=None, help="Instruction used as prompt when loading from CSV (default: see train/dataset_adapter.py)")
    p.add_argument(
        "--lora_adapter_path",
        type=str,
        default=None,
        help="Optional bbox-SFT PEFT dir (e.g. .../adapter or .../adapter_latest). Policy loads base + adapter; ref is frozen base without adapter.",
    )
    p.add_argument(
        "--merge_lora_adapter",
        action="store_true",
        help="Merge bounding-box SFT LoRA into base weights before DPO (policy is dense weights).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.merge_lora_adapter and not args.lora_adapter_path:
        raise SystemExit(
            "--merge_lora_adapter requires --lora_adapter_path (PEFT directory, e.g. .../adapter or .../adapter_latest)."
        )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    loaded = load_vlm_with_optional_lora(
        args.model_name,
        dtype=dtype,
        lora_adapter_path=args.lora_adapter_path,
        merge_adapter=args.merge_lora_adapter,
        is_trainable=True,
    )
    model = loaded.model
    tokenizer = loaded.tokenizer

    ref_model = None
    if args.lora_adapter_path and not args.merge_lora_adapter:
        ref_kw: dict = {"dtype": dtype}
        if args.ref_8bit:
            ref_kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        ref_loaded = load_vlm(args.model_name, **ref_kw)
        ref_model = ref_loaded.model
        for p in ref_model.parameters():
            p.requires_grad = False
    elif args.ref_8bit:
        ref_loaded = load_vlm(
            args.model_name,
            dtype=dtype,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        ref_model = ref_loaded.model
    # If ref_model is None, DPOTrainer creates a copy of the policy as ref (no separate frozen base).

    train_dataset = load_dpo_dataset(
        args.data_path,
        prompt_instruction=args.prompt_instruction or DEFAULT_PROMPT_INSTRUCTION,
    )

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    )

    trainer = VGSPVTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        alpha_vdpo=args.alpha_vdpo,
        vdpo_margin_m=args.vdpo_margin_m,
        alpha_format=args.alpha_format,
        grounding_mode=args.grounding_mode,
        alpha_sem=args.alpha_sem,
        alpha_iou=args.alpha_iou,
        s_fallback_value=args.s_fallback_value,
        strict_scaler_inputs=args.strict_scaler_inputs,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
