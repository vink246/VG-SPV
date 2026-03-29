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
from utils import load_vl_model_and_processor


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
    p.add_argument("--ref_8bit", action="store_true", help="Load reference model in 8-bit for memory savings")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 (recommended for Ampere+)")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--prompt_instruction", type=str, default=None, help="Instruction used as prompt when loading from CSV (default: see train/dataset_adapter.py)")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model, processor = load_vl_model_and_processor(args.model_name, dtype=dtype)
    tokenizer = getattr(processor, "tokenizer", processor)

    ref_model = None
    if args.ref_8bit:
        ref_model, _ = load_vl_model_and_processor(
            args.model_name,
            dtype=dtype,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
    # If ref_model is None, DPOTrainer creates a copy of the model as ref

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
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
