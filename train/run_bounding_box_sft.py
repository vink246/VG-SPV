"""
LoRA supervised finetuning: teach a VLM to emit `<risk_factors_with_boxes>` + grid-normalized coords.

Run from repo root:
  python train/run_bounding_box_sft.py --model_name llava-hf/llava-v1.6-mistral-7b-hf --output_dir outputs/bbox_sft_llava

Saved layout (compatible with `inference/run_inference.py --lora-adapter` and `train/run_dpo.py --lora_adapter_path`):
  ``{output_dir}/adapter`` — final PEFT weights + tokenizer/processor
  ``{output_dir}/adapter_latest`` — overwritten on each periodic save (crash recovery)

Requires: torch, transformers, peft, datasets, accelerate, pillow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from peft import PeftModel
from transformers import Trainer, TrainerCallback, TrainingArguments

from train.bounding_box_sft_collator import BoundingBoxSFTCollator
from train.bounding_box_sft_dataset import load_bbox_sft_hf_dataset, load_vgspv_csv_rows_for_sft
from train.bounding_box_sft_schema import BOX_COORD_SCALE
from train.bounding_box_sft_torch_dataset import BoundingBoxSFTMixedIndexDataset
from train.dataset_adapter import DEFAULT_PROMPT_INSTRUCTION
from train.lora_factory import attach_lora, default_lora_config
from vlm import load_vlm


def _save_peft_bundle(model: torch.nn.Module, tokenizer, processor, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dest))
    tokenizer.save_pretrained(str(dest))
    if processor is not None and processor is not tokenizer:
        processor.save_pretrained(str(dest))


class SaveAdapterLatestCallback(TrainerCallback):
    """Mirror PEFT + tokenizer/processor to a fixed directory on each HF checkpoint save."""

    def __init__(self, adapter_latest_dir: Path, tokenizer, processor):
        self.adapter_latest_dir = adapter_latest_dir
        self.tokenizer = tokenizer
        self.processor = processor

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        _save_peft_bundle(model, self.tokenizer, self.processor, self.adapter_latest_dir)
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        _save_peft_bundle(model, self.tokenizer, self.processor, self.adapter_latest_dir)
        return control


class SaveEveryNEpochsCallback(TrainerCallback):
    """Call ``trainer.save_model()`` every N completed epochs and mirror PEFT to adapter_latest."""

    def __init__(self, every_n_epochs: int, adapter_latest_dir: Path, tokenizer, processor):
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.adapter_latest_dir = adapter_latest_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self._trainer = None

    def set_trainer(self, trainer) -> None:
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._trainer is None:
            return control
        ep = int(round(float(state.epoch))) if state.epoch is not None else 0
        if ep <= 0:
            return control
        if ep % self.every_n_epochs != 0:
            return control
        self._trainer.save_model()
        model = self._trainer.model
        _save_peft_bundle(model, self.tokenizer, self.processor, self.adapter_latest_dir)
        return control


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bounding-box SFT (LoRA) for VG-SPV.")
    p.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HF model id or local path (base weights), e.g. llava-hf/llava-v1.6-mistral-7b-hf.",
    )
    p.add_argument(
        "--resume_adapter_path",
        type=str,
        default=None,
        help="Optional PEFT adapter directory to continue training (same base as --model_name).",
    )
    p.add_argument(
        "--dataset_id",
        type=str,
        default="PaDT-MLLM/RefCOCO",
        help="HF dataset (default PaDT-MLLM/RefCOCO — train split; Shikra/VoCoT-style REC).",
    )
    p.add_argument("--dataset_config", type=str, default=None, help="Optional HF config name for the dataset.")
    p.add_argument("--split", type=str, default="train", help="Dataset split (PaDT RefCOCO uses `train`).")
    p.add_argument("--max_samples", type=int, default=None, help="Cap HF dataset size for debugging.")
    p.add_argument(
        "--vgspv_csv",
        type=str,
        default=None,
        help="Optional VG-fDPO-style CSV (image, chosen_reasoning_trace, …) to mix into SFT.",
    )
    p.add_argument(
        "--vgspv_mix_fraction",
        type=float,
        default=0.0,
        help="Fraction of steps that sample from --vgspv_csv (0–1). Ignored if no CSV.",
    )
    p.add_argument(
        "--vgspv_prompt_instruction",
        type=str,
        default=None,
        help="User text for CSV rows (default: train/dataset_adapter.DEFAULT_PROMPT_INSTRUCTION).",
    )
    p.add_argument(
        "--mix_seed",
        type=int,
        default=42,
        help="Seed for deterministic HF vs CSV selection per index.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/bounding_box_sft",
        help="Where to save adapter + metadata.",
    )
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 training.")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument(
        "--save_every_steps",
        type=int,
        default=500,
        help="Save a HF checkpoint (and mirror adapter_latest) every N global steps; 0 disables step-based saves.",
    )
    p.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=0,
        help="Additionally save a checkpoint every N epochs (0 = off). Works alongside save_every_steps.",
    )
    p.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Max full checkpoints to keep under output_dir (oldest removed). adapter_latest is always overwritten.",
    )
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument(
        "--model_family",
        type=str,
        default=None,
        choices=["qwen3_vl", "llava", "tinyllava", "mllama"],
        help="Optional backend override (see vlm/registry.py).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapter_dir = out / "adapter"
    adapter_latest = out / "adapter_latest"

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    loaded = load_vlm(args.model_name, dtype=dtype, model_family=args.model_family)
    if loaded.family in ("tinyllava",):
        raise SystemExit(
            "Bounding-box SFT trainer currently supports HF processor models (llava, mllama, qwen3_vl). "
            f"Got family={loaded.family}. Extend train/bounding_box_sft_collator.py for TinyLLaVA if needed."
        )
    if loaded.processor is None:
        raise SystemExit("Bounding-box SFT requires a processor (LLaVA / Mllama / Qwen-VL).")

    hf_ds = load_bbox_sft_hf_dataset(
        args.dataset_id,
        args.split,
        max_samples=args.max_samples,
        config_name=args.dataset_config,
    )

    csv_rows: list | None = None
    vgspv_prompt = args.vgspv_prompt_instruction or DEFAULT_PROMPT_INSTRUCTION
    if args.vgspv_csv:
        csv_rows = load_vgspv_csv_rows_for_sft(args.vgspv_csv)
        mf = float(args.vgspv_mix_fraction)
        if mf <= 0:
            raise SystemExit("--vgspv_csv requires --vgspv_mix_fraction > 0.")
    elif args.vgspv_mix_fraction > 0:
        raise SystemExit("--vgspv_mix_fraction > 0 requires --vgspv_csv.")

    collator = BoundingBoxSFTCollator(
        loaded.processor,
        hf_ds,
        loaded.family,
        csv_rows=csv_rows,
        vgspv_prompt_instruction=vgspv_prompt,
    )
    torch_ds = BoundingBoxSFTMixedIndexDataset(
        hf_ds,
        csv_rows,
        mix_fraction=args.vgspv_mix_fraction if csv_rows else 0.0,
        seed=args.mix_seed,
    )

    if args.resume_adapter_path:
        model = PeftModel.from_pretrained(
            loaded.model,
            args.resume_adapter_path,
            is_trainable=True,
        )
    else:
        lcfg = default_lora_config(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        model = attach_lora(loaded.model, lora_config=lcfg, freeze_vision=True, prepare_kbit=False)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    save_strategy = "no"
    save_steps_val = 500
    if args.save_every_steps and args.save_every_steps > 0:
        save_strategy = "steps"
        save_steps_val = int(args.save_every_steps)

    targs_kw: dict = dict(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_total_limit=max(1, int(args.save_total_limit)),
        bf16=args.bf16,
        fp16=not args.bf16 and dtype == torch.float16,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        save_on_train_end=True,
    )
    if save_strategy == "steps":
        targs_kw["save_steps"] = save_steps_val
    targs = TrainingArguments(**targs_kw)

    callbacks: list[TrainerCallback] = [
        SaveAdapterLatestCallback(adapter_latest, loaded.tokenizer, loaded.processor),
    ]
    epoch_cb: SaveEveryNEpochsCallback | None = None
    if args.save_every_n_epochs and int(args.save_every_n_epochs) > 0:
        epoch_cb = SaveEveryNEpochsCallback(
            int(args.save_every_n_epochs),
            adapter_latest,
            loaded.tokenizer,
            loaded.processor,
        )
        callbacks.append(epoch_cb)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=torch_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    if epoch_cb is not None:
        epoch_cb.set_trainer(trainer)

    trainer.train()

    _save_peft_bundle(model, loaded.tokenizer, loaded.processor, adapter_dir)
    _save_peft_bundle(model, loaded.tokenizer, loaded.processor, adapter_latest)

    meta = {
        "base_model_name": args.model_name,
        "vlm_family": loaded.family,
        "box_format": f"int_grid_{BOX_COORD_SCALE}",
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "max_samples": args.max_samples,
        "vgspv_csv": args.vgspv_csv,
        "vgspv_mix_fraction": args.vgspv_mix_fraction if csv_rows else 0.0,
        "vgspv_prompt_instruction": vgspv_prompt if csv_rows else None,
        "mix_seed": args.mix_seed,
        "resume_adapter_path": args.resume_adapter_path,
        "save_every_steps": args.save_every_steps,
        "save_every_n_epochs": args.save_every_n_epochs,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "adapter_path": str(adapter_dir.resolve()),
        "adapter_latest_path": str(adapter_latest.resolve()),
    }
    meta_path = out / "bounding_box_sft_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter to {adapter_dir}")
    print(f"Latest mirror: {adapter_latest}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
