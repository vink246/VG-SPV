"""
LoRA supervised finetuning: teach a VLM to emit `<risk_factors_with_boxes>` + grid-normalized coords.

Run from repo root:
  python train/run_bounding_box_sft.py --model_name llava-hf/llava-v1.6-mistral-7b-hf --output_dir outputs/bbox_sft_llava

Saved layout (compatible with `inference/run_inference.py --lora-adapter` and `train/run_dpo.py --lora_adapter_path`):
  ``{output_dir}/adapter`` — final PEFT weights + tokenizer/processor
  ``{output_dir}/adapter_latest`` — overwritten on each periodic save (crash recovery)

Requires: torch, transformers, peft, datasets, accelerate, pillow.

By default, uses HF ``train`` / ``val`` (or holdout from train) for REC loss plus
``data/.../train_method2.csv`` mixed into training and ``test_method2.csv`` for eval loss
(``eval_loss`` in logs). Disable with ``--skip_eval`` / ``--no_vgspv_csv``.
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
from transformers import Trainer, TrainerCallback

from train.bounding_box_sft_collator import BoundingBoxSFTCollator
from train.bounding_box_sft_dataset import load_bbox_sft_hf_datasets, load_vgspv_csv_rows_for_sft
from train.bounding_box_sft_schema import BOX_COORD_SCALE
from train.bounding_box_sft_torch_dataset import BoundingBoxSFTConcatEvalDataset, BoundingBoxSFTMixedIndexDataset
from train.dataset_adapter import DEFAULT_PROMPT_INSTRUCTION
from train.hf_training_args_compat import apply_eval_scheduling_kwargs, instantiate_training_arguments
from train.lora_factory import attach_lora, default_lora_config
from vlm import load_vlm

_DEFAULT_MM_TRAIN_CSV = _REPO_ROOT / "data/mm-safebench_1/extracted_data/traces/train_method2.csv"
_DEFAULT_MM_EVAL_CSV = _REPO_ROOT / "data/mm-safebench_1/extracted_data/traces/test_method2.csv"
_EVAL_SPLIT_FALLBACKS = ("val", "validation", "test")


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
        help="Primary HF hub id, or path to save_to_disk REC data (first shard of combined training).",
    )
    p.add_argument(
        "--extra_dataset",
        action="append",
        default=[],
        metavar="ID_OR_PATH",
        help="Additional hub id or save_to_disk path (repeatable). Concatenated after --dataset_id when schemas match. "
        "PaDT: use only PaDT-MLLM/RefCOCO (it already includes refcoco+ / refcocog data); separate PaDT RefCOCO+ hubs are often absent.",
    )
    p.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Optional HF builder config (``name=``). PaDT RefCOCO hubs on recent ``datasets`` expose "
        "``default`` only — omit this flag or pass ``default``. Wrong names raise ValueError (available configs).",
    )
    p.add_argument("--split", type=str, default="train", help="Dataset split (PaDT RefCOCO uses `train`).")
    p.add_argument(
        "--hf_local_files_only",
        action="store_true",
        help="Hub loads only: pass DownloadConfig(local_files_only=True) (no network). Set HF_HOME on PACE.",
    )
    p.add_argument("--max_samples", type=int, default=None, help="Cap HF dataset size for debugging.")
    p.add_argument(
        "--vgspv_csv",
        type=str,
        default=None,
        help="Train VG-SPV CSV (image, chosen_reasoning_trace, …). Default: repo train_method2.csv if present.",
    )
    p.add_argument(
        "--vgspv_mix_fraction",
        type=float,
        default=0.25,
        help="Fraction of train steps from VG-SPV CSV vs HF (0–1). Ignored when no train CSV is loaded.",
    )
    p.add_argument(
        "--vgspv_prompt_instruction",
        type=str,
        default=None,
        help="Fallback user text when CSV has no `prompt` column (default: DEFAULT_PROMPT_INSTRUCTION).",
    )
    p.add_argument(
        "--vgspv_image_root",
        type=str,
        default=None,
        help="Optional base dir for resolving relative `image` paths in --vgspv_csv (default: repo root + cwd).",
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
    p.add_argument(
        "--eval_split",
        type=str,
        default="val",
        help="HF split for eval loss (tried first, then validation, test). If none load, see --hf_eval_holdout_fraction.",
    )
    p.add_argument(
        "--hf_eval_holdout_fraction",
        type=float,
        default=0.02,
        help="If no HF eval split loads, hold out this fraction of train HF rows for eval (disjoint).",
    )
    p.add_argument(
        "--eval_max_samples",
        type=int,
        default=None,
        help="Cap HF eval rows after loading (None = all).",
    )
    p.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Run eval every N steps (default: max(logging_steps, 50)). Ignored with --skip_eval.",
    )
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--skip_eval", action="store_true", help="Disable eval loss (train loss only).")
    p.add_argument(
        "--vgspv_eval_csv",
        type=str,
        default=None,
        help="CSV for eval (default: test_method2.csv under repo if present).",
    )
    p.add_argument(
        "--no_vgspv_csv",
        action="store_true",
        help="Do not mix train_method2.csv even if the default file exists.",
    )
    return p.parse_args()


def _try_load_hf_eval(
    sources: list[str],
    preferred_split: str,
    *,
    config_name: str | None,
    local_files_only: bool,
    max_eval_samples: int | None,
):
    """Return (dataset, split_name_used) or (None, None)."""
    order: list[str] = []
    if preferred_split and preferred_split.strip():
        order.append(preferred_split.strip())
    for s in _EVAL_SPLIT_FALLBACKS:
        if s not in order:
            order.append(s)
    for sp in order:
        try:
            ds = load_bbox_sft_hf_datasets(
                sources,
                sp,
                max_samples=max_eval_samples,
                config_name=config_name,
                local_files_only=local_files_only,
            )
            if len(ds) > 0:
                return ds, sp
        except Exception:
            continue
    return None, None


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

    sources = [args.dataset_id] + list(args.extra_dataset or [])
    hf_train_full = load_bbox_sft_hf_datasets(
        sources,
        args.split,
        max_samples=args.max_samples,
        config_name=args.dataset_config,
        local_files_only=bool(args.hf_local_files_only),
    )

    hf_eval_ds = None
    eval_split_used: str | None = None
    hf_train = hf_train_full
    if not args.skip_eval:
        hf_eval_ds, eval_split_used = _try_load_hf_eval(
            sources,
            args.eval_split,
            config_name=args.dataset_config,
            local_files_only=bool(args.hf_local_files_only),
            max_eval_samples=args.eval_max_samples,
        )
        if hf_eval_ds is None or len(hf_eval_ds) == 0:
            hold = float(args.hf_eval_holdout_fraction)
            if hold <= 0.0 or hold >= 1.0:
                raise SystemExit(
                    "No HF eval split could be loaded; set --hf_eval_holdout_fraction in (0, 1) "
                    "to hold out part of train, or use --skip_eval."
                )
            split_d = hf_train_full.train_test_split(test_size=hold, seed=int(args.mix_seed))
            hf_train = split_d["train"]
            hf_eval_ds = split_d["test"]
            eval_split_used = f"train_holdout_{hold:g}"
            print(f"HF eval: holdout {hold * 100:.3g}% of train rows (no hub val/validation/test).")
        else:
            print(f"HF eval: hub split {eval_split_used!r} ({len(hf_eval_ds)} rows).")
        if args.eval_max_samples is not None and len(hf_eval_ds) > int(args.eval_max_samples):
            hf_eval_ds = hf_eval_ds.select(range(int(args.eval_max_samples)))
    else:
        hf_eval_ds = None

    vgspv_prompt = args.vgspv_prompt_instruction or DEFAULT_PROMPT_INSTRUCTION
    img_root = Path(args.vgspv_image_root).resolve() if args.vgspv_image_root else None

    train_csv_path: Path | None = None
    if args.no_vgspv_csv:
        train_csv_path = None
    elif args.vgspv_csv:
        train_csv_path = Path(args.vgspv_csv).expanduser().resolve()
        if not train_csv_path.is_file():
            raise SystemExit(f"--vgspv_csv not found: {train_csv_path}")
    elif _DEFAULT_MM_TRAIN_CSV.is_file():
        train_csv_path = _DEFAULT_MM_TRAIN_CSV.resolve()
    else:
        train_csv_path = None

    csv_rows: list | None = None
    mix_fraction = 0.0
    if train_csv_path is not None:
        csv_rows = load_vgspv_csv_rows_for_sft(str(train_csv_path), image_root=img_root)
        mix_fraction = float(args.vgspv_mix_fraction)
        if mix_fraction <= 0.0:
            raise SystemExit("Train VG-SPV CSV is loaded but --vgspv_mix_fraction must be > 0.")

    eval_csv_path: Path | None = None
    if args.skip_eval:
        eval_csv_path = None
    elif args.vgspv_eval_csv:
        eval_csv_path = Path(args.vgspv_eval_csv).expanduser().resolve()
        if not eval_csv_path.is_file():
            raise SystemExit(f"--vgspv_eval_csv not found: {eval_csv_path}")
    elif _DEFAULT_MM_EVAL_CSV.is_file():
        eval_csv_path = _DEFAULT_MM_EVAL_CSV.resolve()
    else:
        eval_csv_path = None

    csv_eval_rows: list | None = None
    if eval_csv_path is not None:
        csv_eval_rows = load_vgspv_csv_rows_for_sft(str(eval_csv_path), image_root=img_root)

    eval_ds: BoundingBoxSFTConcatEvalDataset | None = None
    if not args.skip_eval:
        eval_ds = BoundingBoxSFTConcatEvalDataset(hf_eval_ds, csv_eval_rows)
        if len(eval_ds) == 0:
            eval_ds = None
            print("Eval disabled: empty HF eval and empty eval CSV.")

    collator = BoundingBoxSFTCollator(
        loaded.processor,
        hf_train,
        loaded.family,
        hf_eval=None if args.skip_eval else hf_eval_ds,
        csv_rows=csv_rows,
        csv_eval_rows=None if args.skip_eval else csv_eval_rows,
        vgspv_prompt_instruction=vgspv_prompt,
    )
    torch_ds = BoundingBoxSFTMixedIndexDataset(
        hf_train,
        csv_rows,
        mix_fraction=mix_fraction if csv_rows else 0.0,
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
    )
    # ``save_on_train_end`` exists only on newer ``transformers``; final PEFT is written after
    # ``trainer.train()`` anyway (``adapter/``, ``adapter_latest/`` + callbacks on_train_end).
    if save_strategy == "steps":
        targs_kw["save_steps"] = save_steps_val
    eval_dataset_arg = eval_ds if eval_ds is not None and len(eval_ds) > 0 else None
    if eval_dataset_arg is not None:
        eval_steps = int(args.eval_steps) if args.eval_steps is not None else max(int(args.logging_steps), 50)
        if not apply_eval_scheduling_kwargs(
            targs_kw,
            eval_steps=eval_steps,
            per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        ):
            eval_dataset_arg = None
    targs = instantiate_training_arguments(**targs_kw)

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
        eval_dataset=eval_dataset_arg,
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
        "dataset_sources": sources,
        "dataset_id": args.dataset_id,
        "extra_dataset": list(args.extra_dataset or []),
        "dataset_config": args.dataset_config,
        "hf_local_files_only": bool(args.hf_local_files_only),
        "split": args.split,
        "max_samples": args.max_samples,
        "eval_split_requested": args.eval_split,
        "eval_split_used": eval_split_used,
        "hf_eval_holdout_fraction": float(args.hf_eval_holdout_fraction),
        "eval_max_samples": args.eval_max_samples,
        "eval_steps": (int(args.eval_steps) if args.eval_steps is not None else max(int(args.logging_steps), 50))
        if eval_ds and len(eval_ds) > 0
        else None,
        "per_device_eval_batch_size": int(args.per_device_eval_batch_size)
        if eval_ds and len(eval_ds) > 0
        else None,
        "skip_eval": bool(args.skip_eval),
        "eval_dataset_len": len(eval_ds) if eval_ds is not None else 0,
        "vgspv_train_csv": str(train_csv_path) if train_csv_path else None,
        "vgspv_csv": args.vgspv_csv,
        "vgspv_eval_csv": str(eval_csv_path) if eval_csv_path else None,
        "vgspv_mix_fraction": mix_fraction if csv_rows else 0.0,
        "vgspv_prompt_instruction": vgspv_prompt if csv_rows else None,
        "vgspv_image_root": str(img_root) if img_root else None,
        "mix_seed": args.mix_seed,
        "no_vgspv_csv": bool(args.no_vgspv_csv),
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
