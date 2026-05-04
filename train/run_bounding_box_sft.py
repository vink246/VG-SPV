"""
LoRA supervised finetuning: teach a VLM to emit `<risk_factors_with_boxes>` + grid-normalized coords.

Training and validation use **MM-SafetyBench / VG-SPV CSV traces** only (Method-2 style: ``image``,
``prompt``, ``chosen_reasoning_trace``).

Run from repo root:
  python train/run_bounding_box_sft.py --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \\
    --output_dir outputs/bbox_sft_llama

Saved layout (compatible with ``inference/run_inference.py --lora-adapter`` and ``train/run_dpo.py``):
  ``{output_dir}/adapter`` — final PEFT weights + tokenizer/processor
  ``{output_dir}/adapter_latest`` — overwritten on each periodic save (crash recovery)

With ``--save_every_steps 0`` (default): **HF checkpoint + eval_loss at the end of every epoch**.
Use ``--save_every_steps N`` for step-based saves and aligned eval (see ``--eval_steps``).

Checkpoints default to **not** saving the optimizer (``save_only_model``). Use ``--save_optimizer_state``
for full HF resume state. Failed checkpoint writes still mirror PEFT to ``adapter_latest/`` when possible
(``BoundingBoxSFTTrainer``).

Requires: torch, transformers, peft, datasets, accelerate, pillow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from peft import PeftModel
from transformers import EarlyStoppingCallback, TrainerCallback

from train.bounding_box_sft_collator import BoundingBoxSFTCollator
from train.bounding_box_sft_dataset import load_vgspv_csv_rows_for_sft
from train.bounding_box_sft_schema import BOX_COORD_SCALE
from train.bounding_box_sft_torch_dataset import BoundingBoxSFTCsvEvalDataset, BoundingBoxSFTCsvTrainDataset
from train.bounding_box_sft_trainer import BoundingBoxSFTTrainer
from train.dataset_adapter import DEFAULT_PROMPT_INSTRUCTION
from train.hf_training_args_compat import (
    apply_epoch_eval_scheduling_kwargs,
    apply_eval_scheduling_kwargs,
    instantiate_training_arguments,
)
from train.lora_factory import attach_lora, default_lora_config, normalize_no_split_modules_for_accelerate
from vlm import load_vlm

_DEFAULT_MM_TRAIN_CSV = _REPO_ROOT / "data/mm-safebench_1/extracted_data/traces/train_method2.csv"
_DEFAULT_MM_EVAL_CSV = _REPO_ROOT / "data/mm-safebench_1/extracted_data/traces/test_method2.csv"


def _print_training_device_banner(loaded) -> None:
    """Log whether CUDA (or MPS) is available and where the model sits."""
    m = loaded.model
    try:
        dev = next(m.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    line = "=" * 64
    print(line)
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"GPU: CUDA is available ({n} device(s))")
        for i in range(n):
            print(f"  cuda:{i} — {torch.cuda.get_device_name(i)}")
    else:
        print("GPU: CUDA is not available on this machine.")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("GPU: Apple MPS backend is available (model may still be on CPU/CUDA depending on load).")
    print(f"Model parameter device: {dev}")
    print(line)


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
    p = argparse.ArgumentParser(description="Bounding-box SFT (LoRA) on MM-SafetyBench / VG-SPV CSV traces.")
    p.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HF model id or local path (base weights), e.g. meta-llama/Llama-3.2-11B-Vision-Instruct.",
    )
    p.add_argument(
        "--resume_adapter_path",
        type=str,
        default=None,
        help="Optional PEFT adapter directory to continue training (same base as --model_name).",
    )
    p.add_argument(
        "--vgspv_csv",
        type=str,
        default=None,
        help="Train CSV (image, prompt, chosen_reasoning_trace, …). Default: train_method2.csv under repo if present.",
    )
    p.add_argument(
        "--vgspv_eval_csv",
        type=str,
        default=None,
        help="Validation CSV for eval_loss each epoch (default: test_method2.csv under repo if present).",
    )
    p.add_argument(
        "--vgspv_prompt_instruction",
        type=str,
        default=None,
        help="Fallback user text when a row has no `prompt` column (default: DEFAULT_PROMPT_INSTRUCTION).",
    )
    p.add_argument(
        "--vgspv_image_root",
        type=str,
        default=None,
        help="Optional base dir for resolving relative `image` paths in CSVs.",
    )
    p.add_argument(
        "--max_train_rows",
        type=int,
        default=None,
        help="Cap training rows after load (debug / smoke runs).",
    )
    p.add_argument(
        "--max_eval_rows",
        type=int,
        default=None,
        help="Cap validation rows after load (debug).",
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
        default=0,
        help="If > 0: save HF checkpoints every N global steps and run eval on --eval_steps. "
        "If 0 (default): save at end of every epoch and run eval_loss each epoch.",
    )
    p.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=0,
        help="When using --save_every_steps > 0 only: additionally call save_model every N completed epochs "
        "(0 = off). Ignored when --save_every_steps is 0.",
    )
    p.add_argument(
        "--save_total_limit",
        type=int,
        default=0,
        help="Max HF checkpoint folders to keep (oldest removed). 0 = no limit (keep every epoch).",
    )
    p.add_argument(
        "--save_optimizer_state",
        action="store_true",
        help="Include optimizer/scheduler in HF checkpoints (large). Default: adapter weights only.",
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
        "--eval_steps",
        type=int,
        default=None,
        help="Only when --save_every_steps > 0: run eval every N global steps (default: max(logging_steps, 50)).",
    )
    p.add_argument(
        "--eval_early_stopping_patience",
        type=int,
        default=0,
        help="Stop after this many evals without eval_loss improvement (0 = off). With per-epoch eval, "
        "one eval per epoch.",
    )
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--skip_eval", action="store_true", help="Disable eval_loss (train only).")
    return p.parse_args()


def _resolve_bbox_eval_csv_path(args: argparse.Namespace) -> Path | None:
    if args.skip_eval:
        return None
    if args.vgspv_eval_csv:
        p = Path(args.vgspv_eval_csv).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"--vgspv_eval_csv not found: {p}")
        return p
    if _DEFAULT_MM_EVAL_CSV.is_file():
        return _DEFAULT_MM_EVAL_CSV.resolve()
    return None


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapter_dir = out / "adapter"
    adapter_latest = out / "adapter_latest"

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    loaded = load_vlm(args.model_name, dtype=dtype, model_family=args.model_family)
    _print_training_device_banner(loaded)
    if loaded.family in ("tinyllava",):
        raise SystemExit(
            "Bounding-box SFT trainer currently supports HF processor models (llava, mllama, qwen3_vl). "
            f"Got family={loaded.family}. Extend train/bounding_box_sft_collator.py for TinyLLaVA if needed."
        )
    if loaded.processor is None:
        raise SystemExit("Bounding-box SFT requires a processor (LLaVA / Mllama / Qwen-VL).")

    if args.vgspv_csv:
        train_csv_path = Path(args.vgspv_csv).expanduser().resolve()
    elif _DEFAULT_MM_TRAIN_CSV.is_file():
        train_csv_path = _DEFAULT_MM_TRAIN_CSV.resolve()
    else:
        raise SystemExit(
            f"No train CSV: pass --vgspv_csv or add the default file at {_DEFAULT_MM_TRAIN_CSV}."
        )
    if not train_csv_path.is_file():
        raise SystemExit(f"Train CSV not found: {train_csv_path}")

    img_root = Path(args.vgspv_image_root).resolve() if args.vgspv_image_root else None
    csv_rows = load_vgspv_csv_rows_for_sft(str(train_csv_path), image_root=img_root)
    if args.max_train_rows is not None and int(args.max_train_rows) > 0:
        csv_rows = csv_rows[: int(args.max_train_rows)]
    print(f"[bbox_sft] Train CSV: {train_csv_path} ({len(csv_rows)} rows)", flush=True)

    vgspv_prompt = args.vgspv_prompt_instruction or DEFAULT_PROMPT_INSTRUCTION

    eval_csv_path = _resolve_bbox_eval_csv_path(args)
    csv_eval_rows: list[dict[str, Any]] | None = None
    if eval_csv_path is not None:
        csv_eval_rows = load_vgspv_csv_rows_for_sft(str(eval_csv_path), image_root=img_root)
        if args.max_eval_rows is not None and int(args.max_eval_rows) > 0:
            csv_eval_rows = csv_eval_rows[: int(args.max_eval_rows)]
        print(f"[bbox_sft] Eval CSV: {eval_csv_path} ({len(csv_eval_rows)} rows)", flush=True)

    eval_dataset_arg: BoundingBoxSFTCsvEvalDataset | None = None
    if not args.skip_eval and csv_eval_rows:
        eval_dataset_arg = BoundingBoxSFTCsvEvalDataset(csv_eval_rows)
    elif not args.skip_eval:
        print(
            "[bbox_sft] No eval CSV resolved; validation disabled. Set --vgspv_eval_csv or add "
            f"{_DEFAULT_MM_EVAL_CSV.name} under the repo.",
            flush=True,
        )

    collator = BoundingBoxSFTCollator(
        loaded.processor,
        loaded.family,
        csv_rows,
        csv_eval_rows=csv_eval_rows if eval_dataset_arg is not None else None,
        vgspv_prompt_instruction=vgspv_prompt,
    )
    torch_ds = BoundingBoxSFTCsvTrainDataset(csv_rows)

    if args.resume_adapter_path:
        normalize_no_split_modules_for_accelerate(loaded.model)
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

    use_step_checkpointing = bool(args.save_every_steps and int(args.save_every_steps) > 0)
    save_strategy = "steps" if use_step_checkpointing else "epoch"
    save_steps_val = int(args.save_every_steps) if use_step_checkpointing else 0

    save_total_limit_arg = int(args.save_total_limit)
    save_total_limit_kw: int | None = None if save_total_limit_arg <= 0 else save_total_limit_arg

    targs_kw: dict[str, Any] = dict(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit_kw,
        bf16=args.bf16,
        fp16=not args.bf16 and dtype == torch.float16,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        save_only_model=not bool(args.save_optimizer_state),
        save_safetensors=True,
    )
    if save_strategy == "steps":
        targs_kw["save_steps"] = save_steps_val

    eval_dataset_final = eval_dataset_arg
    esp = max(0, int(args.eval_early_stopping_patience))
    orig_save_strategy = save_strategy
    orig_save_steps_val = save_steps_val

    eval_steps_effective: int | None = None
    eval_schedule = "steps" if use_step_checkpointing else "epoch"

    if eval_dataset_final is not None:
        if use_step_checkpointing:
            eval_steps_effective = int(args.eval_steps) if args.eval_steps is not None else max(
                int(args.logging_steps), 50
            )
            eval_steps_effective = max(1, int(eval_steps_effective))
            if esp > 0:
                sync_steps = eval_steps_effective
                if save_strategy == "steps" and int(args.save_every_steps) > 0:
                    sync_steps = max(sync_steps, int(args.save_every_steps))
                sync_steps = max(1, sync_steps)
                eval_steps_effective = sync_steps
                targs_kw["save_strategy"] = "steps"
                targs_kw["save_steps"] = sync_steps
            if not apply_eval_scheduling_kwargs(
                targs_kw,
                eval_steps=int(eval_steps_effective),
                per_device_eval_batch_size=int(args.per_device_eval_batch_size),
            ):
                eval_dataset_final = None
                if esp > 0:
                    targs_kw["save_strategy"] = orig_save_strategy
                    if orig_save_strategy == "steps":
                        targs_kw["save_steps"] = orig_save_steps_val
                    else:
                        targs_kw.pop("save_steps", None)
            elif esp > 0:
                targs_kw["load_best_model_at_end"] = True
                targs_kw["metric_for_best_model"] = "eval_loss"
                targs_kw["greater_is_better"] = False
                print(
                    f"[bbox_sft] Early stopping on eval_loss: patience={esp} evals; "
                    f"eval/save every {eval_steps_effective} steps; load_best_model_at_end.",
                    flush=True,
                )
            if eval_dataset_final is not None:
                print(
                    f"[bbox_sft] Step mode: HF checkpoint + eval_loss every {eval_steps_effective} global steps.",
                    flush=True,
                )
        else:
            if not apply_epoch_eval_scheduling_kwargs(
                targs_kw,
                per_device_eval_batch_size=int(args.per_device_eval_batch_size),
            ):
                eval_dataset_final = None
            else:
                print(
                    "[bbox_sft] Epoch mode: HF checkpoint + eval_loss at the end of each epoch.",
                    flush=True,
                )
                if esp > 0:
                    targs_kw["load_best_model_at_end"] = True
                    targs_kw["metric_for_best_model"] = "eval_loss"
                    targs_kw["greater_is_better"] = False
                    print(
                        f"[bbox_sft] Early stopping on eval_loss: patience={esp} epochs without improvement; "
                        "load_best_model_at_end.",
                        flush=True,
                    )

    final_has_eval = eval_dataset_final is not None

    targs = instantiate_training_arguments(**targs_kw)
    if not args.save_optimizer_state:
        ckpt_desc = "per-epoch" if not use_step_checkpointing else "step"
        print(
            f"[bbox_sft] HF {ckpt_desc} checkpoints: model/adapter only (optimizer not saved). "
            "Use --save_optimizer_state for full HF state (large).",
            flush=True,
        )
    if save_total_limit_kw is None:
        print("[bbox_sft] save_total_limit: none (keep every epoch checkpoint).", flush=True)
    else:
        print(f"[bbox_sft] save_total_limit: {save_total_limit_kw}", flush=True)

    callbacks: list[TrainerCallback] = [
        SaveAdapterLatestCallback(adapter_latest, loaded.tokenizer, loaded.processor),
    ]
    if esp > 0 and eval_dataset_final is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=esp))
    epoch_cb: SaveEveryNEpochsCallback | None = None
    if use_step_checkpointing and args.save_every_n_epochs and int(args.save_every_n_epochs) > 0:
        epoch_cb = SaveEveryNEpochsCallback(
            int(args.save_every_n_epochs),
            adapter_latest,
            loaded.tokenizer,
            loaded.processor,
        )
        callbacks.append(epoch_cb)

    trainer = BoundingBoxSFTTrainer(
        model=model,
        args=targs,
        train_dataset=torch_ds,
        eval_dataset=eval_dataset_final,
        data_collator=collator,
        callbacks=callbacks,
        bbox_tokenizer=loaded.tokenizer,
        bbox_peft_save=_save_peft_bundle,
        bbox_adapter_latest_dir=adapter_latest,
        bbox_processor=loaded.processor,
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
        "train_csv": str(train_csv_path),
        "train_rows": len(csv_rows),
        "eval_csv": str(eval_csv_path) if eval_csv_path else None,
        "eval_rows": len(csv_eval_rows) if csv_eval_rows else 0,
        "eval_schedule": eval_schedule,
        "eval_steps": eval_steps_effective if final_has_eval and use_step_checkpointing else None,
        "eval_early_stopping_patience": esp,
        "per_device_eval_batch_size": int(args.per_device_eval_batch_size) if final_has_eval else None,
        "skip_eval": bool(args.skip_eval),
        "eval_enabled_for_training": final_has_eval,
        "vgspv_prompt_instruction": vgspv_prompt,
        "vgspv_image_root": str(img_root) if img_root else None,
        "max_train_rows": args.max_train_rows,
        "max_eval_rows": args.max_eval_rows,
        "resume_adapter_path": args.resume_adapter_path,
        "save_every_steps": int(args.save_every_steps),
        "save_strategy": save_strategy,
        "use_step_checkpointing": use_step_checkpointing,
        "save_every_n_epochs": args.save_every_n_epochs,
        "save_total_limit": save_total_limit_arg,
        "save_optimizer_state": bool(args.save_optimizer_state),
        "save_only_model": not bool(args.save_optimizer_state),
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
