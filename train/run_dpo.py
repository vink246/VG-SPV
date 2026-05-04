"""
Launch DPO training with any supported VL model (TinyLLaVA, LLaVA, etc.) using VGSPVTrainer.

All hyperparameters live in ``configs/dpo.yaml``. Pass ``--config`` for another file;
CLI flags override YAML when provided.

Policy is trained with LoRA by default (``train/lora_factory.attach_lora``). Optional
``lora_adapter_path`` loads a bbox-SFT adapter first, then adds fresh DPO LoRA when
that checkpoint was merged into dense weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Allow running as python train/run_dpo.py from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig
from trl import DPOConfig

from train.dataset_adapter import DEFAULT_PROMPT_INSTRUCTION, load_dpo_dataset
from train.dpo_trainer import VGSPVTrainer
from train.dpo_yaml import DPOTrainConfig, default_dpo_config_path, dump_dpo_train_config_yaml, load_dpo_train_config, merge_dpo_train_config
from train.lora_factory import attach_lora, default_lora_config
from vlm import load_vlm, load_vlm_with_optional_lora


def _str2bool(s: str) -> bool:
    v = s.strip().lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean string, got {s!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run DPO training (VG-SPV; YAML config + CLI overrides).")
    p.add_argument("--config", type=str, default=None, help=f"YAML config path (default: {default_dpo_config_path()})")

    def add(name: str, t: Any, help: str):
        p.add_argument(f"--{name.replace('_', '-')}", type=t, default=None, help=help)

    add("model_name", str, "HF model id or local path")
    add("data_path", str, "Train split: DPO dataset CSV / dir / HF id")
    add("eval_data_path", str, "Eval/test split CSV / dir / HF id (optional; eval each epoch when set)")
    add("output_dir", str, "Training output directory")
    add("prompt_instruction", str, "Prompt text when loading from CSV")
    add("num_train_epochs", int, "")
    add("per_device_train_batch_size", int, "")
    add("gradient_accumulation_steps", int, "")
    add("learning_rate", float, "")
    add("max_length", int, "")
    add("beta", float, "DPO temperature")
    add("bf16", _str2bool, "Use bfloat16")
    add("ref_8bit", _str2bool, "Load reference model in 8-bit")
    add("logging_steps", int, "")
    add("save_steps", int, "")
    add("save_total_limit", int, "")
    add("use_vgfdpo", _str2bool, "Segment-masked VG-fDPO loss (paper L_VG-fDPO); on by default")
    add(
        "use_vdpo",
        _str2bool,
        "V-DPO contrastive term (paper L_V-DPO); OFF by default. Enabling requires "
        "chosen_perturbed_* dataset columns and ~doubles per-step forward FLOPs.",
    )
    add("alpha_vdpo", float, "Weight alpha on V-DPO term (paper Eq. 7)")
    add("vdpo_margin_m", float, "V-DPO hinge margin m")
    add("alpha_format", float, "Format-fail scaler (paper Eq. 5; typically 12–15)")
    add("grounding_mode", str, "semantic | spatial (VG-fDPO scaler s)")
    add("alpha_sem", float, "")
    add("alpha_iou", float, "")
    add("s_fallback_value", float, "")
    add("strict_scaler_inputs", _str2bool, "")
    add("lora_r", int, "LoRA rank for DPO policy")
    add("lora_alpha", int, "LoRA alpha")
    add("lora_dropout", float, "LoRA dropout")
    add("lora_adapter_path", str, "Optional bbox-SFT PEFT dir before DPO LoRA")
    add("merge_lora_adapter", _str2bool, "Merge bbox LoRA into dense weights before DPO LoRA")
    p.add_argument(
        "--dump-default-config",
        action="store_true",
        help="Print the default YAML config (from configs/dpo.yaml) to stdout and exit.",
    )
    return p.parse_args()


def _args_to_override_dict(ns: argparse.Namespace) -> dict[str, Any]:
    skip = {"config", "dump_default_config"}
    out: dict[str, Any] = {}
    for k, v in vars(ns).items():
        if k in skip:
            continue
        if v is not None:
            out[k] = v
    return out


def _prepare_policy_and_ref(cfg: DPOTrainConfig, dtype: torch.dtype) -> tuple[Any, Any | None, Any]:
    """
    Returns (policy_model, ref_model, tokenizer).

    Policy always uses LoRA: either continued from a merged bbox checkpoint, trainable
    bbox PEFT, or freshly attached DPO LoRA on the base VLM. Reference is a frozen
    copy of π_ref without the trainable DPO adapter (separate load when needed).
    """
    tokenizer: Any

    if cfg.lora_adapter_path:
        loaded = load_vlm_with_optional_lora(
            cfg.model_name,
            dtype=dtype,
            lora_adapter_path=cfg.lora_adapter_path,
            merge_adapter=cfg.merge_lora_adapter,
            is_trainable=True,
        )
        tokenizer = loaded.tokenizer
        inner = loaded.model

        if isinstance(inner, PeftModel) and not cfg.merge_lora_adapter:
            ref_kw: dict[str, Any] = {"dtype": dtype}
            if cfg.ref_8bit:
                ref_kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            ref_loaded = load_vlm(cfg.model_name, **ref_kw)
            ref_model = ref_loaded.model
            for p in ref_model.parameters():
                p.requires_grad = False
            return inner, ref_model, tokenizer

        for p in inner.parameters():
            p.requires_grad = False
        lcfg = default_lora_config(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
        policy = attach_lora(inner, lora_config=lcfg, freeze_vision=True, prepare_kbit=cfg.ref_8bit)
        return policy, inner, tokenizer

    if cfg.ref_8bit:
        policy_loaded = load_vlm(cfg.model_name, dtype=dtype)
        ref_loaded = load_vlm(
            cfg.model_name,
            dtype=dtype,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        ref_model = ref_loaded.model
        for p in ref_model.parameters():
            p.requires_grad = False
        tokenizer = policy_loaded.tokenizer
        lcfg = default_lora_config(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
        policy = attach_lora(
            policy_loaded.model,
            lora_config=lcfg,
            freeze_vision=True,
            prepare_kbit=False,
        )
        return policy, ref_model, tokenizer

    loaded = load_vlm(cfg.model_name, dtype=dtype)
    ref_model = loaded.model
    for p in ref_model.parameters():
        p.requires_grad = False
    tokenizer = loaded.tokenizer
    lcfg = default_lora_config(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
    policy = attach_lora(loaded.model, lora_config=lcfg, freeze_vision=True, prepare_kbit=False)
    return policy, ref_model, tokenizer


def main() -> None:
    args = parse_args()
    if args.dump_default_config:
        cfg0 = load_dpo_train_config(default_dpo_config_path())
        print(dump_dpo_train_config_yaml(cfg0), end="")
        return

    cfg_path = args.config or str(default_dpo_config_path())
    base_cfg = load_dpo_train_config(cfg_path)
    cfg = merge_dpo_train_config(base_cfg, _args_to_override_dict(args))

    if cfg.merge_lora_adapter and not cfg.lora_adapter_path:
        raise SystemExit("--merge-lora-adapter requires --lora-adapter-path in YAML or CLI.")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    model, ref_model, tokenizer = _prepare_policy_and_ref(cfg, dtype)

    pi = cfg.prompt_instruction or DEFAULT_PROMPT_INSTRUCTION
    train_dataset = load_dpo_dataset(cfg.data_path, prompt_instruction=pi)

    eval_dataset = None
    if cfg.eval_data_path:
        eval_dataset = load_dpo_dataset(cfg.eval_data_path, prompt_instruction=pi, csv_hf_split="test")

    training_args = DPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        max_length=cfg.max_length,
        beta=cfg.beta,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="epoch" if eval_dataset is not None else "no",
    )

    trainer = VGSPVTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        use_vgfdpo=cfg.use_vgfdpo,
        use_vdpo=cfg.use_vdpo,
        alpha_vdpo=cfg.alpha_vdpo,
        vdpo_margin_m=cfg.vdpo_margin_m,
        alpha_format=cfg.alpha_format,
        grounding_mode=cfg.grounding_mode,
        alpha_sem=cfg.alpha_sem,
        alpha_iou=cfg.alpha_iou,
        s_fallback_value=cfg.s_fallback_value,
        strict_scaler_inputs=cfg.strict_scaler_inputs,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)


if __name__ == "__main__":
    main()
