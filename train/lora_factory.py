"""
Attach LoRA (PEFT) to a loaded HF VLM for bounding-box SFT (and similar finetunes).

Freezes vision towers by default; trains language-model LoRA (+ optional projector LoRA).
"""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

_DEFAULT_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def freeze_vision_parameters(model: Any) -> None:
    """Disable gradients for common vision encoder name patterns."""
    for name, p in model.named_parameters():
        ln = name.lower()
        if any(
            k in ln
            for k in (
                "vision_tower",
                "vision_model",
                "visual_encoder",
                "image_newline",
                "image_pooling",
            )
        ):
            p.requires_grad = False


def default_lora_config(
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    bias: str = "none",
) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type="CAUSAL_LM",
        target_modules=list(target_modules or _DEFAULT_TARGETS),
    )


def attach_lora(
    model: Any,
    *,
    lora_config: LoraConfig | None = None,
    freeze_vision: bool = True,
    prepare_kbit: bool = False,
) -> PeftModel:
    """
    Wrap `model` with trainable LoRA adapters.

    If the base weights are loaded in 8-bit, set `prepare_kbit=True`.
    """
    if prepare_kbit:
        model = prepare_model_for_kbit_training(model)
    if freeze_vision:
        freeze_vision_parameters(model)
    cfg = lora_config or default_lora_config()
    return get_peft_model(model, cfg)


def merge_lora_if_requested(model: PeftModel, merge: bool = False) -> Any:
    """Optionally merge adapters into base weights for deployment."""
    if merge:
        return model.merge_and_unload()
    return model
