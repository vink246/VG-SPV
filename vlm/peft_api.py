"""Load HF VLMs with optional bounding-box SFT LoRA adapters (PEFT)."""

from __future__ import annotations

from typing import Any

import torch
from peft import PeftModel

from train.lora_factory import normalize_no_split_modules_for_accelerate
from vlm.api import load_vlm, parse_dtype
from vlm.schema import LoadedVLM


def load_vlm_with_optional_lora(
    model_name: str,
    *,
    lora_adapter_path: str | None = None,
    dtype: torch.dtype | str = "auto",
    device_map: str = "auto",
    quantization_config: Any = None,
    model_family: str | None = None,
    is_trainable: bool = True,
    merge_adapter: bool = False,
) -> LoadedVLM:
    """
    Load a VLM and optionally attach a PEFT adapter saved by `train/run_bounding_box_sft.py`
    (e.g. output ``adapter/`` or periodically overwritten ``adapter_latest/``).

    - `merge_adapter=True` merges LoRA into base weights (single forward pass, no PEFT at runtime).
    - For DPO after bbox SFT, use `merge_adapter=False` and `is_trainable=True` on the policy;
      `train/run_dpo.py` then adds a separate DPO LoRA stack when the bbox weights are merged
      into dense base weights.
    """
    if merge_adapter and not lora_adapter_path:
        raise ValueError("merge_adapter=True requires lora_adapter_path to a saved PEFT adapter directory.")
    if isinstance(dtype, str):
        dtype = parse_dtype(dtype)
    loaded = load_vlm(
        model_name,
        dtype=dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        model_family=model_family,
    )
    if not lora_adapter_path:
        return loaded
    normalize_no_split_modules_for_accelerate(loaded.model)
    model = PeftModel.from_pretrained(
        loaded.model,
        lora_adapter_path,
        is_trainable=is_trainable,
    )
    if merge_adapter:
        model = model.merge_and_unload()
    loaded.model = model
    return loaded
