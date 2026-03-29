"""Public load API for VLMs."""

from __future__ import annotations

from typing import Any

import torch

from vlm.registry import get_backend, resolve_family
from vlm.schema import LoadedVLM


def parse_dtype(dtype_str: str) -> torch.dtype:
    return {
        "auto": torch.bfloat16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_str]


def load_vlm(
    model_name: str,
    dtype: torch.dtype | str = "auto",
    device_map: str = "auto",
    quantization_config: Any = None,
    model_family: str | None = None,
) -> LoadedVLM:
    """
    Load a VLM via the appropriate backend.

    Returns LoadedVLM with .model, .tokenizer, .processor (if any), .family, .model_name.
    """
    if isinstance(dtype, str):
        dtype = parse_dtype(dtype)
    family = resolve_family(model_name, model_family)
    backend = get_backend(family)
    return backend.load(
        model_name,
        dtype=dtype,
        device_map=device_map,
        quantization_config=quantization_config,
    )


def load_vl_model_and_processor(
    model_name: str,
    dtype: torch.dtype | str = "auto",
    device_map: str = "auto",
    quantization_config: Any = None,
    model_family: str | None = None,
) -> tuple[Any, Any]:
    """
    Backward-compatible (model, processor_or_tokenizer) for scripts expecting the old tuple.

    For Qwen/LLaVA returns (model, processor). For TinyLLaVA returns (model, tokenizer).
    """
    loaded = load_vlm(
        model_name,
        dtype=dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        model_family=model_family,
    )
    second = loaded.processor if loaded.processor is not None else loaded.tokenizer
    return loaded.model, second
