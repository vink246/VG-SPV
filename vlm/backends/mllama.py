"""Llama 3.2 Vision (Mllama) — Hugging Face `model_type`: mllama."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration

from vlm.schema import LoadedVLM


class MllamaBackend:
    family = "mllama"

    def load(
        self,
        model_name: str,
        *,
        dtype: torch.dtype,
        device_map: str,
        quantization_config: Any,
    ) -> LoadedVLM:
        kwargs: dict[str, Any] = {"torch_dtype": dtype, "device_map": device_map}
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        model = MllamaForConditionalGeneration.from_pretrained(model_name, **kwargs)
        processor = AutoProcessor.from_pretrained(model_name)
        return LoadedVLM(
            model=model,
            tokenizer=processor.tokenizer,
            family=self.family,
            model_name=model_name,
            processor=processor,
        )
