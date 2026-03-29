from __future__ import annotations

from typing import Any

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from vlm.schema import LoadedVLM


class LlavaBackend:
    family = "llava"

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
        model = LlavaForConditionalGeneration.from_pretrained(model_name, **kwargs)
        processor = AutoProcessor.from_pretrained(model_name)
        return LoadedVLM(
            model=model,
            tokenizer=processor.tokenizer,
            family=self.family,
            model_name=model_name,
            processor=processor,
        )
