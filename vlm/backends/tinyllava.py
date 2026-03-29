from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vlm.schema import LoadedVLM
from vlm.tinyllava_compat import apply_tinyllava_transformers_compat_patches


class TinyLLaVABackend:
    family = "tinyllava"

    def load(
        self,
        model_name: str,
        *,
        dtype: torch.dtype,
        device_map: str,
        quantization_config: Any,
    ) -> LoadedVLM:
        apply_tinyllava_transformers_compat_patches()

        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": dtype,
            "device_map": device_map,
            "attn_implementation": "eager",
        }
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        config = model.config
        tok_kwargs: dict[str, Any] = {"use_fast": False, "trust_remote_code": True}
        if hasattr(config, "tokenizer_model_max_length"):
            tok_kwargs["model_max_length"] = config.tokenizer_model_max_length
        if hasattr(config, "tokenizer_padding_side"):
            tok_kwargs["padding_side"] = config.tokenizer_padding_side
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        return LoadedVLM(
            model=model,
            tokenizer=tokenizer,
            family=self.family,
            model_name=model_name,
            processor=None,
        )
