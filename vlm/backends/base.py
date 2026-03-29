"""VLM backend protocol."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from vlm.schema import LoadedVLM


class VLMBackend(Protocol):
    """Loads a model + tokenizer (+ optional HF processor) for one family."""

    family: str

    def load(
        self,
        model_name: str,
        *,
        dtype: torch.dtype,
        device_map: str,
        quantization_config: Any,
    ) -> LoadedVLM:
        ...
