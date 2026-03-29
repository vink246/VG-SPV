"""Loaded model bundle returned by all VLM backends."""

from dataclasses import dataclass
from typing import Any


@dataclass
class LoadedVLM:
    """
    Unified result of loading a vision-language model.

    - tokenizer: always a PreTrainedTokenizer (for TRL / training).
    - processor: HF AutoProcessor for Qwen-VL / LLaVA; None for TinyLLaVA (tokenizer-only API).
    """

    model: Any
    tokenizer: Any
    family: str
    model_name: str
    processor: Any | None = None
