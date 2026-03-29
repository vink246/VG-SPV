from __future__ import annotations

from vlm.backends.llava import LlavaBackend
from vlm.backends.mllama import MllamaBackend
from vlm.backends.qwen_vl import QwenVLBackend
from vlm.backends.tinyllava import TinyLLaVABackend

BACKENDS = {
    "qwen3_vl": QwenVLBackend(),
    "llava": LlavaBackend(),
    "tinyllava": TinyLLaVABackend(),
    "mllama": MllamaBackend(),
}

__all__ = ["BACKENDS", "QwenVLBackend", "LlavaBackend", "TinyLLaVABackend", "MllamaBackend"]
