"""
Shared utilities for inference, training, and other scripts.

VLM loading and inference live in the ``vlm`` package (``LoadedVLM``, backends, ``load_vlm``).
This module re-exports the stable API and keeps ``build_messages`` for convenience.
"""

from __future__ import annotations

from typing import Any

from vlm import (
    LoadedVLM,
    get_model_family,
    load_vlm,
    load_vl_model_and_processor,
    parse_dtype,
)
from vlm.registry import VL_FAMILY_PATTERNS

__all__ = [
    "LoadedVLM",
    "VL_FAMILY_PATTERNS",
    "build_messages",
    "get_model_family",
    "load_vlm",
    "load_vl_model_and_processor",
    "parse_dtype",
]


def build_messages(image_paths: list[str], prompt: str) -> list[dict[str, Any]]:
    """Build chat messages with image(s) and text. Each image can be path or URL."""
    content = []
    for path in image_paths:
        path = path.strip()
        content.append({"type": "image", "image": path})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]
