"""
Inference helpers.

Prefer ``from vlm import load_vlm, run_vl_inference`` and pass a ``LoadedVLM``.

Legacy: ``run_vl_inference(model, processor, ...)`` still works via ``run_vl_inference_legacy``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_INFERENCE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _INFERENCE_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vlm import get_model_family
from vlm.inference import run_vl_inference as run_vl_inference_loaded
from vlm.inference import run_vl_inference_legacy


def run_vl_inference(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    max_new_tokens: int = 256,
    do_sample: bool = False,
    model_family: str | None = None,
    model_name: str | None = None,
) -> str:
    """Legacy signature (separate model + processor). Prefer ``vlm.run_vl_inference(loaded, ...)``."""
    return run_vl_inference_legacy(
        model,
        processor,
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        model_family=model_family,
        model_name=model_name,
    )


__all__ = [
    "get_model_family",
    "run_vl_inference",
    "run_vl_inference_loaded",
    "run_vl_inference_legacy",
]
