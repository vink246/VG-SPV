"""
Inference-specific utilities (VL generate + decode). Model-agnostic; dispatches by family.
Shared VL helpers and model registry live in root utils.py.
"""

import sys
from pathlib import Path
from typing import Any

# Ensure repo root on path when this module is imported (e.g. from eval or notebooks)
_INFERENCE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _INFERENCE_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import get_model_family


def _prepare_inputs_qwen3_vl(model: Any, processor: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Prepare model inputs for Qwen3-VL / Qwen2-VL using qwen_vl_utils."""
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return _to_device(inputs, model.device)


def _prepare_inputs_llava(model: Any, processor: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Prepare model inputs for LLaVA: load images from content and run processor."""
    from PIL import Image
    from pathlib import Path
    content = messages[0]["content"] if messages else []
    images = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "image":
            src = item.get("image")
            if isinstance(src, str):
                if src.startswith(("http://", "https://")):
                    import urllib.request
                    with urllib.request.urlopen(src) as f:
                        images.append(Image.open(f).convert("RGB"))
                else:
                    images.append(Image.open(Path(src)).convert("RGB"))
            else:
                images.append(src)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(images=images, text=[text], return_tensors="pt", padding=True)
    return _to_device(inputs, model.device)


def _to_device(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


_PREPARE_INPUTS: dict[str, Any] = {
    "qwen3_vl": _prepare_inputs_qwen3_vl,
    "llava": _prepare_inputs_llava,
}


def run_vl_inference(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    max_new_tokens: int = 256,
    do_sample: bool = False,
    model_family: str | None = None,
    model_name: str | None = None,
) -> str:
    """
    Run generate and decode for VL chat messages. Returns the assistant response text.
    model_family can be set explicitly (e.g. 'qwen3_vl', 'llava'); otherwise inferred from model_name.
    """
    family = model_family
    if family is None and model_name is not None:
        family = get_model_family(model_name)
    if family is None:
        family = "qwen3_vl"
    if family not in _PREPARE_INPUTS:
        raise ValueError(f"No inference path for VL family: {family}. Supported: {list(_PREPARE_INPUTS.keys())}")
    prepare_fn = _PREPARE_INPUTS[family]
    inputs = prepare_fn(model, processor, messages)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    input_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[:, input_len:]
    response = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return response
