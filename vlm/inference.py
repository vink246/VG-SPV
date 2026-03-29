"""Family-specific VL generation (chat or generate+decode)."""

from __future__ import annotations

from typing import Any

from vlm.registry import get_model_family
from vlm.schema import LoadedVLM


def _prepare_inputs_qwen3_vl(model: Any, processor: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
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
    from pathlib import Path

    from PIL import Image

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


def _extract_image_and_text_from_messages(messages: list[dict[str, Any]]) -> tuple[str | None, str]:
    content = messages[0]["content"] if messages else []
    image_path: str | None = None
    prompt_text = ""
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "image":
            image_path = item.get("image")
        elif item.get("type") == "text":
            prompt_text = item.get("text", "")
    return image_path, prompt_text


def _run_tinyllava_chat(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_new_tokens: int,
    do_sample: bool,
) -> str:
    image_path, prompt_text = _extract_image_and_text_from_messages(messages)
    if not image_path:
        raise ValueError("TinyLLaVA inference requires an image in messages")
    temperature = 0.7 if do_sample else 0.0
    out = model.chat(
        prompt=prompt_text,
        tokenizer=tokenizer,
        image=image_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    if isinstance(out, tuple):
        return out[0]
    return out


def run_vl_inference(
    loaded: LoadedVLM,
    messages: list[dict[str, Any]],
    max_new_tokens: int = 256,
    do_sample: bool = False,
) -> str:
    """
    Run inference for a loaded VLM. Dispatches by loaded.family.
    """
    family = loaded.family
    model = loaded.model

    if family == "tinyllava":
        return _run_tinyllava_chat(model, loaded.tokenizer, messages, max_new_tokens, do_sample)

    processor = loaded.processor
    if processor is None:
        raise ValueError(f"Family {family} requires a HF processor for inference")

    if family == "qwen3_vl":
        inputs = _prepare_inputs_qwen3_vl(model, processor, messages)
    elif family == "llava":
        inputs = _prepare_inputs_llava(model, processor, messages)
    else:
        raise ValueError(f"No inference path for family: {family}")

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    input_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[:, input_len:]
    return processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def run_vl_inference_legacy(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    max_new_tokens: int = 256,
    do_sample: bool = False,
    model_family: str | None = None,
    model_name: str | None = None,
) -> str:
    """
    Legacy signature: infer family from model_name and build a minimal LoadedVLM-like path.
    Prefer run_vl_inference(loaded, ...) when possible.
    """
    family = model_family
    if family is None and model_name is not None:
        family = get_model_family(model_name)
    if family is None:
        family = "qwen3_vl"

    if family == "tinyllava":
        return _run_tinyllava_chat(model, processor, messages, max_new_tokens, do_sample)

    if family not in ("qwen3_vl", "llava"):
        raise ValueError(f"No inference path for family: {family}")

    loaded = LoadedVLM(
        model=model,
        tokenizer=getattr(processor, "tokenizer", processor),
        family=family,
        model_name=model_name or "",
        processor=processor,
    )
    return run_vl_inference(loaded, messages, max_new_tokens, do_sample)
