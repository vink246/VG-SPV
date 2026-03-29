"""
Shared utilities for inference, training, and other scripts.
Model-agnostic: supports Qwen3-VL, LLaVA, and other VL families via a registry.
"""

from typing import Any

import types

import torch
from transformers import AutoProcessor


def _ensure_pretrained_tie_weights_compat_patch() -> None:
    """
    Newer transformers calls tie_weights(recompute_mapping=False) from init_weights.
    TinyLLaVA remote code defines tie_weights(self) without that kwarg — patch once per process.
    """
    import transformers.modeling_utils as modeling_utils

    if getattr(modeling_utils.PreTrainedModel, "_vg_spv_tie_weights_compat", False):
        return

    _orig_init_weights = modeling_utils.PreTrainedModel.init_weights

    def init_weights(self):
        orig_tw = self.tie_weights

        # MethodType passes the model as the first positional arg; orig_tw is already bound — use kwargs only.
        def wrapped(*args, **kwargs):
            kwargs.pop("recompute_mapping", None)
            try:
                return orig_tw(**kwargs)
            except TypeError:
                return orig_tw()

        self.tie_weights = types.MethodType(wrapped, self)
        try:
            _orig_init_weights(self)
        finally:
            self.tie_weights = orig_tw

    modeling_utils.PreTrainedModel.init_weights = init_weights
    modeling_utils.PreTrainedModel._vg_spv_tie_weights_compat = True


def _ensure_tinyllava_finalize_tie_weights_compat_patch() -> None:
    """
    After loading weights, transformers calls model.tie_weights(missing_keys=..., recompute_mapping=False).
    TinyLLaVA remote tie_weights(self) accepts neither — wrap only for model_type tinyllava.
    """
    import transformers.modeling_utils as modeling_utils

    if getattr(modeling_utils.PreTrainedModel, "_vg_spv_finalize_tie_compat", False):
        return

    _orig_finalize = modeling_utils.PreTrainedModel._finalize_model_loading

    @classmethod
    def _finalize_model_loading(cls, model, load_config, loading_info):
        is_tinyllava = getattr(model.config, "model_type", None) == "tinyllava" or type(model).__name__ == "TinyLlavaForConditionalGeneration"
        if not is_tinyllava:
            return _orig_finalize(cls, model, load_config, loading_info)

        orig_tw = model.tie_weights

        def compat_tw(*args, **kwargs):
            kwargs.pop("recompute_mapping", None)
            kwargs.pop("missing_keys", None)
            try:
                return orig_tw(**kwargs)
            except TypeError:
                return orig_tw()

        model.tie_weights = types.MethodType(compat_tw, model)
        try:
            return _orig_finalize(cls, model, load_config, loading_info)
        finally:
            model.tie_weights = orig_tw

    modeling_utils.PreTrainedModel._finalize_model_loading = _finalize_model_loading
    modeling_utils.PreTrainedModel._vg_spv_finalize_tie_compat = True


# Model family registry: maps family key to (model_class, processor_class) via lazy import.
# Add new families here; get_model_family() infers family from model name.
def _get_qwen3_vl_classes() -> tuple[type, type]:
    from transformers import Qwen3VLForConditionalGeneration
    return Qwen3VLForConditionalGeneration, AutoProcessor


def _get_llava_classes() -> tuple[type, type]:
    from transformers import LlavaForConditionalGeneration
    return LlavaForConditionalGeneration, AutoProcessor


VL_FAMILY_REGISTRY: dict[str, Any] = {
    "qwen3_vl": _get_qwen3_vl_classes,
    "llava": _get_llava_classes,
}

# Substrings in model name (lowercase) that select a family. First match wins.
# "tinyllava" must come before "llava" so tinyllava/* repos are not treated as HF LLaVA.
VL_FAMILY_PATTERNS: list[tuple[str, str]] = [
    ("qwen3-vl", "qwen3_vl"),
    ("qwen2-vl", "qwen3_vl"),  # Qwen2-VL uses same/similar API
    ("qwen", "qwen3_vl"),      # fallback for Qwen VL
    ("tinyllava", "tinyllava"),
    ("llava", "llava"),
]


def _load_tinyllava(
    model_name: str,
    dtype: torch.dtype,
    device_map: str,
    quantization_config: Any,
) -> tuple[Any, Any]:
    """
    TinyLLaVA Factory models (e.g. tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B) use custom code:
    AutoModelForCausalLM + trust_remote_code, and AutoTokenizer with config-driven settings.
    See https://huggingface.co/tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ensure_pretrained_tie_weights_compat_patch()
    _ensure_tinyllava_finalize_tie_weights_compat_patch()

    # attn_implementation="eager": TinyLLaVA's _supports_sdpa delegates to language_model, but
    # transformers checks SDPA during PreTrainedModel.__init__ before language_model is built — fixed in eager path.
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
    return model, tokenizer


def get_model_family(model_name: str) -> str:
    """
    Infer VL model family from model name or path.
    Used to select the right model class and inference path.
    """
    name_lower = model_name.lower()
    for pattern, family in VL_FAMILY_PATTERNS:
        if pattern in name_lower:
            return family
    return "qwen3_vl"  # default to Qwen3-VL for unknown names


def build_messages(image_paths: list[str], prompt: str) -> list[dict[str, Any]]:
    """Build chat messages with image(s) and text. Each image can be path or URL."""
    content = []
    for path in image_paths:
        path = path.strip()
        content.append({"type": "image", "image": path})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def parse_dtype(dtype_str: str) -> torch.dtype:
    """Map string to torch dtype for model loading."""
    return {
        "auto": torch.bfloat16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_str]


def load_vl_model_and_processor(
    model_name: str,
    dtype: torch.dtype | str = "auto",
    device_map: str = "auto",
    quantization_config: Any = None,
    model_family: str | None = None,
) -> tuple[Any, Any]:
    """
    Load a vision-language model and its processor by name/path.
    Model family is inferred from model_name if not provided (e.g. qwen3_vl, llava, tinyllava).
    Returns (model, processor). For Qwen/LLaVA, processor is AutoProcessor (use .tokenizer for training).
    For TinyLLaVA, the second value is the tokenizer itself (no nested .tokenizer).
    """
    if isinstance(dtype, str):
        dtype = parse_dtype(dtype)
    family = model_family if model_family is not None else get_model_family(model_name)
    if family == "tinyllava":
        return _load_tinyllava(model_name, dtype, device_map, quantization_config)
    if family not in VL_FAMILY_REGISTRY:
        raise ValueError(f"Unknown VL family: {family}. Known: {list(VL_FAMILY_REGISTRY.keys())} plus tinyllava")
    model_cls, processor_cls = VL_FAMILY_REGISTRY[family]()
    kwargs = {"torch_dtype": dtype, "device_map": device_map}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    model = model_cls.from_pretrained(model_name, **kwargs)
    processor = processor_cls.from_pretrained(model_name)
    return model, processor
