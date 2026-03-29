"""Map model name patterns to backend family keys."""

from __future__ import annotations

from typing import Any

# Substrings in model name (lowercase). First match wins.
# "tinyllava" must come before "llava".
# Mllama (Llama 3.2 Vision): match common Hub id shapes before falling back to config.model_type.
VL_FAMILY_PATTERNS: list[tuple[str, str]] = [
    ("qwen3-vl", "qwen3_vl"),
    ("qwen2-vl", "qwen3_vl"),
    ("qwen", "qwen3_vl"),
    ("llama-3.2v", "mllama"),
    ("llama-3.2-11b-vision", "mllama"),
    ("llama-3.2-90b-vision", "mllama"),
    ("tinyllava", "tinyllava"),
    ("llava", "llava"),
]

KNOWN_FAMILIES = frozenset({"qwen3_vl", "llava", "tinyllava", "mllama"})

# When config.model_type is read (no substring match), map to our backend key.
_MODEL_TYPE_TO_FAMILY: dict[str, str] = {
    "mllama": "mllama",
    "qwen2_vl": "qwen3_vl",
    "qwen3_vl": "qwen3_vl",
    "llava": "llava",
    "tinyllava": "tinyllava",
}


def _infer_family_from_config(model_name: str) -> str | None:
    """Use AutoConfig.model_type when the repo id does not match VL_FAMILY_PATTERNS."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        mt = getattr(config, "model_type", None)
        if isinstance(mt, str) and mt in _MODEL_TYPE_TO_FAMILY:
            return _MODEL_TYPE_TO_FAMILY[mt]
    except Exception:
        return None
    return None


def get_model_family(model_name: str) -> str:
    """Infer VL family from model id or path, then optional config.model_type."""
    name_lower = model_name.lower()
    for pattern, family in VL_FAMILY_PATTERNS:
        if pattern in name_lower:
            return family
    inferred = _infer_family_from_config(model_name)
    if inferred is not None:
        return inferred
    return "qwen3_vl"


def resolve_family(model_name: str, model_family: str | None) -> str:
    if model_family is not None:
        if model_family not in KNOWN_FAMILIES:
            raise ValueError(f"Unknown model_family: {model_family}. Known: {sorted(KNOWN_FAMILIES)}")
        return model_family
    return get_model_family(model_name)


def get_backend(family: str) -> Any:
    from vlm.backends import BACKENDS

    if family not in BACKENDS:
        raise ValueError(f"Unknown VL family: {family}. Known: {sorted(BACKENDS.keys())}")
    return BACKENDS[family]
