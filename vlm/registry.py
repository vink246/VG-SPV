"""Map model name patterns to backend family keys."""

from __future__ import annotations

from typing import Any

# Substrings in model name (lowercase). First match wins.
# "tinyllava" must come before "llava".
VL_FAMILY_PATTERNS: list[tuple[str, str]] = [
    ("qwen3-vl", "qwen3_vl"),
    ("qwen2-vl", "qwen3_vl"),
    ("qwen", "qwen3_vl"),
    ("tinyllava", "tinyllava"),
    ("llava", "llava"),
]

KNOWN_FAMILIES = frozenset({"qwen3_vl", "llava", "tinyllava"})


def get_model_family(model_name: str) -> str:
    """Infer VL family from model id or path."""
    name_lower = model_name.lower()
    for pattern, family in VL_FAMILY_PATTERNS:
        if pattern in name_lower:
            return family
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
