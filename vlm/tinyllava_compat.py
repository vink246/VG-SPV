"""
Compatibility shims for TinyLLaVA remote code with newer transformers.

Apply only when loading TinyLLaVA (see backends/tinyllava.py). Patches are process-global
on PreTrainedModel; idempotent per process.
"""

from __future__ import annotations

import types
from typing import Any


def apply_tinyllava_transformers_compat_patches() -> None:
    """Apply init_weights + _finalize_model_loading tie_weights compatibility. Safe to call once."""
    _ensure_pretrained_tie_weights_compat_patch()
    _ensure_tinyllava_finalize_tie_weights_compat_patch()


def _ensure_pretrained_tie_weights_compat_patch() -> None:
    import transformers.modeling_utils as modeling_utils

    if getattr(modeling_utils.PreTrainedModel, "_vg_spv_tie_weights_compat", False):
        return

    _orig_init_weights = modeling_utils.PreTrainedModel.init_weights

    def init_weights(self):
        orig_tw = self.tie_weights

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
    import transformers.modeling_utils as modeling_utils

    if getattr(modeling_utils.PreTrainedModel, "_vg_spv_finalize_tie_compat", False):
        return

    _finalize_desc = modeling_utils.PreTrainedModel.__dict__.get("_finalize_model_loading")
    _finalize_is_static = isinstance(_finalize_desc, staticmethod)
    _finalize_is_class = isinstance(_finalize_desc, classmethod)
    _orig_finalize = modeling_utils.PreTrainedModel._finalize_model_loading

    def _call_orig_finalize(model_cls: Any, model: Any, load_config: Any, loading_info: Any) -> Any:
        if _finalize_is_static:
            return _orig_finalize(model, load_config, loading_info)
        if _finalize_is_class:
            return _orig_finalize(model_cls, model, load_config, loading_info)
        return _orig_finalize(model_cls, model, load_config, loading_info)

    def _finalize_impl(model_cls: Any, model: Any, load_config: Any, loading_info: Any) -> Any:
        is_tinyllava = getattr(model.config, "model_type", None) == "tinyllava" or type(model).__name__ == "TinyLlavaForConditionalGeneration"
        if not is_tinyllava:
            return _call_orig_finalize(model_cls, model, load_config, loading_info)

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
            return _call_orig_finalize(model_cls, model, load_config, loading_info)
        finally:
            model.tie_weights = orig_tw

    if _finalize_is_static:

        def _finalize_model_loading(model, load_config, loading_info):
            return _finalize_impl(None, model, load_config, loading_info)

        modeling_utils.PreTrainedModel._finalize_model_loading = staticmethod(_finalize_model_loading)
    elif _finalize_is_class:

        @classmethod
        def _finalize_model_loading(cls, model, load_config, loading_info):
            return _finalize_impl(cls, model, load_config, loading_info)

        modeling_utils.PreTrainedModel._finalize_model_loading = classmethod(_finalize_model_loading)
    else:

        @classmethod
        def _finalize_model_loading(cls, model, load_config, loading_info):
            return _finalize_impl(cls, model, load_config, loading_info)

        modeling_utils.PreTrainedModel._finalize_model_loading = _finalize_model_loading
    modeling_utils.PreTrainedModel._vg_spv_finalize_tie_compat = True
