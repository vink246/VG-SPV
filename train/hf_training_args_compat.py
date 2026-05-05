"""
Hugging Face ``TrainingArguments`` compatibility across ``transformers`` versions.

Some keyword names and flags appear or disappear between releases; unknown kwargs
raise ``TypeError``. We filter kwargs to the installed signature and pick the right
eval-scheduling parameter name (``eval_strategy`` vs ``evaluation_strategy``).
"""

from __future__ import annotations

import inspect
import sys
from typing import Any


def _transformers_version() -> str:
    try:
        import transformers

        return str(getattr(transformers, "__version__", "?"))
    except Exception:
        return "?"


def training_arguments_accepted_kwargs() -> set[str]:
    from transformers import TrainingArguments

    sig = inspect.signature(TrainingArguments.__init__)
    return {n for n in sig.parameters if n != "self"}


def training_arguments_accepts_var_keyword() -> bool:
    from transformers import TrainingArguments

    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in inspect.signature(TrainingArguments.__init__).parameters.values())


def filter_training_arguments_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of ``kwargs`` containing only names accepted by ``TrainingArguments.__init__``.

    If the signature includes ``**kwargs``, returns ``kwargs`` unchanged.
    Otherwise dropped keys are printed once to stderr (version mismatch / preemptive strip).
    """
    if training_arguments_accepts_var_keyword():
        return dict(kwargs)

    valid = training_arguments_accepted_kwargs()
    out: dict[str, Any] = {}
    dropped: list[str] = []
    for k, v in kwargs.items():
        if k in valid:
            out[k] = v
        else:
            dropped.append(k)
    if dropped:
        print(
            f"[hf_training_args_compat] transformers {_transformers_version()}: "
            f"ignoring unsupported TrainingArguments keys: {sorted(dropped)}",
            file=sys.stderr,
        )
    return out


def eval_strategy_parameter_name() -> str | None:
    """``eval_strategy`` (newer) or ``evaluation_strategy`` (older), or ``None`` if neither."""
    valid = training_arguments_accepted_kwargs()
    if "eval_strategy" in valid:
        return "eval_strategy"
    if "evaluation_strategy" in valid:
        return "evaluation_strategy"
    return None


def apply_epoch_eval_scheduling_kwargs(
    targs_kw: dict[str, Any],
    *,
    per_device_eval_batch_size: int,
) -> bool:
    """
    Run ``eval_loss`` at the end of each training epoch (aligned with ``save_strategy="epoch"``).
    """
    es = eval_strategy_parameter_name()
    if es is None:
        print(
            "[hf_training_args_compat] TrainingArguments has neither eval_strategy nor "
            "evaluation_strategy; epoch eval scheduling disabled.",
            file=sys.stderr,
        )
        return False
    targs_kw[es] = "epoch"
    targs_kw["per_device_eval_batch_size"] = int(per_device_eval_batch_size)
    return True


def apply_eval_scheduling_kwargs(
    targs_kw: dict[str, Any],
    *,
    eval_steps: int,
    per_device_eval_batch_size: int,
) -> bool:
    """
    Set eval-related keys on ``targs_kw`` in-place. Returns ``True`` if eval scheduling was applied.
    """
    es = eval_strategy_parameter_name()
    if es is None:
        print(
            "[hf_training_args_compat] TrainingArguments has neither eval_strategy nor "
            "evaluation_strategy; eval scheduling disabled.",
            file=sys.stderr,
        )
        return False
    targs_kw[es] = "steps"
    targs_kw["eval_steps"] = max(1, int(eval_steps))
    targs_kw["per_device_eval_batch_size"] = int(per_device_eval_batch_size)
    return True


def instantiate_training_arguments(**kwargs: Any):
    """Construct ``TrainingArguments`` after filtering unsupported keys for this install."""
    from transformers import TrainingArguments

    return TrainingArguments(**filter_training_arguments_kwargs(kwargs))
