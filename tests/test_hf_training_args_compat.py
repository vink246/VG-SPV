"""Tests for TrainingArguments compatibility helpers."""

from __future__ import annotations

import importlib.util

import pytest

_HAS_TF = importlib.util.find_spec("transformers") is not None


@pytest.mark.skipif(not _HAS_TF, reason="transformers not installed")
def test_filter_training_arguments_kwargs_drops_unknown() -> None:
    from train.hf_training_args_compat import filter_training_arguments_kwargs

    base = dict(
        output_dir=".",
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        bf16=False,
        fp16=False,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        this_key_should_not_exist_on_any_training_args_ever_xyz=123,
    )
    filtered = filter_training_arguments_kwargs(base)
    assert "this_key_should_not_exist_on_any_training_args_ever_xyz" not in filtered
    assert filtered["output_dir"] == "."


@pytest.mark.skipif(not _HAS_TF, reason="transformers not installed")
def test_eval_strategy_parameter_name_is_one_or_other() -> None:
    from train.hf_training_args_compat import eval_strategy_parameter_name

    name = eval_strategy_parameter_name()
    assert name in (None, "eval_strategy", "evaluation_strategy")
