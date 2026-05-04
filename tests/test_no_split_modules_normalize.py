"""Regression: accelerate + PEFT when ``_no_split_modules`` is a ``set``."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from train.lora_factory import normalize_no_split_modules_for_accelerate


class _ToyWithSet(nn.Module):
    _no_split_modules = {"A", "B"}


def test_normalize_converts_set_on_instance_shadow() -> None:
    m = _ToyWithSet()
    assert isinstance(m.__class__._no_split_modules, set)
    normalize_no_split_modules_for_accelerate(m)
    assert isinstance(m._no_split_modules, tuple)
    assert set(m._no_split_modules) == {"A", "B"}


def test_tuple_unchanged() -> None:
    class _ToyTuple(nn.Module):
        _no_split_modules = ("X",)

    m = _ToyTuple()
    normalize_no_split_modules_for_accelerate(m)
    assert m._no_split_modules == ("X",)
