"""Tests for ``scripts/draw_risk_factor_boxes`` phrase|box parsing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_draw_module():
    path = _REPO_ROOT / "scripts" / "draw_risk_factor_boxes.py"
    spec = importlib.util.spec_from_file_location("_draw_risk_factor_boxes_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_grid_box_and_no_box_bracketed() -> None:
    mod = _load_draw_module()
    text = (
        'phrase: "hand" | box: [0001, 0171, 0640, 0711]\n'
        'phrase: "gun" | box: [no_box]\n'
    )
    rows = mod.parse_phrase_box_lines(text)
    assert len(rows) == 2
    assert rows[0][0] == "hand"
    assert rows[0][1] is not None
    x0, _, _, _ = rows[0][1]
    assert abs(x0 - 1 / 1000) < 1e-6
    assert rows[1] == ("gun", None)


def test_parse_no_box_unbracketed() -> None:
    mod = _load_draw_module()
    rows = mod.parse_phrase_box_lines('phrase: "x" | box: no_box\n')
    assert rows == [("x", None)]


def test_draw_script_smoke(tmp_path: Path) -> None:
    from PIL import Image

    mod = _load_draw_module()
    img = Image.new("RGB", (100, 80), color=(40, 40, 40))
    entries = [("a", (0.1, 0.1, 0.9, 0.9)), ("b", None)]
    out = mod.draw_on_image(img, entries)
    assert out.size == img.size
