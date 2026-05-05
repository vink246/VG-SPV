"""Tests for bbox SFT CSV loading (per-row prompt + image path resolution)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_DATASETS = importlib.util.find_spec("datasets") is not None


@pytest.mark.skipif(not _DATASETS, reason="datasets not installed")
def test_vgspv_csv_uses_per_row_prompt(tmp_path: Path) -> None:
    from PIL import Image

    from train.bounding_box_sft_dataset import (
        load_vgspv_csv_rows_for_sft,
        vgspv_csv_row_to_bbox_sft_sample,
        vgspv_csv_row_to_eval_user_messages,
    )
    from train.dataset_adapter import CHOSEN_REASONING_TRACE_COL, DPO_PROMPT_COL, IMAGE_COL

    img = tmp_path / "x.png"
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(img)

    csv = tmp_path / "rows.csv"
    csv.write_text(
        f"{IMAGE_COL},{DPO_PROMPT_COL},{CHOSEN_REASONING_TRACE_COL}\n"
        f'{img},"Ask A","<risk_factors_with_boxes>no risk</risk_factors_with_boxes>"\n'
        f'{img},"Ask B","<risk_factors_with_boxes>no risk</risk_factors_with_boxes>"\n',
        encoding="utf-8",
    )
    rows = load_vgspv_csv_rows_for_sft(str(csv))
    s0 = vgspv_csv_row_to_bbox_sft_sample(rows[0], "FALLBACK")
    s1 = vgspv_csv_row_to_bbox_sft_sample(rows[1], "FALLBACK")
    u0 = s0.messages[0]["content"][1]["text"]
    u1 = s1.messages[0]["content"][1]["text"]
    assert u0 == "Ask A"
    assert u1 == "Ask B"

    ev0 = vgspv_csv_row_to_eval_user_messages(rows[0], "FALLBACK")
    assert len(ev0) == 1
    assert ev0[0]["role"] == "user"
    assert ev0[0]["content"][1]["text"] == u0


@pytest.mark.skipif(not _DATASETS, reason="datasets not installed")
def test_vgspv_csv_resolves_repo_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    import train.bounding_box_sft_dataset as mod
    from train.bounding_box_sft_dataset import load_vgspv_csv_rows_for_sft
    from train.dataset_adapter import CHOSEN_REASONING_TRACE_COL, DPO_PROMPT_COL, IMAGE_COL

    repo = tmp_path / "repo"
    img_rel = Path("data") / "sub" / "im.png"
    (repo / img_rel).parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), color=(9, 9, 9)).save(repo / img_rel)

    csv = tmp_path / "rows.csv"
    csv.write_text(
        f"{IMAGE_COL},{DPO_PROMPT_COL},{CHOSEN_REASONING_TRACE_COL}\n"
        f'{img_rel.as_posix()},p,"<risk_factors_with_boxes>no risk</risk_factors_with_boxes>"\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "_REPO_ROOT", repo)
    rows = load_vgspv_csv_rows_for_sft(str(csv))
    assert Path(rows[0][IMAGE_COL]).is_file()
