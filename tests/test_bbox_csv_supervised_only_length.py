"""``BoundingBoxSFTCsvTrainDataset`` length and indexing."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from train.bounding_box_sft_torch_dataset import BoundingBoxSFTCsvTrainDataset


def test_csv_train_len_and_order() -> None:
    csv_rows = [{"i": 0}, {"i": 1}, {"i": 2}]
    ds = BoundingBoxSFTCsvTrainDataset(csv_rows)
    assert len(ds) == 3
    assert ds[0] == {"idx": 0, "source": "vgspv_csv", "pool": "train"}
    assert ds[1] == {"idx": 1, "source": "vgspv_csv", "pool": "train"}
    assert ds[2] == {"idx": 2, "source": "vgspv_csv", "pool": "train"}


def test_csv_train_index_wraps() -> None:
    ds = BoundingBoxSFTCsvTrainDataset([{"a": 1}] * 5)
    assert ds[10]["idx"] == 0
