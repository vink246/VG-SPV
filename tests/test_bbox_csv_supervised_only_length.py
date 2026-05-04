"""``BoundingBoxSFTMixedIndexDataset`` CSV-only training length and indexing."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from train.bounding_box_sft_torch_dataset import BoundingBoxSFTMixedIndexDataset


def test_csv_supervised_only_len_and_order() -> None:
    hf = list(range(100))  # len 100; unused when csv_supervised_only
    csv_rows = [{"i": 0}, {"i": 1}, {"i": 2}]
    ds = BoundingBoxSFTMixedIndexDataset(
        hf,
        csv_rows,
        mix_fraction=0.25,
        seed=0,
        csv_supervised_only=True,
    )
    assert len(ds) == 3
    assert ds[0] == {"idx": 0, "source": "vgspv_csv", "pool": "train"}
    assert ds[1] == {"idx": 1, "source": "vgspv_csv", "pool": "train"}
    assert ds[2] == {"idx": 2, "source": "vgspv_csv", "pool": "train"}


def test_default_mix_uses_hf_length() -> None:
    hf = list(range(10))
    csv_rows = [{"i": 0}]
    ds = BoundingBoxSFTMixedIndexDataset(hf, csv_rows, mix_fraction=0.0, seed=0, csv_supervised_only=False)
    assert len(ds) == 10
    assert ds[0]["source"] == "hf"
