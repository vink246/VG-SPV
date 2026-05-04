"""Torch datasets for MM-SafetyBench / VG-SPV CSV-only bounding-box SFT."""

from __future__ import annotations

from typing import Any

import torch.utils.data


class BoundingBoxSFTCsvTrainDataset(torch.utils.data.Dataset):
    """One index per training CSV row; collator loads ``vgspv_csv_row_to_bbox_sft_sample``."""

    def __init__(self, csv_rows: list[dict[str, Any]]):
        if not csv_rows:
            raise ValueError("BoundingBoxSFTCsvTrainDataset requires non-empty csv_rows")
        self._rows = csv_rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, int | str]:
        j = index % len(self._rows)
        return {"idx": int(j), "source": "vgspv_csv", "pool": "train"}


class BoundingBoxSFTCsvEvalDataset(torch.utils.data.Dataset):
    """Validation indices: one per eval CSV row."""

    def __init__(self, csv_eval_rows: list[dict[str, Any]]):
        if not csv_eval_rows:
            raise ValueError("BoundingBoxSFTCsvEvalDataset requires non-empty csv_eval_rows")
        self._rows = csv_eval_rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, int | str]:
        return {"idx": int(index), "source": "vgspv_csv", "pool": "eval"}


# Back-compat for imports expecting the old mixed/HF name
BoundingBoxSFTMixedIndexDataset = BoundingBoxSFTCsvTrainDataset
BoundingBoxSFTConcatEvalDataset = BoundingBoxSFTCsvEvalDataset
BoundingBoxSFTIndexDataset = BoundingBoxSFTCsvTrainDataset
