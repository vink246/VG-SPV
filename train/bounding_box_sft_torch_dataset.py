"""Index wrapper for HF RefCOCO-style rows, optionally mixed with VG-SPV CSV rows (deterministic per index)."""

from __future__ import annotations

from typing import Any

import torch.utils.data


class BoundingBoxSFTMixedIndexDataset(torch.utils.data.Dataset):
    """
    Each index maps to one training example over ``len(hf_dataset)`` steps per epoch.

    With probability ``mix_fraction`` (deterministic from ``index`` and ``seed``), the collator
    loads a VG-SPV CSV row; otherwise the HF dataset row ``index % len(hf)``.
    """

    def __init__(
        self,
        hf_dataset: Any,
        csv_rows: list[dict] | None = None,
        *,
        mix_fraction: float = 0.0,
        seed: int = 42,
    ):
        self.hf = hf_dataset
        self.csv_rows = csv_rows if csv_rows else None
        self.mix_fraction = float(max(0.0, min(1.0, mix_fraction)))
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.hf)

    def __getitem__(self, index: int) -> dict[str, int | str]:
        if self.csv_rows and self.mix_fraction > 0.0:
            u = ((index * 1103515245 + self.seed) & 0x7FFFFFFF) / float(0x7FFFFFFF)
            if u < self.mix_fraction:
                j = (index * 2654435761 + self.seed) % len(self.csv_rows)
                return {"idx": int(j), "source": "vgspv_csv", "pool": "train"}
        return {"idx": int(index % len(self.hf)), "source": "hf", "pool": "train"}


class BoundingBoxSFTConcatEvalDataset(torch.utils.data.Dataset):
    """
    Evaluation indices: all HF eval rows, then all VG-SPV CSV eval rows (disjoint from train).
    Each item includes ``pool="eval"`` for the collator.
    """

    def __init__(self, hf_eval: Any, csv_eval_rows: list[dict] | None):
        self.hf_eval = hf_eval
        self.n_hf = len(hf_eval) if hf_eval is not None else 0
        self.csv_eval = csv_eval_rows if csv_eval_rows else []
        self.n_csv = len(self.csv_eval)

    def __len__(self) -> int:
        return self.n_hf + self.n_csv

    def __getitem__(self, index: int) -> dict[str, int | str]:
        if index < self.n_hf:
            return {"idx": int(index), "source": "hf", "pool": "eval"}
        j = index - self.n_hf
        return {"idx": int(j), "source": "vgspv_csv", "pool": "eval"}


# Back-compat alias
BoundingBoxSFTIndexDataset = BoundingBoxSFTMixedIndexDataset
