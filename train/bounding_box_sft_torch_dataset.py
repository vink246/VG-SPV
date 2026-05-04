"""Index wrapper for HF RefCOCO-style rows, optionally mixed with VG-SPV CSV rows (deterministic per index)."""

from __future__ import annotations

from typing import Any

import torch.utils.data


class BoundingBoxSFTMixedIndexDataset(torch.utils.data.Dataset):
    """
    Each index maps to one training example over ``len(hf_dataset)`` steps per epoch.

    With probability ``mix_fraction`` (deterministic from ``index`` and ``seed``), the collator
    loads a VG-SPV CSV row; otherwise the HF dataset row ``index % len(hf)``.

    When ``csv_supervised_only`` is True and ``csv_rows`` is non-empty, every training step uses
    the CSV path only: epoch length is ``len(csv_rows)`` and index ``i`` supervises row ``i``.
    (HF rows are still loaded for eval / collator pools unless disabled elsewhere.) Use this to
    align SFT with ``train_method2.csv`` prompts + ``chosen_reasoning_trace`` before VG-fDPO.
    """

    def __init__(
        self,
        hf_dataset: Any,
        csv_rows: list[dict] | None = None,
        *,
        mix_fraction: float = 0.0,
        seed: int = 42,
        csv_supervised_only: bool = False,
    ):
        self.hf = hf_dataset
        self.csv_rows = csv_rows if csv_rows else None
        self.mix_fraction = float(max(0.0, min(1.0, mix_fraction)))
        self.seed = int(seed)
        self._csv_supervised_only = bool(csv_supervised_only) and bool(self.csv_rows)

    def __len__(self) -> int:
        if self._csv_supervised_only and self.csv_rows is not None:
            return len(self.csv_rows)
        return len(self.hf)

    def __getitem__(self, index: int) -> dict[str, int | str]:
        if self._csv_supervised_only and self.csv_rows is not None:
            j = index % len(self.csv_rows)
            return {"idx": int(j), "source": "vgspv_csv", "pool": "train"}
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
