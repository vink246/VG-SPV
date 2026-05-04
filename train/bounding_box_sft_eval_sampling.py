"""Balanced HF + CSV eval subsampling (no torch import)."""

from __future__ import annotations

import random
import sys
from typing import Any

EVAL_BALANCE_SEED_OFFSET = 100_003


def subsample_balanced_eval(
    hf_eval: Any,
    csv_rows: list[dict],
    *,
    seed: int,
    n_per_side: int = 100,
) -> tuple[Any, list[dict]]:
    """
    Fixed-size balanced eval: up to ``n_per_side`` HF rows + ``n_per_side`` CSV rows.

    Same ``seed`` (+ ``EVAL_BALANCE_SEED_OFFSET``) always yields the same indices.
    """
    n_hf = len(hf_eval)
    n_csv = len(csv_rows)
    take = min(int(n_per_side), n_hf, n_csv)
    if take <= 0:
        return hf_eval, list(csv_rows)
    rng = random.Random(int(seed) + EVAL_BALANCE_SEED_OFFSET)
    hf_idx = sorted(rng.sample(range(n_hf), take))
    csv_idx = sorted(rng.sample(range(n_csv), take))
    hf_sub = hf_eval.select(hf_idx)
    csv_sub = [csv_rows[i] for i in csv_idx]
    total = take * 2
    print(
        f"[eval] Balanced validation subset: {take} COCO (HF) + {take} VG-fDPO CSV = {total} examples "
        f"(seed offset {EVAL_BALANCE_SEED_OFFSET} on mix_seed={int(seed)}).",
        flush=True,
    )
    if take < n_per_side:
        print(
            f"[eval] Note: requested {n_per_side}+{n_per_side} but pools are n_hf={n_hf}, n_csv={n_csv}; "
            f"using {take} per side.",
            file=sys.stderr,
            flush=True,
        )
    return hf_sub, csv_sub
