"""Balanced HF + CSV eval subsampling for bounding-box SFT."""

from __future__ import annotations

from datasets import Dataset


def test_subsample_balanced_eval_shapes_and_reproducibility() -> None:
    from train.bounding_box_sft_eval_sampling import subsample_balanced_eval

    hf = Dataset.from_dict({"x": list(range(500))})
    csv = [{"i": i} for i in range(300)]
    hf_a, csv_a = subsample_balanced_eval(hf, csv, seed=42, n_per_side=100)
    hf_b, csv_b = subsample_balanced_eval(hf, csv, seed=42, n_per_side=100)
    assert len(hf_a) == 100
    assert len(csv_a) == 100
    assert [r["x"] for r in hf_a] == [r["x"] for r in hf_b]
    assert csv_a == csv_b

    hf_c, csv_c = subsample_balanced_eval(hf, csv, seed=99, n_per_side=100)
    assert [r["x"] for r in hf_a] != [r["x"] for r in hf_c] or csv_a != csv_c
