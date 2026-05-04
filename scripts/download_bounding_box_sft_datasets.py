"""
Warm the Hugging Face cache for bounding-box SFT datasets (RefCOCO family).

Shikra / VoCoT-style training uses referring-expression comprehension data such as
RefCOCO, RefCOCO+, and RefCOCOg. Defaults use PaDT-processed HF releases (train splits)
and lmms-lab eval splits. For offline PACE runs, point ``HF_HOME`` (or ``HF_DATASETS_CACHE``)
at your scratch mirror and use ``train/run_bounding_box_sft.py --hf_local_files_only`` or
pass ``save_to_disk`` directories via ``--dataset_id`` / ``--extra_dataset``.

Usage (from repo root):
  python scripts/download_bounding_box_sft_datasets.py --preset all
  python scripts/download_bounding_box_sft_datasets.py --dataset_id PaDT-MLLM/RefCOCO --split train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PRESETS: dict[str, list[tuple[str, str]]] = {
    "train_rec": [
        ("PaDT-MLLM/RefCOCO", "train"),
    ],
    "eval_rec": [
        ("lmms-lab/RefCOCO", "val"),
        ("lmms-lab/RefCOCOplus", "val"),
        ("lmms-lab/RefCOCOg", "val"),
    ],
    "all": [],
}


def _build_all_preset() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for k, v in PRESETS.items():
        if k != "all":
            out.extend(v)
    return out


PRESETS["all"] = _build_all_preset()


def touch_dataset(dataset_id: str, split: str, config_name: str | None, max_rows: int) -> None:
    from datasets import load_dataset

    kwargs: dict = {}
    if config_name:
        kwargs["name"] = config_name
    print(f"Loading {dataset_id} split={split} ...")
    ds = load_dataset(dataset_id, split=split, **kwargs)
    n = len(ds)
    print(f"  rows={n}")
    if max_rows > 0:
        ds = ds.select(range(min(max_rows, n)))
    _ = ds[0]
    print("  ok (first row readable)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download / cache REC datasets for bounding-box SFT from Hugging Face.")
    p.add_argument(
        "--preset",
        type=str,
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Download a bundle (train_rec=PaDT train splits; eval_rec=lmms-lab val; all=union).",
    )
    p.add_argument("--dataset_id", type=str, default=None, help="Single HF dataset id.")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--max_rows", type=int, default=1, help="Touch at least this many rows (1=only verify).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    jobs: list[tuple[str, str]] = []
    if args.preset:
        jobs = list(PRESETS[args.preset])
    elif args.dataset_id:
        jobs = [(args.dataset_id, args.split)]
    else:
        print("Specify --preset or --dataset_id", file=sys.stderr)
        sys.exit(1)
    failed = 0
    for did, sp in jobs:
        try:
            touch_dataset(did, sp, args.dataset_config, args.max_rows)
        except Exception as e:
            print(f"FAILED {did} {sp}: {e}", file=sys.stderr)
            failed += 1
    if failed:
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
