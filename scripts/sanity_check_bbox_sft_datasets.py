#!/usr/bin/env python3
"""
Quick sanity check: HF REC dataset loads and at least one row becomes a bbox SFT sample.

Uses the same loaders as ``train/run_bounding_box_sft.py``. Intended for PACE / local cache
before a long training run.

Examples (from repo root):

  python scripts/sanity_check_bbox_sft_datasets.py
  python scripts/sanity_check_bbox_sft_datasets.py --hf-local-files-only --max-samples 3
  python scripts/sanity_check_bbox_sft_datasets.py --dataset-id lmms-lab/RefCOCO --split val --max-samples 2
  python scripts/sanity_check_bbox_sft_datasets.py --image-root /path/to/coco  # override data/coco
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    p = argparse.ArgumentParser(description="Sanity-check HF dataset load for bbox SFT.")
    p.add_argument("--dataset-id", type=str, default="PaDT-MLLM/RefCOCO", help="HF hub id (primary source).")
    p.add_argument("--split", type=str, default="train", help="Split to load.")
    p.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Cap rows after load (small = fast). Use 0 for no cap.",
    )
    p.add_argument(
        "--hf-local-files-only",
        action="store_true",
        help="Same as training: only use local HF cache (no network).",
    )
    p.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="COCO root (train2014/, train2017/, …). Default: repo data/coco if present. Same as BBOX_SFT_IMAGE_ROOT.",
    )
    args = p.parse_args()

    from train.bounding_box_sft_dataset import hf_row_to_bbox_sft_sample, load_bbox_sft_hf_datasets

    max_s = args.max_samples if args.max_samples and args.max_samples > 0 else None
    print(f"Loading {args.dataset_id!r} split={args.split!r} max_samples={max_s!r} local_only={args.hf_local_files_only} ...")
    ds = load_bbox_sft_hf_datasets(
        [args.dataset_id],
        args.split,
        max_samples=max_s,
        config_name=None,
        local_files_only=bool(args.hf_local_files_only),
        image_root=args.image_root,
    )
    n = len(ds)
    if n == 0:
        print("FAIL: dataset has zero rows.", file=sys.stderr)
        return 1
    print(f"  rows={n}")

    for i in range(min(3, n)):
        try:
            hf_row_to_bbox_sft_sample(ds[i])
            print(f"  row[{i}]: OK (phrase + boxes -> SFT sample)")
        except Exception as e:
            print(f"FAIL: row[{i}] did not convert: {e}", file=sys.stderr)
            return 2

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
