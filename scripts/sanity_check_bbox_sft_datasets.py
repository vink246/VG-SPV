#!/usr/bin/env python3
"""
Quick sanity check: MM-SafetyBench / VG-SPV train CSV loads and at least one row becomes a bbox SFT sample.

Examples (from repo root):

  python scripts/sanity_check_bbox_sft_datasets.py
  python scripts/sanity_check_bbox_sft_datasets.py --csv data/mm-safebench_1/extracted_data/traces/train_method2.csv --max-rows 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    p = argparse.ArgumentParser(description="Sanity-check Method-2 CSV load for bbox SFT.")
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Train CSV path (default: train_method2.csv under repo if present).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=3,
        help="Cap rows after load (0 = no cap).",
    )
    p.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Optional base dir for resolving relative image paths (same as training --vgspv_image_root).",
    )
    args = p.parse_args()

    default_csv = _REPO / "data/mm-safebench_1/extracted_data/traces/train_method2.csv"
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else default_csv
    if not csv_path.is_file():
        print(f"FAIL: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    from train.bounding_box_sft_dataset import load_vgspv_csv_rows_for_sft, vgspv_csv_row_to_bbox_sft_sample
    from train.dataset_adapter import DEFAULT_PROMPT_INSTRUCTION

    img_root = Path(args.image_root).resolve() if args.image_root else None
    max_r = args.max_rows if args.max_rows and args.max_rows > 0 else None
    print(f"Loading {csv_path!r} max_rows={max_r!r} ...")
    rows = load_vgspv_csv_rows_for_sft(str(csv_path), image_root=img_root)
    if max_r is not None:
        rows = rows[:max_r]
    n = len(rows)
    if n == 0:
        print("FAIL: zero rows after load.", file=sys.stderr)
        return 1
    print(f"  rows={n}")

    for i in range(min(3, n)):
        try:
            vgspv_csv_row_to_bbox_sft_sample(rows[i], DEFAULT_PROMPT_INSTRUCTION)
            print(f"  row[{i}]: OK (SFT sample)")
        except Exception as e:
            print(f"FAIL: row[{i}] did not convert: {e}", file=sys.stderr)
            return 2

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
