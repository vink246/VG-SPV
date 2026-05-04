#!/usr/bin/env python3
"""
Bounding-box SFT in this repo uses **MM-SafetyBench / VG-SPV CSV traces** only (no Hugging Face RefCOCO download).

Prepare ``train_method2.csv`` / ``test_method2.csv`` under ``data/mm-safebench_1/extracted_data/traces/`` with your
pipeline (see ``scripts/generate_rejected_traces.py`` and related docs). Then run::

  python scripts/sanity_check_bbox_sft_datasets.py

This script is kept as a no-op entry point so old commands fail loudly with guidance instead of downloading PaDT/COCO.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "RefCOCO / PaDT / COCO downloads are no longer used for bbox SFT.\n"
        "Use MM-SafetyBench Method-2 CSVs and: python scripts/sanity_check_bbox_sft_datasets.py",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
