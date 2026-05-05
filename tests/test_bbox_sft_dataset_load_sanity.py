"""
Optional HF load sanity (same logic as ``scripts/sanity_check_bbox_sft_datasets.py``).

Set ``RUN_BBOX_DATASET_SANITY=1`` to run against the Hub / your HF cache (quick: max 3 rows).
Otherwise the test is skipped so CI and laptops without cache do not fail.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(
    os.environ.get("RUN_BBOX_DATASET_SANITY", "") != "1",
    reason="Set RUN_BBOX_DATASET_SANITY=1 to run HF load sanity (see scripts/sanity_check_bbox_sft_datasets.py).",
)
def test_hf_refcoco_load_and_first_rows_sft_sample() -> None:
    """Loads PaDT RefCOCO (tiny cap) and ensures first rows convert like training."""
    script = _REPO / "scripts" / "sanity_check_bbox_sft_datasets.py"
    r = subprocess.run(
        [sys.executable, str(script), "--max-samples", "3"],
        cwd=str(_REPO),
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"


def test_sanity_script_exists() -> None:
    assert (_REPO / "scripts" / "sanity_check_bbox_sft_datasets.py").is_file()
