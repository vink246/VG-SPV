"""
Generate Method 2 (spatially-grounded) preferred-response traces for VG-SPV.

This script consumes the trainer-ready Method 1 CSV produced by
``scripts.generate_method1_traces`` and, for each row, parses the Method 1 XML
trace to recover the risk factors, then runs Grounding DINO to obtain bounding
boxes. It then renders a ``<risk_factors_with_boxes>`` block in the exact
format trained for in ``train.bounding_box_sft_schema.USER_INSTRUCTION_BBOX_SFT``:

    phrase: "<risk>" | box: [bx0, by0, bx1, by1]

where each ``bxN/byN`` is an integer in [0, BOX_COORD_SCALE=1000] zero-padded
to 4 digits (produced via ``train.bounding_box_sft_schema.format_norm_box``).
The ``<logic>`` and ``<response>`` blocks are copied verbatim from the
Method 1 trace (string-level replacement, so no formatting drift).

Inputs (per split):
  - data/mm-safebench_1/extracted_data/traces/{split}_method1.csv
        Trainer-ready CSV from ``scripts.generate_method1_traces``.

Outputs (per split):
  - data/mm-safebench_1/extracted_data/traces/{split}_method2.csv
        Trainer-ready CSV (matches ``train/dataset_adapter.py``):
        ``image, perturbed_image, chosen_reasoning_trace, rejected_reasoning_trace``
        where ``chosen_reasoning_trace`` is the Method 2 XML trace.

Special cases:
  - "no risk" propagates as a single literal ``no risk`` line inside the tag.
  - A risk factor that yields zero Grounding DINO detections is recorded as
    ``phrase: "<risk>" | box: [no_box]`` so downstream training can choose to
    penalize or skip; this token does not match the int-grid box regex used by
    ``train.tag_parsing.parse_all_norm_boxes``.

Resume / overwrite semantics:
  - By default rows whose ``image`` is already present in the existing
    ``{split}_method2.csv`` are skipped (cheap resume after partial runs).
  - Pass ``--overwrite`` to re-process every row from scratch.

Invocation (always from repo root):
    python -m scripts.generate_method2_traces --split test \\
        --checkpoint weights/groundingdino_swint_ogc.pth

Pre-req: Grounding DINO weights downloaded (see README Setup §5).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# `inference.run_grounding_dino` imports `cv2` at module load, which pulls in the
# GroundingDINO stack. Defer that import to `cmd_run` so the pure-Python helpers in
# this module (used by smoke tests and by callers that only need formatting) work
# without the heavy CV deps installed.
from scripts.generate_method1_traces import parse_method1_xml
from train.bounding_box_sft_schema import (
    BOX_COORD_SCALE,
    TAG_RISK_WITH_BOXES,
    format_norm_box,
)

# Re-exported so callers can introspect the box scale; silences unused-import linters.
__all__ = ["BOX_COORD_SCALE", "TAG_RISK_WITH_BOXES", "main"]


DEFAULT_DATASET_DIR = Path("data/mm-safebench_1/extracted_data")


CSV_FIELDNAMES = [
    "image",
    "perturbed_image",
    "chosen_reasoning_trace",
    "rejected_reasoning_trace",
]


# ---------------------------- IO helpers ----------------------------


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _atomic_write_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    os.replace(tmp, path)


def _resolve_image_path_str(image_path_str: str) -> Path | None:
    """Resolve the CSV ``image`` field (repo-relative POSIX path) to an actual file."""
    if not image_path_str:
        return None
    candidate = Path(image_path_str)
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute():
        cwd_candidate = Path.cwd() / candidate
        if cwd_candidate.is_file():
            return cwd_candidate
    return None


# ---------------------------- trace assembly ----------------------------


def _build_caption(risk_factors: list[str]) -> str:
    """Format risk factors as a Grounding DINO caption: 'phrase1 . phrase2 . ... .'."""
    cleaned = [r.strip() for r in risk_factors if r.strip()]
    return " . ".join(cleaned) + " ."


def _match_phrase(returned_phrase: str, risk_factors: list[str]) -> str:
    """
    Map a Grounding DINO returned phrase back to the closest input risk factor.
    Case-insensitive substring containment in either direction; falls back to
    the raw phrase if nothing matches.
    """
    rp = returned_phrase.strip().lower()
    if not rp:
        return returned_phrase.strip()
    for rf in risk_factors:
        rfl = rf.strip().lower()
        if not rfl:
            continue
        if rp == rfl or rp in rfl or rfl in rp:
            return rf
    return returned_phrase.strip()


@dataclass
class _BoxLine:
    phrase: str
    formatted_box: str  # e.g. "[0150, 0200, 0550, 0800]" or "[no_box]"
    box_ints: list[int] | None  # parsed [bx0,by0,bx1,by1] or None for no_box


def _format_box_line(phrase: str, formatted_box: str) -> str:
    return f'phrase: "{phrase}" | box: {formatted_box}'


_FORMATTED_BOX_RE = re.compile(
    r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
)


def _parse_formatted_box_to_ints(formatted_box: str) -> list[int] | None:
    m = _FORMATTED_BOX_RE.search(formatted_box)
    if not m:
        return None
    return [int(m.group(i)) for i in range(1, 5)]


def _no_box_line(phrase: str) -> _BoxLine:
    return _BoxLine(phrase=phrase, formatted_box="[no_box]", box_ints=None)


def _detections_to_lines(
    risk_factors: list[str],
    boxes_xyxy: np.ndarray,
    labels: list[str],
    image_w: int,
    image_h: int,
) -> tuple[list[_BoxLine], list[dict[str, Any]]]:
    """
    Group detections by mapped risk-factor phrase, in stable order matching
    ``risk_factors``. Risks with zero detections get a ``[no_box]`` line.

    Returns:
        lines: ordered list of ``_BoxLine`` for the tag body.
        method2_boxes: ``[{phrase, box: [bx0,by0,bx1,by1]}, ...]`` int-grid records.
    """
    grouped: dict[str, list[_BoxLine]] = {rf: [] for rf in risk_factors}
    extras: list[_BoxLine] = []

    n = int(boxes_xyxy.shape[0]) if boxes_xyxy is not None else 0
    for i in range(n):
        x0, y0, x1, y1 = (float(v) for v in boxes_xyxy[i].tolist())
        if image_w <= 0 or image_h <= 0:
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        rx0 = max(0.0, min(1.0, x0 / image_w))
        ry0 = max(0.0, min(1.0, y0 / image_h))
        rx1 = max(0.0, min(1.0, x1 / image_w))
        ry1 = max(0.0, min(1.0, y1 / image_h))
        formatted = format_norm_box(rx0, ry0, rx1, ry1)
        ints = _parse_formatted_box_to_ints(formatted)
        raw_label = labels[i] if i < len(labels) else ""
        phrase = _match_phrase(raw_label, risk_factors)
        line = _BoxLine(phrase=phrase, formatted_box=formatted, box_ints=ints)
        if phrase in grouped:
            grouped[phrase].append(line)
        else:
            extras.append(line)

    ordered_lines: list[_BoxLine] = []
    boxes_records: list[dict[str, Any]] = []
    for rf in risk_factors:
        bucket = grouped.get(rf, [])
        if not bucket:
            ordered_lines.append(_no_box_line(rf))
        else:
            for ln in bucket:
                ordered_lines.append(ln)
                if ln.box_ints is not None:
                    boxes_records.append({"phrase": rf, "box": ln.box_ints})
    for ln in extras:
        ordered_lines.append(ln)
        if ln.box_ints is not None:
            boxes_records.append({"phrase": ln.phrase, "box": ln.box_ints})

    return ordered_lines, boxes_records


def _render_method2_trace(
    method1_trace: str,
    box_lines: list[_BoxLine],
    no_risk: bool,
) -> str:
    """
    Build the Method 2 trace by replacing the Method 1 ``<risk_factors>`` block with a
    new ``<risk_factors_with_boxes>`` block, keeping the original ``<logic>`` and
    ``<response>`` blocks verbatim. Falls back to assembling from parsed segments
    if the original ``<risk_factors>`` block can't be located.
    """
    if no_risk:
        new_block_inner = "no risk"
    else:
        new_block_inner = "\n".join(
            _format_box_line(ln.phrase, ln.formatted_box) for ln in box_lines
        )

    new_block = (
        f"<{TAG_RISK_WITH_BOXES}>\n{new_block_inner}\n</{TAG_RISK_WITH_BOXES}>"
    )

    pattern = re.compile(r"<risk_factors>.*?</risk_factors>", re.DOTALL | re.IGNORECASE)
    if pattern.search(method1_trace):
        return pattern.sub(new_block, method1_trace, count=1).strip()

    return method1_trace.strip()


def _is_no_risk(risk_factors: list[str]) -> bool:
    if not risk_factors:
        return True
    if len(risk_factors) == 1 and risk_factors[0].strip().lower() == "no risk":
        return True
    return False


# ---------------------------- main pipeline ----------------------------


def _filter_risk_factors_for_dino(risk_factors: list[str]) -> list[str]:
    """Drop empties and the literal 'no risk' sentinel before sending to Grounding DINO."""
    return [
        r.strip()
        for r in risk_factors
        if r.strip() and r.strip().lower() != "no risk"
    ]


def cmd_run(args: argparse.Namespace) -> None:
    dataset_dir: Path = args.dataset_dir
    split: str = args.split
    traces_dir = dataset_dir / "traces"
    method1_csv = traces_dir / f"{split}_method1.csv"
    method2_csv = traces_dir / f"{split}_method2.csv"

    if not method1_csv.is_file():
        raise SystemExit(
            f"Method 1 CSV not found: {method1_csv}. "
            f"Run `python -m scripts.generate_method1_traces ... collect --split {split}` first."
        )

    method1_rows = _read_csv_rows(method1_csv)
    print(f"Loaded {len(method1_rows)} rows from {method1_csv}.")

    existing_method2: dict[str, dict[str, str]] = {}
    if method2_csv.is_file() and not args.overwrite:
        for row in _read_csv_rows(method2_csv):
            img = row.get("image", "") or ""
            if img:
                existing_method2[img] = row
        if existing_method2:
            print(
                f"Resuming from {method2_csv}: {len(existing_method2)} rows "
                f"already processed (use --overwrite to redo)."
            )

    from inference.run_grounding_dino import (
        load_grounding_dino_model,
        resolve_config_from_checkpoint,
        run_grounding_dino,
    )

    config_path = args.config or resolve_config_from_checkpoint(str(args.checkpoint))
    print(f"Loading Grounding DINO from {config_path} and {args.checkpoint} on {args.device}...")
    model = load_grounding_dino_model(
        config_path=config_path,
        checkpoint_path=str(args.checkpoint),
        device=args.device,
    )

    output_rows: dict[str, dict[str, str]] = dict(existing_method2)
    n_processed = 0
    n_no_risk = 0
    n_skipped_no_image = 0
    n_skipped_unparseable = 0

    for row in method1_rows:
        image_field = (row.get("image") or "").strip()
        method1_trace = (row.get("chosen_reasoning_trace") or "").strip()

        if not method1_trace:
            n_skipped_unparseable += 1
            continue
        if not args.overwrite and image_field in output_rows:
            continue

        parsed = parse_method1_xml(method1_trace)
        if parsed is None:
            print(f"  [skip] image={image_field!r}: Method 1 trace XML did not parse")
            n_skipped_unparseable += 1
            continue
        risk_factors = list(parsed.get("risk_factors") or [])

        if _is_no_risk(risk_factors):
            method2_trace = _render_method2_trace(method1_trace, [], no_risk=True)
            output_rows[image_field] = {
                "image": image_field,
                "perturbed_image": "",
                "chosen_reasoning_trace": method2_trace,
                "rejected_reasoning_trace": "",
            }
            n_no_risk += 1
            n_processed += 1
            continue

        image_path = _resolve_image_path_str(image_field)
        if image_path is None:
            print(f"  [skip] image={image_field!r}: file not found on disk")
            n_skipped_no_image += 1
            continue

        try:
            with Image.open(image_path) as im:
                im_w, im_h = im.size
        except Exception as exc:
            print(f"  [skip] image={image_field!r}: failed to open ({exc})")
            n_skipped_no_image += 1
            continue

        dino_phrases = _filter_risk_factors_for_dino(risk_factors)
        if not dino_phrases:
            method2_trace = _render_method2_trace(method1_trace, [], no_risk=True)
            output_rows[image_field] = {
                "image": image_field,
                "perturbed_image": "",
                "chosen_reasoning_trace": method2_trace,
                "rejected_reasoning_trace": "",
            }
            n_no_risk += 1
            n_processed += 1
            continue

        caption = _build_caption(dino_phrases)
        try:
            boxes_xyxy, _scores, labels, _logits = run_grounding_dino(
                model=model,
                image_path=str(image_path),
                text_prompt=caption,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
        except Exception as exc:
            print(
                f"  [warn] image={image_field!r}: Grounding DINO failed ({exc}); "
                f"writing no_box trace"
            )
            box_lines, _boxes_records = _detections_to_lines(
                risk_factors=dino_phrases,
                boxes_xyxy=np.empty((0, 4), dtype=np.float32),
                labels=[],
                image_w=im_w,
                image_h=im_h,
            )
            method2_trace = _render_method2_trace(method1_trace, box_lines, no_risk=False)
            output_rows[image_field] = {
                "image": image_field,
                "perturbed_image": "",
                "chosen_reasoning_trace": method2_trace,
                "rejected_reasoning_trace": "",
            }
            n_processed += 1
            continue

        box_lines, _boxes_records = _detections_to_lines(
            risk_factors=dino_phrases,
            boxes_xyxy=boxes_xyxy,
            labels=labels,
            image_w=im_w,
            image_h=im_h,
        )
        method2_trace = _render_method2_trace(method1_trace, box_lines, no_risk=False)
        output_rows[image_field] = {
            "image": image_field,
            "perturbed_image": "",
            "chosen_reasoning_trace": method2_trace,
            "rejected_reasoning_trace": "",
        }
        n_processed += 1

        if n_processed % 25 == 0:
            print(
                f"  ...{n_processed} processed (no_risk={n_no_risk}, "
                f"skipped_no_image={n_skipped_no_image}, "
                f"skipped_unparseable={n_skipped_unparseable})"
            )

    sorted_rows = [output_rows[k] for k in sorted(output_rows.keys())]
    _atomic_write_csv(method2_csv, sorted_rows, CSV_FIELDNAMES)
    print(
        f"Wrote trainer-ready CSV {method2_csv} ({len(sorted_rows)} rows). "
        f"This run: processed={n_processed}, no_risk={n_no_risk}, "
        f"skipped_no_image={n_skipped_no_image}, "
        f"skipped_unparseable={n_skipped_unparseable}."
    )


# ---------------------------- CLI ----------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m scripts.generate_method2_traces",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "test"],
        help="MM-SafetyBench split to process.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to GroundingDINO checkpoint (.pth), e.g. weights/groundingdino_swint_ogc.pth.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional explicit GroundingDINO config (.py). If omitted, inferred from the checkpoint.",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=(
            f"MM-SafetyBench extracted_data root used to locate "
            f"traces/{{split}}_method1.csv and write traces/{{split}}_method2.csv "
            f"(default: {DEFAULT_DATASET_DIR})."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device for Grounding DINO (e.g. "cuda", "cpu"). Default: auto-detect.',
    )
    p.add_argument(
        "--box-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold on box logits.",
    )
    p.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Threshold on text scores.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-process every row from scratch (default: skip rows whose `image` "
            "is already present in the existing {split}_method2.csv)."
        ),
    )

    args = p.parse_args(argv)
    if args.device is None:
        try:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cmd_run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
