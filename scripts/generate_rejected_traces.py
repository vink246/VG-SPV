"""
Fill ``rejected_reasoning_trace`` for VG-SPV DPO CSVs using:

  - **abliterated**: conditional prompts + Llama 3.1 abliterated LM for branches A/B;
    branch C is fully hardcoded (no LM).
  - **bbox_perturb** (Method 2 only): corrupt ``<risk_factors_with_boxes>`` coordinates;
    ``<logic>`` / ``<response>`` stay verbatim from the chosen trace.

Invoke from repo root:

    python -m scripts.generate_rejected_traces \\
        --input data/mm-safebench_1/extracted_data/traces/test_method2.csv \\
        --output data/mm-safebench_1/extracted_data/traces/test_method2_dpo.csv \\
        --split test \\
        --method method2 \\
        --rejection-modes abliterated bbox_perturb

Requires metadata JSONL (default ``{dataset_dir}/{split}_metadata.jsonl``) to recover the
MM-SafetyBench **question** text; join keys match ``scripts.generate_method1_traces``
collect semantics (run with **cwd = repo root** so ``image`` paths align).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import zlib
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference.run_abliterated_llama import (  # noqa: E402
    DEFAULT_MODEL_ID,
    generate_completion,
    load_model_and_tokenizer,
)
from scripts.generate_method1_traces import (  # noqa: E402
    _csv_image_field,
    _iter_metadata,
    _resolve_image_path,
    _row_passes_mmsb_filter,
)
from train.rejected_trace_builder import (  # noqa: E402
    MethodTag,
    ParsedTrace,
    build_method2_bbox_perturb_rejected,
    build_rejected_trace_branch_ab,
    build_rejected_trace_branch_c,
    classify_branch,
    parse_trace_for_rejection,
    prepare_abliterated_prompt,
)

CSV_FIELDNAMES = (
    "image",
    "perturbed_image",
    "chosen_reasoning_trace",
    "rejected_reasoning_trace",
)
VARIANT_COL = "rejected_variant"

DEFAULT_DATASET_DIR = Path("data/mm-safebench_1/extracted_data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=str, required=True, help="Source CSV (method1 or method2 traces).")
    p.add_argument("--output", type=str, required=True, help="Output CSV path.")
    p.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR), help="MM-SafetyBench extracted_data root.")
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override for image resolution (defaults to --dataset-dir).",
    )
    p.add_argument("--split", type=str, default="test", help="Split name for default metadata path.")
    p.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to {split}_metadata.jsonl (default: {dataset-dir}/{split}_metadata.jsonl).",
    )
    p.add_argument("--method", type=str, choices=("method1", "method2"), required=True)
    p.add_argument(
        "--rejection-modes",
        nargs="+",
        choices=("abliterated", "bbox_perturb"),
        default=["abliterated"],
        help="One or both; bbox_perturb requires --method method2.",
    )
    p.add_argument(
        "--bbox-zero-iou-fraction",
        type=float,
        default=0.2,
        help="Per box line: probability of sampling an IoU≈0 perturbation vs partial overlap.",
    )
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--device-map", type=str, default=None)
    p.add_argument("--cache-dir", type=str, default=None)
    return p.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    from utils import parse_dtype

    if name == "float32":
        return torch.float32
    return parse_dtype(name)


def build_question_by_image_map(
    metadata_path: Path,
    data_root: Path,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in _iter_metadata(metadata_path):
        if not _row_passes_mmsb_filter(row):
            continue
        rel_image = row.get("image")
        question = row.get("question")
        if not rel_image or question is None:
            continue
        resolved = _resolve_image_path(str(rel_image), data_root)
        key = _csv_image_field(resolved)
        out[key] = str(question).strip()
    return out


def row_rng(seed: int | None, row_idx: int, image_key: str) -> random.Random:
    base = (int(seed) if seed is not None else 0) & 0xFFFFFFFF
    h = zlib.adler32(image_key.encode("utf-8", errors="ignore")) & 0xFFFFFFFF
    return random.Random((base ^ h ^ (row_idx * 0x9E3779B9)) & 0xFFFFFFFF)


def _needs_variant_column(modes: set[str]) -> bool:
    return len(modes) > 1 or "bbox_perturb" in modes


def run_abliterated_rejected(
    *,
    parsed: ParsedTrace,
    question: str,
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> str | None:
    br = classify_branch(parsed)
    if br == "C":
        return build_rejected_trace_branch_c(
            risk_block=parsed.risk_block_verbatim,
            logic_inner=parsed.logic_inner,
        )
    prep = prepare_abliterated_prompt(parsed, question)
    if prep is None:
        return None
    _branch, user_prompt = prep
    messages = [{"role": "user", "content": user_prompt}]
    _full, assistant_text = generate_completion(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    return build_rejected_trace_branch_ab(
        parsed=parsed,
        abliterated_response_body=assistant_text,
    )


def main() -> None:
    args = parse_args()
    modes = set(args.rejection_modes)
    method: MethodTag = args.method  # type: ignore[assignment]

    if "bbox_perturb" in modes and method != "method2":
        raise SystemExit("bbox_perturb rejection mode requires --method method2.")

    dataset_dir = Path(args.dataset_dir)
    data_root = Path(args.data_root) if args.data_root else dataset_dir
    meta_path = Path(args.metadata) if args.metadata else dataset_dir / f"{args.split}_metadata.jsonl"
    if "abliterated" in modes:
        if not meta_path.is_file():
            raise SystemExit(f"Metadata not found (required for abliterated mode): {meta_path}")
        qmap = build_question_by_image_map(meta_path, data_root)
    else:
        qmap = {}

    model = tokenizer = None
    if "abliterated" in modes:
        torch_dtype = _torch_dtype(args.dtype)
        model, tokenizer = load_model_and_tokenizer(
            model_id=args.model_id,
            torch_dtype=torch_dtype,
            device=args.device,
            device_map=args.device_map,
            cache_dir=args.cache_dir,
        )

    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Input CSV not found: {in_path}")

    rows_in = _read_csv_rows(in_path)
    out_rows: list[dict[str, str]] = []
    n_ab_fail = 0
    n_bbox_fail = 0

    for idx, row in enumerate(rows_in):
        chosen = row.get("chosen_reasoning_trace", "").strip()
        img_key = row.get("image", "").strip()
        base_row = {k: row.get(k, "") for k in CSV_FIELDNAMES}

        if "abliterated" in modes:
            assert model is not None and tokenizer is not None
            question = qmap.get(img_key)
            parsed = parse_trace_for_rejection(chosen, method)
            if not chosen or not question or parsed is None:
                n_ab_fail += 1
            else:
                rej_a = run_abliterated_rejected(
                    parsed=parsed,
                    question=question,
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                )
                if rej_a:
                    r_out = dict(base_row)
                    r_out["rejected_reasoning_trace"] = rej_a
                    if _needs_variant_column(modes):
                        r_out[VARIANT_COL] = "abliterated"
                    out_rows.append(r_out)
                else:
                    n_ab_fail += 1

        if "bbox_perturb" in modes:
            if not chosen:
                n_bbox_fail += 1
            else:
                rng = row_rng(args.seed, idx, img_key)
                rej_b = build_method2_bbox_perturb_rejected(
                    chosen,
                    rng,
                    bbox_zero_iou_fraction=args.bbox_zero_iou_fraction,
                )
                if rej_b:
                    r_out = dict(base_row)
                    r_out["rejected_reasoning_trace"] = rej_b
                    if _needs_variant_column(modes):
                        r_out[VARIANT_COL] = "bbox_perturb"
                    out_rows.append(r_out)
                else:
                    n_bbox_fail += 1

    fieldnames = list(CSV_FIELDNAMES)
    if _needs_variant_column(modes):
        fieldnames.append(VARIANT_COL)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    tmp.replace(out_path)
    fail_parts = []
    if "abliterated" in modes:
        fail_parts.append(f"abliterated_failures={n_ab_fail}")
    if "bbox_perturb" in modes:
        fail_parts.append(f"bbox_perturb_failures={n_bbox_fail}")
    extra = "; ".join(fail_parts) if fail_parts else ""
    print(f"Wrote {len(out_rows)} row(s) to {out_path}" + (f" ({extra})" if extra else "") + ".")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
