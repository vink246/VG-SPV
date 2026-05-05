"""
IoU evaluation for Method-2 / MM-SafetyBench CSV traces (teacher vs model boxes).

Loads ``image``, ``prompt``, and ``chosen_reasoning_trace``; treats boxes parsed from the teacher
``chosen_reasoning_trace`` as GT; runs the VLM with the CSV ``prompt``; parses predicted
``<risk_factors_with_boxes>`` boxes and reports mean/max IoU and coverage metrics.

Example:
  python eval/run_method2_csv_bbox_eval.py \\
    --csv data/mm-safebench_1/extracted_data/traces/test_method2.csv \\
    --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \\
    --lora_adapter outputs/bbox_sft_llama/adapter \\
    --bf16 --max_samples 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from train.bounding_box_sft_dataset import load_vgspv_csv_rows_for_sft, vgspv_csv_row_to_eval_user_messages
from train.dataset_adapter import CHOSEN_REASONING_TRACE_COL, DEFAULT_PROMPT_INSTRUCTION
from train.tag_parsing import all_gts_matched_at_iou, mean_max_iou_per_gt, parse_all_norm_boxes
from vlm import load_vlm_with_optional_lora, run_vl_inference


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Method-2 CSV bounding-box IoU evaluation (teacher vs model).")
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to Method-2 CSV (image, prompt, chosen_reasoning_trace, …).",
    )
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument(
        "--lora_adapter",
        type=str,
        default=None,
        help="PEFT adapter dir (adapter/ or adapter_latest/).",
    )
    p.add_argument("--merge_adapter", action="store_true", help="Merge LoRA into base before eval.")
    p.add_argument(
        "--vgspv_image_root",
        type=str,
        default=None,
        help="Optional base dir for resolving relative image paths in the CSV.",
    )
    p.add_argument(
        "--prompt_instruction",
        type=str,
        default=None,
        help="Fallback user text when a row has no prompt column (default: dataset_adapter.DEFAULT_PROMPT_INSTRUCTION).",
    )
    p.add_argument("--max_samples", type=int, default=None, help="Cap number of CSV rows (after load).")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument(
        "--model_family",
        type=str,
        default=None,
        choices=["qwen3_vl", "llava", "tinyllava", "mllama"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    loaded = load_vlm_with_optional_lora(
        args.model_name,
        lora_adapter_path=args.lora_adapter,
        dtype=dtype,
        merge_adapter=args.merge_adapter,
        model_family=args.model_family,
        is_trainable=False,
    )
    if loaded.family == "tinyllava":
        raise SystemExit("TinyLLaVA: extend eval chat path if needed.")

    img_root = Path(args.vgspv_image_root).resolve() if args.vgspv_image_root else None
    rows = load_vgspv_csv_rows_for_sft(str(csv_path), image_root=img_root)
    if args.max_samples is not None and int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]

    prompt_fallback = (args.prompt_instruction or DEFAULT_PROMPT_INSTRUCTION).strip()

    ious: list[float] = []
    parsed_any: list[bool] = []
    all_gt_hit: list[bool] = []
    failures = 0
    gt_counts: list[int] = []
    pred_counts: list[int] = []

    for row in rows:
        chosen = row.get(CHOSEN_REASONING_TRACE_COL)
        if not isinstance(chosen, str) or not chosen.strip():
            failures += 1
            continue
        gt_boxes = parse_all_norm_boxes(chosen)
        if not gt_boxes:
            failures += 1
            continue
        gt_counts.append(len(gt_boxes))
        try:
            msgs = vgspv_csv_row_to_eval_user_messages(row, prompt_fallback)
            text = run_vl_inference(loaded, msgs, max_new_tokens=args.max_new_tokens, do_sample=False)
        except Exception:
            failures += 1
            continue
        pred_boxes = parse_all_norm_boxes(text)
        pred_counts.append(len(pred_boxes))
        parsed_any.append(len(pred_boxes) > 0)
        score = mean_max_iou_per_gt(gt_boxes, pred_boxes)
        ious.append(score)
        all_gt_hit.append(all_gts_matched_at_iou(gt_boxes, pred_boxes, threshold=0.5))

    n = len(ious)
    acc50 = sum(1 for x in ious if x >= 0.5) / n if n else 0.0
    mean_iou = sum(ious) / n if n else 0.0
    acc_all_gt50 = sum(1 for x in all_gt_hit if x) / n if n else 0.0
    mean_num_gt = sum(gt_counts) / len(gt_counts) if gt_counts else 0.0
    mean_num_pred = sum(pred_counts) / len(pred_counts) if pred_counts else 0.0
    report = {
        "eval_kind": "method2_csv_teacher_boxes_vs_model",
        "num_evaluated": n,
        "mean_iou_mean_max_per_gt": mean_iou,
        "acc_mean_max_iou_0.5": acc50,
        "acc_all_gt_boxes_iou_0.5": acc_all_gt50,
        "num_with_any_parsed_pred_box": sum(1 for x in parsed_any if x),
        "failures_skipped": failures,
        "mean_gt_boxes_per_sample": mean_num_gt,
        "mean_pred_boxes_per_sample": mean_num_pred,
        "model_name": args.model_name,
        "lora_adapter": args.lora_adapter,
        "csv": str(csv_path),
        "max_samples": args.max_samples,
    }
    print(json.dumps(report, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
