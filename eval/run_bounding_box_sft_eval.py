"""
Evaluate bounding-box SFT: IoU between predicted boxes (parsed from `<risk_factors_with_boxes>`) and GT.

Supports multiple GT boxes per row (list in ``bbox`` or columns ``bboxes`` / ``boxes`` / ``gt_boxes``).
Metrics: mean of (per-GT max IoU to any prediction), Acc@0.5 on that mean, and strict all-GT-covered@0.5.

Example:
  python eval/run_bounding_box_sft_eval.py \\
    --model_name llava-hf/llava-1.6-mistral-7b-hf \\
    --lora_adapter outputs/bbox_sft_llava/adapter \\
    --dataset_id lmms-lab/RefCOCO --split val --max_samples 200
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

from train.bounding_box_sft_dataset import (
    get_row_pil_image,
    hf_row_to_bbox_sft_eval_user_messages,
    infer_norm_boxes_from_row,
    load_bbox_sft_hf_dataset,
)
from train.tag_parsing import (
    all_gts_matched_at_iou,
    mean_max_iou_per_gt,
    parse_all_norm_boxes,
)
from vlm import load_vlm_with_optional_lora, run_vl_inference


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bounding-box SFT grounding IoU evaluation.")
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument(
        "--lora_adapter",
        type=str,
        default=None,
        help="PEFT adapter dir from train/run_bounding_box_sft.py (adapter/ or adapter_latest/).",
    )
    p.add_argument("--merge_adapter", action="store_true", help="Merge LoRA into base weights before eval.")
    p.add_argument("--dataset_id", type=str, default="lmms-lab/RefCOCO")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--max_samples", type=int, default=500)
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
        raise SystemExit("TinyLLaVA: extend eval to use TinyLLaVA chat path if needed.")

    ds = load_bbox_sft_hf_dataset(
        args.dataset_id,
        args.split,
        max_samples=args.max_samples,
        config_name=args.dataset_config,
    )

    ious: list[float] = []
    parsed_any: list[bool] = []
    all_gt_hit: list[bool] = []
    failures = 0
    gt_counts: list[int] = []
    pred_counts: list[int] = []

    for i in range(len(ds)):
        row = ds[i]
        try:
            pil = get_row_pil_image(row)
            gt_boxes = infer_norm_boxes_from_row(row, pil)
        except Exception:
            failures += 1
            continue
        if not gt_boxes:
            failures += 1
            continue
        gt_counts.append(len(gt_boxes))
        msgs = hf_row_to_bbox_sft_eval_user_messages(row)
        try:
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
        "dataset_id": args.dataset_id,
        "split": args.split,
    }
    print(json.dumps(report, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
