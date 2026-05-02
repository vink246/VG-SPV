"""
HF datasets → bounding-box SFT rows (image, referring expression, one or more normalized xyxy boxes).

Supported hubs (see `data/download_bounding_box_sft_datasets.py`):
  - lmms-lab/RefCOCO (val/test splits; eval-oriented)
  - PaDT-MLLM/RefCOCO, RefCOCOPlus, RefCOCOg (typically include train; Shikra/VoCoT-style REC)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pathlib import Path

from PIL import Image

from train.bounding_box_sft_schema import build_assistant_bbox_sft_multi, user_text_with_expression
from train.dataset_adapter import (
    CHOSEN_REASONING_TRACE_COL,
    IMAGE_COL,
    PERTURBED_IMAGE_COL,
    REJECTED_REASONING_TRACE_COL,
)


@dataclass
class BoundingBoxSFTSample:
    image: Any  # PIL.Image
    messages: list[dict[str, Any]]


def _pil_size(img: Any) -> tuple[int, int]:
    if hasattr(img, "size"):
        w, h = img.size
        return int(w), int(h)
    raise TypeError(f"Expected PIL-like image, got {type(img)}")


def _to_xyxy_norm_pixels(xyxy_px: list[float], w: int, h: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(t) for t in xyxy_px[:4]]
    return (
        max(0.0, min(1.0, x0 / w)),
        max(0.0, min(1.0, y0 / h)),
        max(0.0, min(1.0, x1 / w)),
        max(0.0, min(1.0, y1 / h)),
    )


def _to_xyxy_norm_xywh(xywh_px: list[float], w: int, h: int) -> tuple[float, float, float, float]:
    x, y, bw, bh = [float(t) for t in xywh_px[:4]]
    x0, y0, x1, y1 = x, y, x + max(bw, 0), y + max(bh, 0)
    return _to_xyxy_norm_pixels([x0, y0, x1, y1], w, h)


def _norm_xyxy_one(b: list[float], w: int, h: int) -> tuple[float, float, float, float] | None:
    """Normalize one box field to [0,1] xyxy (already-xyxy vs COCO xywh in pixels)."""
    if len(b) < 4:
        return None
    b = [float(x) for x in b[:4]]
    if all(0.0 <= x <= 1.0 for x in b) and b[2] > b[0] and b[3] > b[1]:
        return b[0], b[1], b[2], b[3]
    return _to_xyxy_norm_xywh(b, w, h)


def infer_norm_boxes_from_row(row: dict[str, Any], image: Any) -> list[tuple[float, float, float, float]]:
    """
    All GT boxes for this row in [0,1] xyxy.

    - If ``bbox`` is a list of four-number boxes (some hubs), returns each normalized box.
    - If ``bboxes`` / ``boxes`` / ``gt_boxes`` / ``bbox_list`` holds a list of boxes, uses that.
    - Otherwise a single flat ``bbox`` (length 4) yields a one-element list.
    """
    w, h = _pil_size(image)
    for key in ("bboxes", "boxes", "gt_boxes", "bbox_list"):
        raw = row.get(key)
        if raw is None:
            continue
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if not isinstance(raw, (list, tuple)) or len(raw) == 0:
            continue
        el0 = raw[0]
        if isinstance(el0, (list, tuple)) and len(el0) >= 4:
            out: list[tuple[float, float, float, float]] = []
            for item in raw:
                if not isinstance(item, (list, tuple)) or len(item) < 4:
                    continue
                t = _norm_xyxy_one(list(item[:4]), w, h)
                if t is not None:
                    out.append(t)
            if out:
                return out
    raw = row.get("bbox")
    if raw is None:
        return []
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        return []
    if len(raw) >= 1:
        el0 = raw[0]
        if isinstance(el0, (list, tuple)) and len(el0) >= 4:
            out = []
            for item in raw:
                if not isinstance(item, (list, tuple)) or len(item) < 4:
                    continue
                t = _norm_xyxy_one(list(item[:4]), w, h)
                if t is not None:
                    out.append(t)
            return out
    if len(raw) >= 4:
        t = _norm_xyxy_one([float(x) for x in raw[:4]], w, h)
        return [t] if t is not None else []
    return []


def infer_norm_box_from_row(row: dict[str, Any], image: Any) -> tuple[float, float, float, float] | None:
    """First GT box only (backward compatible helper)."""
    boxes = infer_norm_boxes_from_row(row, image)
    return boxes[0] if boxes else None


def _get_question(row: dict[str, Any]) -> str:
    for key in ("question", "sentence", "expression", "text", "referring_expression", "prompt"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def get_row_pil_image(row: dict[str, Any]) -> Any:
    """Public helper for eval / tooling."""
    return _get_image(row)


def _get_image(row: dict[str, Any]) -> Any:
    img = row.get("image")
    if img is None:
        raise KeyError("Row missing 'image'")
    if isinstance(img, dict):
        from io import BytesIO

        if img.get("bytes"):
            return Image.open(BytesIO(img["bytes"])).convert("RGB")
        if img.get("path"):
            return Image.open(img["path"]).convert("RGB")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return img


def hf_row_to_bbox_sft_eval_user_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    """User-only messages for generation (same prompt contract as training)."""
    sample = hf_row_to_bbox_sft_sample(row)
    return [sample.messages[0]]


def hf_row_to_bbox_sft_sample(row: dict[str, Any]) -> BoundingBoxSFTSample:
    image = _get_image(row)
    phrase = _get_question(row)
    if not phrase:
        raise ValueError("Could not find referring expression in row")
    boxes = infer_norm_boxes_from_row(row, image)
    if not boxes:
        raise ValueError("Could not infer bounding box(es) from row")
    user_txt = user_text_with_expression(phrase)
    assistant = build_assistant_bbox_sft_multi(phrase, boxes)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_txt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant}],
        },
    ]
    return BoundingBoxSFTSample(image=image, messages=messages)


def load_bbox_sft_hf_dataset(
    dataset_id: str,
    split: str,
    *,
    max_samples: int | None = None,
    trust_remote_code: bool = True,
    config_name: str | None = None,
):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if config_name:
        kwargs["name"] = config_name
    ds = load_dataset(dataset_id, split=split, **kwargs)
    if max_samples is not None and max_samples > 0:
        n = min(max_samples, len(ds))
        ds = ds.select(range(n))
    return ds


def build_bbox_sft_sample_from_hf(ds, index: int) -> BoundingBoxSFTSample:
    return hf_row_to_bbox_sft_sample(ds[index])


def _norm_csv_column_names(ds):
    """Match `train/dataset_adapter.load_dpo_dataset` CSV normalization."""

    def norm(s: str) -> str:
        return s.replace(" ", "_").strip().lower()

    want = {
        norm(IMAGE_COL): IMAGE_COL,
        norm(PERTURBED_IMAGE_COL): PERTURBED_IMAGE_COL,
        norm(CHOSEN_REASONING_TRACE_COL): CHOSEN_REASONING_TRACE_COL,
        norm(REJECTED_REASONING_TRACE_COL): REJECTED_REASONING_TRACE_COL,
    }
    renames = {}
    for c in list(ds.column_names):
        n = norm(c)
        if n in want and c != want[n]:
            renames[c] = want[n]
    for old, new in renames.items():
        ds = ds.rename_column(old, new)
    return ds


def load_vgspv_csv_rows_for_sft(csv_path: str) -> list[dict[str, Any]]:
    """
    Load VG-fDPO-style CSV rows (same schema as DPO: image + chosen_reasoning_trace + …).

    Used to mix safety / risky-object supervision into bounding-box SFT (chosen trace as target).
    """
    from datasets import load_dataset

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"VG-SPV CSV not found: {path}")
    ds = load_dataset("csv", data_files=str(path), split="train")
    ds = _norm_csv_column_names(ds)
    if IMAGE_COL not in ds.column_names:
        raise ValueError(f"CSV must have column {IMAGE_COL}. Got: {ds.column_names}")
    if CHOSEN_REASONING_TRACE_COL not in ds.column_names:
        raise ValueError(
            f"CSV must have {CHOSEN_REASONING_TRACE_COL} for SFT targets. Got: {ds.column_names}"
        )
    return [ds[i] for i in range(len(ds))]


def vgspv_csv_row_to_bbox_sft_sample(row: dict[str, Any], prompt_instruction: str) -> BoundingBoxSFTSample:
    """One SFT example: user image + instruction, assistant = chosen_reasoning_trace (teacher text)."""
    img_path = row.get(IMAGE_COL)
    if not isinstance(img_path, str) or not img_path.strip():
        raise ValueError(f"Row missing {IMAGE_COL}")
    path = Path(img_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image path not found: {path}")
    chosen = row.get(CHOSEN_REASONING_TRACE_COL)
    if not isinstance(chosen, str) or not chosen.strip():
        raise ValueError(f"Row missing {CHOSEN_REASONING_TRACE_COL}")
    image = Image.open(path).convert("RGB")
    prompt = prompt_instruction.strip()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": chosen.strip()}],
        },
    ]
    return BoundingBoxSFTSample(image=image, messages=messages)
