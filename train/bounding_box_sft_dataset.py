"""
HF datasets → bounding-box SFT rows (image, referring expression, one or more normalized xyxy boxes).

Supported hubs (see ``scripts/download_bounding_box_sft_datasets.py``):
  - lmms-lab/RefCOCO (val/test splits; eval-oriented)
  - PaDT-MLLM/RefCOCO (single hub; default config bundles refcoco / refcoco+ / refcocog JSON splits).
    Rows store a COCO JPEG basename (often ``COCO_train2014_*.jpg``). Resolve under ``data/coco/`` using
    MSCOCO split folders (``train2014/`` then ``train2017/`` for train basenames, etc.). Override with
    ``BBOX_SFT_IMAGE_ROOT`` or ``--bbox_hf_image_root`` / ``load_bbox_sft_hf_datasets(..., image_root=...)``.

Local directories created with ``Dataset.save_to_disk`` / ``DatasetDict.save_to_disk`` can be passed
instead of a hub id (PACE scratch mirrors of HF caches).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from pathlib import Path

from PIL import Image

from train.bounding_box_sft_schema import build_assistant_bbox_sft_multi, user_text_with_expression
from train.dataset_adapter import (
    CHOSEN_REASONING_TRACE_COL,
    DPO_PROMPT_COL,
    IMAGE_COL,
    PERTURBED_IMAGE_COL,
    REJECTED_REASONING_TRACE_COL,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Default layout: repo ``data/coco/{train2014,train2017,...}/<basename>`` (see ``_coco_candidate_subdirs``).
DEFAULT_BBOX_COCO_ROOT = _REPO_ROOT / "data" / "coco"

# Set on each ``load_bbox_sft_hf_datasets`` / ``load_bbox_sft_hf_dataset`` call (CLI ``--bbox_hf_image_root``).
_ACTIVE_BBOX_COCO_ROOT: Path | None = None


def reset_bbox_hf_coco_root_for_tests() -> None:
    """Reset COCO root override (unit tests only)."""
    global _ACTIVE_BBOX_COCO_ROOT
    _ACTIVE_BBOX_COCO_ROOT = None


def set_bbox_hf_coco_root_for_tests(root: Path | str | None) -> None:
    """Force COCO root for tests that call ``hf_row_to_bbox_sft_sample`` without loading a HF dataset."""
    global _ACTIVE_BBOX_COCO_ROOT
    if root is None:
        _ACTIVE_BBOX_COCO_ROOT = None
        return
    p = Path(root).expanduser().resolve()
    _ACTIVE_BBOX_COCO_ROOT = p if p.is_dir() else None


def _refresh_active_bbox_coco_root(image_root: str | Path | None) -> None:
    """Pick COCO root: explicit arg > BBOX_SFT_IMAGE_ROOT > data/coco if present."""
    global _ACTIVE_BBOX_COCO_ROOT
    if image_root is not None:
        p = Path(image_root).expanduser().resolve()
        _ACTIVE_BBOX_COCO_ROOT = p if p.is_dir() else None
        return
    env = (os.environ.get("BBOX_SFT_IMAGE_ROOT") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        _ACTIVE_BBOX_COCO_ROOT = p if p.is_dir() else None
        return
    _ACTIVE_BBOX_COCO_ROOT = DEFAULT_BBOX_COCO_ROOT.resolve() if DEFAULT_BBOX_COCO_ROOT.is_dir() else None


def _coco_roots_for_resolve() -> list[Path]:
    if _ACTIVE_BBOX_COCO_ROOT is not None and _ACTIVE_BBOX_COCO_ROOT.is_dir():
        return [_ACTIVE_BBOX_COCO_ROOT]
    env = (os.environ.get("BBOX_SFT_IMAGE_ROOT") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return [p]
    if DEFAULT_BBOX_COCO_ROOT.is_dir():
        return [DEFAULT_BBOX_COCO_ROOT.resolve()]
    return []


def _coco_candidate_subdirs(basename: str) -> list[str]:
    """
    MSCOCO subfolders to try under the COCO root, in order.

    PaDT train rows typically use **2014-style basenames** (``COCO_train2014_*.jpg``), which match the
    official **COCO 2014 train** image release (``train2014/``), not ``COCO_train2017_*.jpg`` from the
    2017 zip. We try ``train2014`` first, then ``train2017`` so a standard 2014 unzip "just works".
    """
    lower = basename.lower()
    if lower.startswith("coco_train2014_"):
        return ["train2014", "train2017"]
    if lower.startswith("coco_train2017_"):
        return ["train2017"]
    if lower.startswith("coco_val2014_"):
        return ["val2014", "val2017"]
    if lower.startswith("coco_val2017_"):
        return ["val2017"]
    if lower.startswith("coco_test2017_"):
        return ["test2017"]
    if lower.startswith("coco_test2014_"):
        return ["test2014", "test2017"]
    return []


def _resolve_image_path_string(s: str) -> Path | None:
    """Map a row ``image`` string to an existing file, including COCO basenames under the active root."""
    path = Path(s).expanduser()
    if path.is_file():
        return path.resolve()
    name = path.name
    subdirs = _coco_candidate_subdirs(name)
    for root in _coco_roots_for_resolve():
        for sub in subdirs:
            cand = root / sub / name
            if cand.is_file():
                return cand.resolve()
        cand = root / name
        if cand.is_file():
            return cand.resolve()
    return None


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
    - PaDT-style ``objects``: a **list** of ``{bbox, label, …}``, or a **dict** with a ``value`` / ``instances``
      list of the same shape.
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
    if raw is not None:
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if isinstance(raw, (list, tuple)):
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
                    if out:
                        return out
            if len(raw) >= 4:
                t = _norm_xyxy_one([float(x) for x in raw[:4]], w, h)
                if t is not None:
                    return [t]
    boxes_obj = _infer_norm_boxes_from_objects(row, w, h)
    if boxes_obj:
        return boxes_obj
    return []


def _boxes_from_object_items(
    items: list[Any], w: int, h: int
) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        bb = item.get("bbox")
        if not isinstance(bb, (list, tuple)) or len(bb) < 4:
            continue
        t = _norm_xyxy_one([float(x) for x in bb[:4]], w, h)
        if t is not None:
            out.append(t)
    return out


def _infer_norm_boxes_from_objects(
    row: dict[str, Any], w: int, h: int
) -> list[tuple[float, float, float, float]]:
    """PaDT-style ``objects``: list of ``{bbox, label, ...}``, or dict with ``value`` / ``instances``."""
    o = row.get("objects")
    if o is None:
        return []
    if hasattr(o, "tolist"):
        o = o.tolist()
    if isinstance(o, list):
        out = _boxes_from_object_items(o, w, h)
        if out:
            return out
        return []
    if isinstance(o, dict):
        items = o.get("value") or o.get("objects") or o.get("instances")
        if isinstance(items, list):
            out = _boxes_from_object_items(items, w, h)
            if out:
                return out
        bb = o.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) >= 4:
            t = _norm_xyxy_one([float(x) for x in bb[:4]], w, h)
            return [t] if t is not None else []
    return []


def infer_norm_box_from_row(row: dict[str, Any], image: Any) -> tuple[float, float, float, float] | None:
    """First GT box only (backward compatible helper)."""
    boxes = infer_norm_boxes_from_row(row, image)
    return boxes[0] if boxes else None


_RE_DESC_QUOTED = re.compile(
    r"""this\s+sentence\s+describes\s*:\s*["']([^"']+)["']""",
    re.IGNORECASE | re.DOTALL,
)


def _scalar_to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace").strip()
    if hasattr(v, "item") and callable(getattr(v, "item")):
        try:
            return str(v.item()).strip()
        except Exception:
            pass
    return str(v).strip()


def _normalize_conversation_list(raw: Any) -> list[dict[str, Any]] | None:
    if raw is None:
        return None
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") or s.startswith("{"):
            try:
                raw = json.loads(s)
            except json.JSONDecodeError:
                return None
        else:
            return None
    if not isinstance(raw, (list, tuple)):
        return None
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
    return out or None


def _phrase_from_conversation_text(text: str) -> str:
    if not text:
        return ""
    m = _RE_DESC_QUOTED.search(text)
    if m:
        return m.group(1).strip()
    return ""


def _phrase_from_conversation_lists(row: dict[str, Any]) -> str:
    for key in ("conversations", "messages", "dialog", "instruction", "instruction_input"):
        conv = _normalize_conversation_list(row.get(key))
        if not conv:
            continue
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            role = (turn.get("from") or turn.get("role") or "").strip().lower()
            if role in ("gpt", "assistant", "model"):
                continue
            val = turn.get("value") or turn.get("content") or turn.get("text")
            if isinstance(val, list):
                parts = []
                for block in val:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(_scalar_to_str(block.get("text")))
                blob = "\n".join(p for p in parts if p)
            else:
                blob = _scalar_to_str(val)
            hit = _phrase_from_conversation_text(blob)
            if hit:
                return hit
    return ""


def _get_question(row: dict[str, Any]) -> str:
    keys = (
        "question",
        "sentence",
        "expression",
        "text",
        "referring_expression",
        "prompt",
        "utterance",
        "utter",
        "sent",
        "refExp",
        "ref_exp",
        "query",
        "noun_phrase",
        "description",
        "caption",
        "instruction",
        "refer_query",
    )
    for key in keys:
        v = row.get(key)
        if v is None:
            continue
        s = _scalar_to_str(v)
        if s:
            return s
    conv_phrase = _phrase_from_conversation_lists(row)
    if conv_phrase:
        return conv_phrase
    return ""


def get_row_pil_image(row: dict[str, Any]) -> Any:
    """Public helper for eval / tooling."""
    return _get_image(row)


def _get_image(row: dict[str, Any]) -> Any:
    from io import BytesIO
    from urllib.error import URLError
    from urllib.request import Request, urlopen

    img = row.get("image")
    if img is None:
        raise KeyError("Row missing 'image'")
    if isinstance(img, dict):
        if img.get("bytes"):
            return Image.open(BytesIO(img["bytes"])).convert("RGB")
        if img.get("path"):
            return Image.open(img["path"]).convert("RGB")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, (bytes, bytearray)):
        return Image.open(BytesIO(bytes(img))).convert("RGB")
    if isinstance(img, str):
        s = img.strip()
        if not s:
            raise ValueError("Row has empty image path")
        if s.startswith(("http://", "https://")):
            req = Request(s, headers={"User-Agent": "VG-SPV-bbox-sft/1.0"})
            try:
                with urlopen(req, timeout=120) as r:
                    return Image.open(BytesIO(r.read())).convert("RGB")
            except URLError as e:
                raise ValueError(f"Could not download image URL: {s!r}") from e
        resolved = _resolve_image_path_string(s)
        if resolved is not None:
            return Image.open(resolved).convert("RGB")
        try:
            return Image.open(s).convert("RGB")
        except OSError as e:
            hint = ""
            if _coco_candidate_subdirs(Path(s.strip()).name):
                hint = (
                    f" Expected JPEG under {DEFAULT_BBOX_COCO_ROOT}/train2014/ or .../train2017/ (see README). "
                    "Or set BBOX_SFT_IMAGE_ROOT / --bbox_hf_image_root / image_root=."
                )
            raise ValueError(f"Could not open image path: {s!r}.{hint}") from e
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


def _load_one_bbox_sft_source(
    dataset_id_or_path: str,
    split: str,
    *,
    trust_remote_code: bool = False,
    config_name: str | None = None,
    local_files_only: bool = False,
):
    """Load a single HF hub dataset or a ``save_to_disk`` directory (Dataset or DatasetDict)."""
    from datasets import DatasetDict, load_dataset, load_from_disk

    p = Path(dataset_id_or_path).expanduser()
    if p.is_dir():
        try:
            d = load_from_disk(str(p))
        except Exception as e:
            raise RuntimeError(
                f"Could not load_dataset from local path {p}. "
                "Expected a directory produced by Dataset.save_to_disk / DatasetDict.save_to_disk."
            ) from e
        if isinstance(d, DatasetDict):
            if split not in d:
                raise ValueError(f"Split {split!r} not found in disk dataset. Available: {list(d.keys())}")
            return d[split]
        return d

    kwargs: dict[str, Any] = {}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if config_name:
        kwargs["name"] = config_name
    if local_files_only:
        # PaDT JSON-style hubs reject top-level ``local_files_only`` (it is mistaken for a
        # builder config field). Use ``DownloadConfig`` so only the downloader stays local.
        from datasets import DownloadConfig

        kwargs["download_config"] = DownloadConfig(local_files_only=True)
    return load_dataset(dataset_id_or_path, split=split, **kwargs)


def load_bbox_sft_hf_datasets(
    dataset_sources: list[str],
    split: str,
    *,
    max_samples: int | None = None,
    trust_remote_code: bool = False,
    config_name: str | None = None,
    local_files_only: bool = False,
    image_root: str | Path | None = None,
):
    """
    Load one or more REC-style datasets and concatenate (same split key for each hub source).

    All parts must be row-compatible with ``hf_row_to_bbox_sft_sample`` (image + expression + bbox).

    ``image_root``: optional COCO root (directory containing ``train2014/``, ``train2017/``, …). Overrides
    ``BBOX_SFT_IMAGE_ROOT`` and the default ``data/coco`` for this process until the next load.
    """
    from datasets import concatenate_datasets

    if not dataset_sources:
        raise ValueError("dataset_sources must be non-empty")
    _refresh_active_bbox_coco_root(image_root)
    parts: list[Any] = []
    for src in dataset_sources:
        try:
            parts.append(
                _load_one_bbox_sft_source(
                    src,
                    split,
                    trust_remote_code=trust_remote_code,
                    config_name=config_name,
                    local_files_only=local_files_only,
                )
            )
        except Exception as e:
            hint = ""
            if "RefCOCO" in src or "PaDT" in src:
                hint = (
                    " PaDT’s public `PaDT-MLLM/RefCOCO` hub usually bundles refcoco, refcoco+, and refcocog "
                    "in one dataset (multiple JSON files under the default builder). "
                    "There is often no separate `PaDT-MLLM/RefCOCOPlus` / `RefCOCOg` hub — use only "
                    "`--dataset_id PaDT-MLLM/RefCOCO` and omit those `--extra_dataset` flags. "
                    "Offline: set `HF_HOME` / `HF_DATASETS_CACHE` to your scratch cache and keep the hub id."
                )
            raise RuntimeError(
                f"Failed to load HF dataset source {src!r} (split={split!r}).{hint} Original error: {e}"
            ) from e
    if len(parts) == 1:
        ds = parts[0]
    else:
        try:
            ds = concatenate_datasets(parts)
        except Exception as e:
            raise RuntimeError(
                "concatenate_datasets failed — sources may use incompatible column/feature schemas. "
                "Train on one source at a time or preprocess to a common schema."
            ) from e
    if max_samples is not None and max_samples > 0:
        n = min(max_samples, len(ds))
        ds = ds.select(range(n))
    return ds


def load_bbox_sft_hf_dataset(
    dataset_id: str,
    split: str,
    *,
    max_samples: int | None = None,
    trust_remote_code: bool = False,
    config_name: str | None = None,
    local_files_only: bool = False,
    image_root: str | Path | None = None,
):
    return load_bbox_sft_hf_datasets(
        [dataset_id],
        split,
        max_samples=max_samples,
        trust_remote_code=trust_remote_code,
        config_name=config_name,
        local_files_only=local_files_only,
        image_root=image_root,
    )


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
        norm(DPO_PROMPT_COL): DPO_PROMPT_COL,
    }
    renames = {}
    for c in list(ds.column_names):
        n = norm(c)
        if n in want and c != want[n]:
            renames[c] = want[n]
    for old, new in renames.items():
        ds = ds.rename_column(old, new)
    return ds


def resolve_vgspv_image_path(
    image_field: str,
    *,
    image_root: Path | None = None,
) -> Path:
    """Resolve CSV ``image`` paths (often repo-relative) to an existing file."""
    raw = image_field.strip()
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    candidates: list[Path] = []
    if image_root is not None:
        candidates.append((image_root / raw).resolve())
    candidates.append((_REPO_ROOT / raw).resolve())
    candidates.append((Path.cwd() / raw).resolve())
    for c in candidates:
        if c.is_file():
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Image not found for {raw!r}. Tried:\n  {tried}")


def load_vgspv_csv_rows_for_sft(
    csv_path: str,
    *,
    image_root: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Load VG-fDPO-style CSV rows (same schema as DPO: image + chosen_reasoning_trace + …).

    Used to mix safety / risky-object supervision into bounding-box SFT (chosen trace as target).
    ``image`` paths are resolved with ``resolve_vgspv_image_path`` so repo-relative paths work on
    PACE when the job cwd is the repo root (override layout with ``image_root``).
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
    rows: list[dict[str, Any]] = []
    for i in range(len(ds)):
        row = dict(ds[i])
        img = row.get(IMAGE_COL)
        if isinstance(img, str) and img.strip():
            resolved = resolve_vgspv_image_path(img, image_root=image_root)
            row[IMAGE_COL] = str(resolved)
        rows.append(row)
    return rows


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
    per_row = row.get(DPO_PROMPT_COL)
    if isinstance(per_row, str) and per_row.strip():
        prompt = per_row.strip()
    else:
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
