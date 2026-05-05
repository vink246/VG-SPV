"""
MM-SafetyBench / VG-SPV CSV traces → bounding-box SFT chat samples (image + prompt + chosen trace).

``load_vgspv_csv_rows_for_sft`` loads rows with resolved ``image`` paths; ``vgspv_csv_row_to_bbox_sft_sample``
builds user/assistant messages for LoRA SFT (assistant = ``chosen_reasoning_trace``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from train.dataset_adapter import (
    CHOSEN_REASONING_TRACE_COL,
    DPO_PROMPT_COL,
    IMAGE_COL,
    PERTURBED_IMAGE_COL,
    REJECTED_REASONING_TRACE_COL,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class BoundingBoxSFTSample:
    image: Any  # PIL.Image
    messages: list[dict[str, Any]]


def _norm_csv_column_names(ds):
    """Match ``train/dataset_adapter.load_dpo_dataset`` CSV normalization."""

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
    Load VG-fDPO-style CSV rows (image + chosen_reasoning_trace + …).

    ``image`` paths are resolved with ``resolve_vgspv_image_path`` so repo-relative paths work when
    the job cwd is the repo root (override layout with ``image_root``).
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


def vgspv_csv_row_to_eval_user_messages(row: dict[str, Any], prompt_instruction: str) -> list[dict[str, Any]]:
    """
    User-only chat messages for Method-2 / VG-SPV CSV eval (image + ``prompt`` column or fallback instruction).

    Matches the user turn used in ``vgspv_csv_row_to_bbox_sft_sample`` without the assistant target.
    """
    sample = vgspv_csv_row_to_bbox_sft_sample(row, prompt_instruction)
    return [sample.messages[0]]


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
