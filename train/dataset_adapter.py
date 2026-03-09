"""
DPO dataset contract and adapter for VG-SPV.

Dataset format (CSV):
  - image: path to the input image
  - perturbed_image: path to the perturbed/jailbreak image (for VG-fDPO / visual grounding)
  - chosen_reasoning_trace: preferred reasoning trace (DPO "chosen")
  - rejected_reasoning_trace: dispreferred reasoning trace (DPO "rejected")

Standard DPO uses (image, chosen, rejected). The VG-SPV visual-grounding pipeline
additionally uses perturbed_image for VG-fDPO loss. Images are stored as paths in the CSV.
"""

from pathlib import Path
from typing import Any

from datasets import Dataset

# --- CSV / source columns (stored in CSV) ---
IMAGE_COL = "image"
PERTURBED_IMAGE_COL = "perturbed_image"
CHOSEN_REASONING_TRACE_COL = "chosen_reasoning_trace"
REJECTED_REASONING_TRACE_COL = "rejected_reasoning_trace"

# --- DPO contract: column names expected by DPOTrainer / our pipeline ---
# Standard DPO: prompt + chosen + rejected (optionally image per row).
# VG-fDPO: same plus perturbed_image for VG-fDPO (Visually-Grounded Fine-Grained DPO) loss.
DPO_PROMPT_COL = "prompt"
DPO_CHOSEN_COL = "chosen"
DPO_REJECTED_COL = "rejected"
DPO_IMAGE_COL = "image"
DPO_PERTURBED_IMAGE_COL = "perturbed_image"

# Default instruction used as prompt when building from CSV (image path is passed separately).
DEFAULT_PROMPT_INSTRUCTION = "Is there anything unsafe in this image? Provide your reasoning."


def csv_to_dpo_format(
    rows: list[dict[str, Any]],
    prompt_instruction: str = DEFAULT_PROMPT_INSTRUCTION,
) -> dict[str, list]:
    """
    Map CSV-style rows (image, perturbed_image, chosen_reasoning_trace, rejected_reasoning_trace)
    to DPO columns (prompt, chosen, rejected, image, perturbed_image).
    """
    out = {
        DPO_PROMPT_COL: [],
        DPO_CHOSEN_COL: [],
        DPO_REJECTED_COL: [],
        DPO_IMAGE_COL: [],
        DPO_PERTURBED_IMAGE_COL: [],
    }
    for row in rows:
        out[DPO_PROMPT_COL].append(prompt_instruction)
        out[DPO_CHOSEN_COL].append(row[CHOSEN_REASONING_TRACE_COL])
        out[DPO_REJECTED_COL].append(row[REJECTED_REASONING_TRACE_COL])
        out[DPO_IMAGE_COL].append(row[IMAGE_COL])
        out[DPO_PERTURBED_IMAGE_COL].append(row.get(PERTURBED_IMAGE_COL, ""))
    return out


def get_example_dpo_dataset() -> Dataset:
    """
    Minimal in-memory DPO dataset for pipeline testing.
    Uses the standard CSV schema (image, perturbed_image, chosen_reasoning_trace, rejected_reasoning_trace).
    """
    example = [
        {
            IMAGE_COL: "image1.png",
            PERTURBED_IMAGE_COL: "image1_perturbed.png",
            CHOSEN_REASONING_TRACE_COL: "Yes, there is a weapon visible. I will not describe it further.",
            REJECTED_REASONING_TRACE_COL: "I see a harmless scene with no safety concerns.",
        },
        {
            IMAGE_COL: "image2.png",
            PERTURBED_IMAGE_COL: "image2_perturbed.png",
            CHOSEN_REASONING_TRACE_COL: "I see a cat sitting on a couch.",
            REJECTED_REASONING_TRACE_COL: "I cannot answer that.",
        },
    ]
    mapped = csv_to_dpo_format(example)
    return Dataset.from_dict(mapped)


def load_dpo_dataset(
    path_or_name: str | None = None,
    prompt_instruction: str = DEFAULT_PROMPT_INSTRUCTION,
) -> Dataset:
    """
    Load a dataset and convert to DPO contract.

    - If path_or_name is None: returns the minimal example dataset.
    - If path_or_name is a .csv file: loads CSV with columns image, perturbed_image,
      chosen_reasoning_trace, rejected_reasoning_trace and maps to DPO columns.
    - If path_or_name is a directory: loads from disk (e.g. saved Dataset).
    - Otherwise: treats as HuggingFace dataset name and loads "train" split.

    Resulting dataset has: prompt, chosen, rejected, image, perturbed_image.
    """
    if path_or_name is None:
        return get_example_dpo_dataset()

    path = Path(path_or_name)
    if path.suffix.lower() == ".csv":
        from datasets import load_dataset
        ds = load_dataset("csv", data_files=str(path), split="train")
        # Normalize column names: "perturbed image" -> perturbed_image, etc.
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
        if CHOSEN_REASONING_TRACE_COL not in ds.column_names or REJECTED_REASONING_TRACE_COL not in ds.column_names:
            raise ValueError(
                f"CSV must have columns: {IMAGE_COL}, {PERTURBED_IMAGE_COL}, {CHOSEN_REASONING_TRACE_COL}, {REJECTED_REASONING_TRACE_COL}. Got: {ds.column_names}"
            )
        if IMAGE_COL not in ds.column_names:
            raise ValueError(f"CSV must have column {IMAGE_COL}. Got: {ds.column_names}")
        if PERTURBED_IMAGE_COL not in ds.column_names:
            # Allow missing perturbed_image: fill with empty string
            ds = ds.add_column(PERTURBED_IMAGE_COL, [""] * len(ds))
        rows = [ds[i] for i in range(len(ds))]
        mapped = csv_to_dpo_format(rows, prompt_instruction=prompt_instruction)
        return Dataset.from_dict(mapped)

    if path.is_dir():
        from datasets import load_from_disk
        ds = load_from_disk(path_or_name)
    else:
        from datasets import load_dataset
        ds = load_dataset(path_or_name, split="train")

    # Already in DPO contract?
    if DPO_PROMPT_COL in ds.column_names and DPO_CHOSEN_COL in ds.column_names:
        return ds
    # Has our CSV-style columns?
    if IMAGE_COL in ds.column_names and CHOSEN_REASONING_TRACE_COL in ds.column_names:
        rows = [ds[i] for i in range(len(ds))]
        if PERTURBED_IMAGE_COL not in ds.column_names:
            for r in rows:
                r[PERTURBED_IMAGE_COL] = ""
        mapped = csv_to_dpo_format(rows, prompt_instruction=prompt_instruction)
        return Dataset.from_dict(mapped)
    raise ValueError(
        f"Dataset must have DPO columns ({DPO_PROMPT_COL}, {DPO_CHOSEN_COL}, {DPO_REJECTED_COL}) "
        f"or CSV columns ({IMAGE_COL}, {CHOSEN_REASONING_TRACE_COL}, {REJECTED_REASONING_TRACE_COL}). Got: {ds.column_names}"
    )
