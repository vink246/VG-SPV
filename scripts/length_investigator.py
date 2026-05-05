#!/usr/bin/env python3
"""
Scan VG-SPV DPO CSV files and report max token counts for reasoning traces (text-only).

Uses the same CSV columns as ``train/dataset_adapter.py``:
  chosen_reasoning_trace, rejected_reasoning_trace

Tokenization is **text-only** (no images). Training adds vision placeholder tokens via the
processor/chat template, so real batches can exceed these counts — use this as a lower bound
on text capacity and size ``max_length`` with headroom.

Examples:

  python scripts/report_dpo_csv_token_lengths.py --model meta-llama/Llama-3.2-11B-Vision-Instruct \\
      --glob "data/**/traces/*dpo*.csv"

  python scripts/report_dpo_csv_token_lengths.py --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --roots data/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.dataset_adapter import (  # noqa: E402
    CHOSEN_REASONING_TRACE_COL,
    DEFAULT_PROMPT_INSTRUCTION,
    REJECTED_REASONING_TRACE_COL,
)

_STAT_KEYS = (
    "chosen_trace_only",
    "rejected_trace_only",
    "prompt_plus_chosen",
    "prompt_plus_rejected",
    "max_pair_row",
)


def _norm_header(s: str) -> str:
    return s.replace(" ", "_").strip().lower()


def _rows_from_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        rename = {}
        for h in reader.fieldnames:
            n = _norm_header(h)
            if n in (
                _norm_header(CHOSEN_REASONING_TRACE_COL),
                _norm_header(REJECTED_REASONING_TRACE_COL),
            ):
                target = (
                    CHOSEN_REASONING_TRACE_COL
                    if n == _norm_header(CHOSEN_REASONING_TRACE_COL)
                    else REJECTED_REASONING_TRACE_COL
                )
                rename[h] = target
        rows = []
        for raw in reader:
            row = {}
            for k, v in raw.items():
                nk = rename.get(k, k)
                row[nk] = v if v is not None else ""
            rows.append(row)
        return rows


def _load_tokenizer(model_name: str, use_processor: bool):
    if use_processor:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_name)
        return proc.tokenizer if hasattr(proc, "tokenizer") else proc
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def _encode_len(tok, text: str, *, add_special_tokens: bool = False) -> int:
    if not text:
        return 0
    return len(tok.encode(text, add_special_tokens=add_special_tokens))


def main() -> None:
    p = argparse.ArgumentParser(description="Max token lengths in DPO CSV reasoning traces (text-only).")
    p.add_argument("--model", type=str, required=True, help="HF model id or path for tokenizer/processor.")
    p.add_argument(
        "--use-processor",
        action="store_true",
        help="Load AutoProcessor and tokenize with processor.tokenizer (recommended for VLMs).",
    )
    p.add_argument(
        "--prompt-instruction",
        type=str,
        default=DEFAULT_PROMPT_INSTRUCTION,
        help="Same prompt string as training ``prompt_instruction`` / dataset_adapter default.",
    )
    p.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob pattern relative to repo root (repeatable). Default: common trace globs.",
    )
    p.add_argument(
        "--roots",
        nargs="*",
        default=[],
        help="Optional directories under repo root to recursively find *.csv (in addition to --glob).",
    )
    args = p.parse_args()

    globs = args.glob or [
        "data/**/traces/*dpo*.csv",
    ]

    paths_acc: set[Path] = set()
    for g in globs:
        paths_acc.update(_REPO_ROOT.glob(g))
    for root in args.roots:
        r = (_REPO_ROOT / root).resolve() if not Path(root).is_absolute() else Path(root)
        if r.is_dir():
            paths_acc.update(r.rglob("*.csv"))

    paths = sorted({x for x in paths_acc if x.is_file()}, key=lambda x: str(x))

    # Keep likely DPO CSVs: must mention dpo or traces path heuristic
    def looks_dpo(path: Path) -> bool:
        s = str(path).lower()
        return "dpo" in path.name.lower() or "/traces/" in s or "\\traces\\" in s

    paths = [x for x in paths if looks_dpo(x)]

    if not paths:
        print("No CSV files matched. Adjust --glob / --roots.", file=sys.stderr)
        sys.exit(1)

    tok = _load_tokenizer(args.model, args.use_processor)
    prompt = args.prompt_instruction

    global_max = {k: 0 for k in _STAT_KEYS}
    per_file: dict[str, dict[str, int]] = {}
    total_rows = 0

    for csv_path in paths:
        rows = _rows_from_csv(csv_path)
        fm = {k: 0 for k in _STAT_KEYS}
        for row in rows:
            total_rows += 1
            c = row.get(CHOSEN_REASONING_TRACE_COL) or ""
            rj = row.get(REJECTED_REASONING_TRACE_COL) or ""

            lc_t = _encode_len(tok, c)
            lr_t = _encode_len(tok, rj)
            lc_full = _encode_len(tok, f"{prompt}\n{c}")
            lr_full = _encode_len(tok, f"{prompt}\n{rj}")

            fm["chosen_trace_only"] = max(fm["chosen_trace_only"], lc_t)
            fm["rejected_trace_only"] = max(fm["rejected_trace_only"], lr_t)
            fm["prompt_plus_chosen"] = max(fm["prompt_plus_chosen"], lc_full)
            fm["prompt_plus_rejected"] = max(fm["prompt_plus_rejected"], lr_full)
            fm["max_pair_row"] = max(fm["max_pair_row"], max(lc_full, lr_full))

        try:
            key = str(csv_path.relative_to(_REPO_ROOT))
        except ValueError:
            key = str(csv_path)
        per_file[key] = dict(fm)

        global_max["chosen_trace_only"] = max(global_max["chosen_trace_only"], fm["chosen_trace_only"])
        global_max["rejected_trace_only"] = max(global_max["rejected_trace_only"], fm["rejected_trace_only"])
        global_max["prompt_plus_chosen"] = max(global_max["prompt_plus_chosen"], fm["prompt_plus_chosen"])
        global_max["prompt_plus_rejected"] = max(global_max["prompt_plus_rejected"], fm["prompt_plus_rejected"])
        global_max["max_pair_row"] = max(global_max["max_pair_row"], fm["max_pair_row"])

    print(f"Model / tokenizer: {args.model}  (processor={'yes' if args.use_processor else 'no'})")
    print(f"CSV files scanned: {len(paths)}  |  rows read: {total_rows}")
    print(f"Prompt instruction ({len(tok.encode(prompt, add_special_tokens=False))} tokens): {prompt[:80]!r}...")
    print()
    print("=== Per file (text-only token counts; add_special_tokens=False on trace chunks) ===")
    for name in sorted(per_file.keys()):
        m = per_file[name]
        print(f"\n{name}")
        for k in _STAT_KEYS:
            print(f"  {k}: {m[k]}")

    print("\n=== Global max across all listed CSVs ===")
    for k, v in global_max.items():
        print(f"  {k}: {v}")

    print(
        "\nNote: Chat-template / processor formatting in real DPO adds special tokens and image tokens; "
        "use max_length above tokenizer vocab estimates with margin."
    )


if __name__ == "__main__":
    main()
