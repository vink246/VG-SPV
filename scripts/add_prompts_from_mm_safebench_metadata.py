"""
Join VG-SPV train/test CSV rows with prompts from mm-safebench_1 metadata.

For each row, the script matches the row's `image` path to the `image` column in
`train_metadata.csv` (for train_method*.csv) or `test_metadata.csv` (for
test_method*.csv), then copies the corresponding `question` field into a new
`prompt` column (inserted immediately after `image`).

Paths are matched flexibly: exact string, POSIX-style slashes, paths resolved
relative to the repo root, and basename fallback within the chosen metadata file.

Default layout (paths relative to repo root):

  data/mm-safebench_1/extracted_data/train_metadata.csv
  data/mm-safebench_1/extracted_data/test_metadata.csv
  data/mm-safebench_1/extracted_data/traces/train_method{1,2}.csv
  data/mm-safebench_1/extracted_data/traces/test_method{1,2}.csv

Relative CLI arguments (--metadata-dir, --traces-dir, --inputs) are resolved
against --repo-root so jobs on PACE ICE work regardless of current working directory.

Usage (from repo root; do not run unless you intend to rewrite files):

  python scripts/add_prompts_from_mm_safebench_metadata.py --dry-run
  python scripts/add_prompts_from_mm_safebench_metadata.py --in-place

  python scripts/add_prompts_from_mm_safebench_metadata.py \\
      --metadata-dir data/mm-safebench_1/extracted_data \\
      --traces-dir data/mm-safebench_1/extracted_data/traces
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Relative to repo root (portable on PACE ICE and local checkouts).
_METADATA_DIR = Path("data") / "mm-safebench_1" / "extracted_data"
_TRACES_DIR = _METADATA_DIR / "traces"


def resolve_cli_path(path: Path, repo_root: Path) -> Path:
    """Resolve a user-supplied path; relative paths are anchored at repo_root."""
    path = Path(path)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def metadata_split_for_input(csv_path: Path) -> str | None:
    """
    Decide whether to use train_metadata or test_metadata.

    Prefer filename prefixes (traces/train_method1.csv -> train). Fall back to
    a parent directory named 'train' or 'test' for older layouts.
    """
    name = csv_path.name.lower()
    if name.startswith("train_"):
        return "train"
    if name.startswith("test_"):
        return "test"
    parts = {p.lower() for p in csv_path.parts}
    if "train" in parts:
        return "train"
    if "test" in parts:
        return "test"
    return None


def _posix(s: str) -> str:
    return s.strip().replace("\\", "/")


def _suffix_after(marker: str, posix_path: str) -> str | None:
    m = marker.lower()
    lower = posix_path.lower()
    i = lower.find(m)
    if i < 0:
        return None
    return posix_path[i + len(marker) :].lstrip("/\\")


def key_variants(path_str: str, repo_root: Path) -> list[str]:
    """Generate comparable path strings for lookup (order: most specific first)."""
    if not path_str or not path_str.strip():
        return []
    raw = path_str.strip()
    out: list[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        x = x.strip()
        if not x or x in seen:
            return
        seen.add(x)
        out.append(x)

    add(raw)
    px = _posix(raw)
    add(px)

    for marker in ("extracted_data/", "mm-safebench_1/"):
        tail = _suffix_after(marker, px)
        if tail:
            add(tail)
            add(marker.rstrip("/") + "/" + tail)

    rel = Path(px)
    if not rel.is_absolute():
        abs_p = (repo_root / rel).resolve()
        add(_posix(str(abs_p)))
        try:
            add(_posix(str(abs_p.relative_to(repo_root.resolve()))))
        except ValueError:
            pass

    add(rel.name)
    return out


def build_lookup(
    metadata_path: Path,
    repo_root: Path,
) -> dict[str, str]:
    """Map normalized path keys -> question text."""
    lookup: dict[str, str] = {}
    with metadata_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header row in {metadata_path}")
        fields = {h.strip().lower(): h for h in reader.fieldnames}
        if "image" not in fields:
            raise KeyError(f"'image' column missing in {metadata_path}; got {reader.fieldnames}")
        if "question" not in fields:
            raise KeyError(f"'question' column missing in {metadata_path}; got {reader.fieldnames}")
        img_col = fields["image"]
        q_col = fields["question"]
        for row in reader:
            img = row.get(img_col, "") or ""
            q = row.get(q_col, "") or ""
            for key in key_variants(img, repo_root):
                lookup[key] = q
    return lookup


def resolve_prompt(
    image_cell: str,
    lookup: dict[str, str],
    repo_root: Path,
) -> str | None:
    for key in key_variants(image_cell, repo_root):
        if key in lookup:
            return lookup[key]
    return None


def process_one_csv(
    csv_path: Path,
    lookup: dict[str, str],
    repo_root: Path,
    out_path: Path,
    dry_run: bool,
) -> tuple[int, int]:
    """
    Returns (rows_written, unmatched_rows).
    """
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header in {csv_path}")
        fieldnames = list(reader.fieldnames)
        if "image" not in fieldnames:
            raise KeyError(f"'image' column missing in {csv_path}")
        if "prompt" in fieldnames:
            fieldnames.remove("prompt")
        idx = fieldnames.index("image") + 1
        fieldnames.insert(idx, "prompt")
        rows = list(reader)

    unmatched = 0
    for row in rows:
        img = row.get("image", "") or ""
        prompt = resolve_prompt(img, lookup, repo_root)
        if prompt is None:
            unmatched += 1
            row["prompt"] = ""
        else:
            row["prompt"] = prompt

    if dry_run:
        return len(rows), unmatched

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), unmatched


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root for resolving relative image paths (default: VG-SPV root).",
    )
    p.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing train_metadata.csv and test_metadata.csv "
            "(default: data/mm-safebench_1/extracted_data under --repo-root)."
        ),
    )
    p.add_argument(
        "--traces-dir",
        type=Path,
        default=None,
        help=(
            "Directory holding trace CSVs when --inputs is omitted "
            "(default: data/mm-safebench_1/extracted_data/traces under --repo-root)."
        ),
    )
    p.add_argument(
        "--train-metadata",
        type=Path,
        default=None,
        help="Override path to train_metadata.csv (default: <metadata-dir>/train_metadata.csv).",
    )
    p.add_argument(
        "--test-metadata",
        type=Path,
        default=None,
        help="Override path to test_metadata.csv (default: <metadata-dir>/test_metadata.csv).",
    )
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Explicit trace CSV paths; default: train_method{1,2}.csv and test_method{1,2}.csv "
            "under --traces-dir."
        ),
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite each input CSV instead of writing *_with_prompt.csv alongside.",
    )
    p.add_argument(
        "--output-suffix",
        type=str,
        default="_with_prompt",
        help="When not --in-place, output filename is stem+suffix+.csv (default: _with_prompt).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load metadata and report match counts only; do not write files.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    meta_dir = resolve_cli_path(args.metadata_dir or _METADATA_DIR, repo_root)
    traces_dir = resolve_cli_path(args.traces_dir or _TRACES_DIR, repo_root)
    train_meta = resolve_cli_path(
        args.train_metadata or (meta_dir / "train_metadata.csv"),
        repo_root,
    )
    test_meta = resolve_cli_path(
        args.test_metadata or (meta_dir / "test_metadata.csv"),
        repo_root,
    )

    if not train_meta.is_file():
        print(f"error: train metadata not found: {train_meta}", file=sys.stderr)
        return 1
    if not test_meta.is_file():
        print(f"error: test metadata not found: {test_meta}", file=sys.stderr)
        return 1

    train_lookup = build_lookup(train_meta, repo_root)
    test_lookup = build_lookup(test_meta, repo_root)

    if args.inputs:
        inputs = [resolve_cli_path(p, repo_root) for p in args.inputs]
    else:
        inputs = [
            traces_dir / "train_method1.csv",
            traces_dir / "train_method2.csv",
            traces_dir / "test_method1.csv",
            traces_dir / "test_method2.csv",
        ]

    missing_inputs = [p for p in inputs if not p.is_file()]
    if missing_inputs:
        for p in missing_inputs:
            print(f"warning: skip missing file: {p}", file=sys.stderr)
        inputs = [p for p in inputs if p.is_file()]
    if not inputs:
        print("error: no input CSVs to process", file=sys.stderr)
        return 1

    total_rows = 0
    total_unmatched = 0
    for csv_path in inputs:
        split = metadata_split_for_input(csv_path)
        if split == "train":
            lookup = train_lookup
        elif split == "test":
            lookup = test_lookup
        else:
            print(
                f"error: cannot infer train vs test (name with train_* / test_* prefix, "
                f"or path under a train/ or test/ directory): {csv_path}",
                file=sys.stderr,
            )
            return 1

        if args.in_place:
            out_path = csv_path
        else:
            out_path = csv_path.with_name(f"{csv_path.stem}{args.output_suffix}{csv_path.suffix}")

        n, bad = process_one_csv(csv_path, lookup, repo_root, out_path, args.dry_run)
        total_rows += n
        total_unmatched += bad
        action = "would write" if args.dry_run else "wrote"
        dest = out_path if not args.dry_run else f"{out_path} (dry-run)"
        print(f"{action} {n} rows from {csv_path} using {split} metadata -> {dest}; unmatched={bad}")

    print(
        f"summary: rows={total_rows}, unmatched={total_unmatched}, "
        f"train_keys={len(train_lookup)}, test_keys={len(test_lookup)}"
    )
    if total_unmatched:
        print(
            "warning: some images had no metadata match; those rows have an empty prompt column.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
