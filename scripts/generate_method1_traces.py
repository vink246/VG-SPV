"""
Generate Method 1 (semantic) preferred-response traces for VG-SPV.

For each MM-SafetyBench (image + question) row, this script asks GPT-5.4-mini
(via the OpenAI Batch API on /v1/responses) to produce a structured trace with
``<risk_factors>``, ``<logic>``, and ``<response>`` tags using the verbatim
prompt template from ``immediate instructions.pdf``.

Reasoning controls are fixed per project convention:
  - ``reasoning.effort = "medium"``
  - ``reasoning.summary = null``  (no summary emitted)
  - ``text.verbosity   = "low"``

Cohort filter (applied in both ``prepare`` and ``collect``):
  Only metadata rows whose ``_category`` is in ``MMSB_COHORT_CATEGORIES``
  (``{"Physical_Harm", "Illegal_Activitiy"}`` -- preserve the upstream typo)
  AND whose ``_subset`` string contains ``MMSB_COHORT_SUBSET_SUBSTRING``
  (``"SD"``) are sent to OpenAI or written to the trainer CSV.

Outputs (per split):
  - data/mm-safebench_1/extracted_data/traces/{split}_method1.csv
        Trainer-ready CSV (matches ``train/dataset_adapter.py`` column contract):
        ``image, perturbed_image, chosen_reasoning_trace, rejected_reasoning_trace``
        where ``chosen_reasoning_trace`` is the Method 1 XML trace from the model
        and ``perturbed_image`` / ``rejected_reasoning_trace`` are empty strings
        for now (reserved for §5.2/§5.3 follow-ups).
  - data/mm-safebench_1/extracted_data/traces/_batches/{split}/
        OpenAI batch artifacts (input/output JSONL, state.json, errors, malformed log)
        kept for reproducibility/auditability.

Subcommands (default ``run`` chains ``prepare -> submit -> wait -> collect``):
    prepare   build OpenAI batch input JSONL(s) from the split's metadata
    submit    upload + create batch jobs against /v1/responses
    wait      poll batch state and download outputs/errors when terminal
    collect   join outputs by custom_id, parse XML, write trainer-ready CSV
    run       prepare -> submit -> wait -> collect end to end

Invocation (always from repo root):
    python -m scripts.generate_method1_traces prepare --split test
    python -m scripts.generate_method1_traces submit  --split test
    python -m scripts.generate_method1_traces wait    --split test
    python -m scripts.generate_method1_traces collect --split test
    python -m scripts.generate_method1_traces run     --split test

Requires the env var ``OPENAI_API_KEY``. Install ``openai>=1.40.0`` in your
environment (already in ``environment.yml``); pip fallback for existing envs:

    pip install 'openai>=1.40.0'
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import mimetypes
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DATASET_DIR = Path("data/mm-safebench_1/extracted_data")
DEFAULT_OUTPUT_TRACES_SUBDIR = "traces"

# Model + endpoint. We use the Responses API (not chat/completions) because it is
# the surface that exposes ``reasoning.effort``, ``reasoning.summary``, and
# ``text.verbosity`` for GPT-5 reasoning models.
DEFAULT_BATCH_MODEL = "gpt-5.4-mini"
BATCH_ENDPOINT = "/v1/responses"

# Reasoning + text-generation knobs (project-fixed defaults).
#   effort:  "minimal" | "low" | "medium" | "high"
#   summary: "auto" | "concise" | "detailed" | None  (None == no summary)
#   verbosity: "low" | "medium" | "high"
DEFAULT_REASONING_EFFORT: str = "medium"
DEFAULT_REASONING_SUMMARY: str | None = None
DEFAULT_TEXT_VERBOSITY: str = "low"

# Reasoning models charge "hidden" reasoning tokens against this budget on top of
# the visible XML output, so leave generous headroom even with verbosity=low.
DEFAULT_MAX_OUTPUT_TOKENS = 4000

# Per OpenAI Batch limits (current docs: 50,000 rows and 200 MB per input file).
# Stay comfortably under both to avoid edge-case rejections.
DEFAULT_MAX_LINES_PER_BATCH = 45_000
DEFAULT_MAX_BYTES_PER_BATCH = 180 * 1024 * 1024

# MM-SafetyBench cohort filter. Only metadata rows whose ``_category`` is in
# ``MMSB_COHORT_CATEGORIES`` AND whose ``_subset`` contains
# ``MMSB_COHORT_SUBSET_SUBSTRING`` are sent to OpenAI. Applied in both
# ``cmd_prepare`` (skips non-cohort rows when building the batch input) and
# ``cmd_collect`` (skips any output row whose source metadata is out-of-cohort,
# defending against stale batches built before this filter was added).
# NOTE: "Illegal_Activitiy" is the upstream dataset's typo and is preserved
# verbatim so the string match works against the on-disk metadata.
MMSB_COHORT_CATEGORIES = frozenset({"Physical_Harm", "Illegal_Activitiy"})
MMSB_COHORT_SUBSET_SUBSTRING = "SD"


# Prompt template. The user query is
# substituted at the indicated position; everything else (including the
# ``[Image attached]`` literal and ``### STRICT RULES``) is preserved as-is.
PROMPT_TEMPLATE = """You are an expert multimodal safety evaluator and data synthesizer. You will be provided with an image and a user query.
Your task is to identify any visual safety threats, evaluate the user's intent, and generate a highly structured "reasoning trace" that explicitly grounds the safety decision in the visual evidence.
You MUST format your output EXACTLY using the following three XML tags. Do not include any conversational filler outside of these tags.
<risk_factors>
[Provide a comma-separated list of keywords that are identifiable visible nouns that may be explicitly circled as evidence in the image, which could be dangerous, sensitive, or policy-violating. Abstract nouns ('violence') are not allowed, while explicit objects ('firearm') or clearly visible actions ('beating') are helpful and encouraged; if an object detection model could pick it out, it's a good risk factor to report. If no visual threat is present in the image, you MUST output exactly: no risk]
</risk_factors>
<logic>
[Provide a concise, logical chain of reasoning. Step 1: Briefly describe the relevant visual elements identified. Step 2: Explain how these visual elements provide context to the user's query. Step 3: Conclude whether fulfilling the query violates safety policies based on the visual evidence.]
</logic>
<response>
[Provide the final response to the user. If a safety risk was identified in the logic step, provide a clear, polite refusal. If "no risk" was identified and the query is benign, safely fulfill the user's request.]
</response>
### STRICT RULES:
1. Output ONLY the XML tags. No introductory or concluding remarks (e.g., do not say "Here is the response:").
2. The <risk_factors> must be explicit and derived strictly from what is physically visible in the image. Do not hallucinate items that are not there.
3. The <logic> must explicitly connect the visual evidence to the safety conclusion.
4. Be concise and use less corporate jargon.
### USER QUERY:
{user_query}
[Image attached]
"""


# ---------------------------- common path helpers ----------------------------


def _split_dirs(dataset_dir: Path, split: str) -> tuple[Path, Path, Path]:
    """Return (traces_dir, batches_dir, split_batches_dir) for a given dataset_dir/split."""
    traces_dir = dataset_dir / DEFAULT_OUTPUT_TRACES_SUBDIR
    batches_dir = traces_dir / "_batches"
    split_batches = batches_dir / split
    return traces_dir, batches_dir, split_batches


def _resolve_image_path(rel_image: str, data_root: Path) -> Path:
    """
    Map MM-SafetyBench's stored ``image`` field (e.g. ``extracted_media/test/Foo.png``)
    to an actual file under ``data_root``. The ``extracted_media/`` prefix is stripped
    because images live directly under ``{data_root}/{split}/...``.
    """
    p = Path(rel_image)
    parts = p.parts
    if parts and parts[0] == "extracted_media":
        parts = parts[1:]
    return data_root.joinpath(*parts) if parts else data_root


def _csv_image_field(resolved: Path) -> str:
    """POSIX-style repo-relative path string for the CSV ``image`` column."""
    try:
        rel = resolved.resolve().relative_to(Path.cwd().resolve())
        return rel.as_posix()
    except ValueError:
        return resolved.as_posix()


def _row_passes_mmsb_filter(row: dict[str, Any]) -> bool:
    """Return True iff a metadata row is in the configured MM-SafetyBench cohort.

    Cohort = ``_category`` in ``MMSB_COHORT_CATEGORIES`` AND ``_subset`` is a
    string that contains ``MMSB_COHORT_SUBSET_SUBSTRING``. Rows missing either
    field, or with the wrong type, are treated as out-of-cohort.
    """
    category = row.get("_category")
    subset = row.get("_subset")
    if category not in MMSB_COHORT_CATEGORIES:
        return False
    if not isinstance(subset, str) or MMSB_COHORT_SUBSET_SUBSTRING not in subset:
        return False
    return True


# ---------------------------- prepare phase ----------------------------


def _iter_metadata(metadata_path: Path) -> Iterable[dict[str, Any]]:
    """Read MM-SafetyBench metadata. Supports both .jsonl and .csv."""
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if metadata_path.suffix.lower() == ".jsonl":
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    elif metadata_path.suffix.lower() == ".csv":
        with metadata_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                yield row
    else:
        raise ValueError(
            f"Unsupported metadata extension: {metadata_path.suffix}. Expected .jsonl or .csv."
        )


def _encode_image_b64(image_path: Path) -> tuple[str, str]:
    """Return (mime_type, base64_str) for a local image file."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is None:
        mime = "image/png"
    with image_path.open("rb") as f:
        raw = f.read()
    return mime, base64.b64encode(raw).decode("ascii")


def build_request(custom_id: str, image_b64: str, mime_type: str, user_query: str) -> dict[str, Any]:
    """Build a single OpenAI Batch API row targeting /v1/responses.

    Notes on the Responses API request shape (vs. chat/completions):
      - top-level field is ``input`` (not ``messages``)
      - text parts use ``type: "input_text"`` (not ``"text"``)
      - image parts use ``type: "input_image"`` with ``image_url`` as a string
        data URL (not a nested ``{"url": ...}`` object)
      - reasoning models do not accept ``temperature``; we omit it
      - cap is ``max_output_tokens`` (not ``max_tokens``)
    """
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": {
            "model": DEFAULT_BATCH_MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": PROMPT_TEMPLATE.format(user_query=user_query),
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime_type};base64,{image_b64}",
                        },
                    ],
                }
            ],
            "reasoning": {
                "effort": DEFAULT_REASONING_EFFORT,
                "summary": DEFAULT_REASONING_SUMMARY,
            },
            "text": {"verbosity": DEFAULT_TEXT_VERBOSITY},
            "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        },
    }


def _format_request_line(request: dict[str, Any]) -> str:
    return json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n"


@dataclass
class _ChunkWriter:
    """Streams batch input JSONL rows to one or more chunked files (line/byte capped)."""

    out_dir: Path
    max_lines: int
    max_bytes: int

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._chunk_idx = 0
        self._handle: io.TextIOBase | None = None
        self._lines = 0
        self._bytes = 0
        self._paths: list[Path] = []
        self._open_new()

    def _open_new(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._chunk_idx += 1
        path = self.out_dir / f"input_{self._chunk_idx:03d}.jsonl"
        self._handle = path.open("w", encoding="utf-8")
        self._lines = 0
        self._bytes = 0
        self._paths.append(path)

    def write(self, request: dict[str, Any]) -> None:
        line = _format_request_line(request)
        line_bytes = len(line.encode("utf-8"))
        if self._lines >= self.max_lines or self._bytes + line_bytes > self.max_bytes:
            self._open_new()
        assert self._handle is not None
        self._handle.write(line)
        self._lines += 1
        self._bytes += line_bytes

    def close(self) -> list[Path]:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        return [p for p in self._paths if p.stat().st_size > 0]


def cmd_prepare(args: argparse.Namespace) -> list[Path]:
    dataset_dir: Path = args.dataset_dir
    split: str = args.split
    metadata_path: Path = (
        args.metadata if args.metadata else dataset_dir / f"{split}_metadata.jsonl"
    )
    data_root: Path = args.data_root if args.data_root else dataset_dir
    _, _, split_batches = _split_dirs(dataset_dir, split)

    writer = _ChunkWriter(split_batches, args.max_lines, args.max_bytes)
    n_written = 0
    n_skipped = 0
    n_filtered = 0
    skipped_log: list[dict[str, Any]] = []

    for i, row in enumerate(_iter_metadata(metadata_path)):
        if args.limit is not None and n_written >= args.limit:
            break
        if not _row_passes_mmsb_filter(row):
            n_filtered += 1
            continue
        # Derive a guaranteed-unique row key. The metadata "id" field on
        # MM-SafetyBench is a per-(category, subset) row index from the source
        # parquet, so it is NOT unique across the cohort -- relying on it
        # produced "duplicate_custom_id" batch rejections from OpenAI. The
        # metadata file's enumeration index ``i`` is unique by construction
        # and is what cmd_collect also uses as its lookup key.
        rid = f"row{i}"
        rel_image = row.get("image")
        question = row.get("question")
        if not rel_image or not question:
            n_skipped += 1
            skipped_log.append({"row_index": i, "id": rid, "reason": "missing image or question"})
            continue
        image_path = _resolve_image_path(str(rel_image), data_root)
        try:
            mime, b64 = _encode_image_b64(image_path)
        except FileNotFoundError:
            n_skipped += 1
            skipped_log.append(
                {"row_index": i, "id": rid, "reason": f"image not found: {image_path}"}
            )
            continue

        custom_id = f"{split}-{rid}"
        request = build_request(
            custom_id=custom_id,
            image_b64=b64,
            mime_type=mime,
            user_query=str(question),
        )
        writer.write(request)
        n_written += 1

    written_paths = writer.close()

    if skipped_log:
        skipped_path = split_batches / "prepare_skipped.jsonl"
        with skipped_path.open("w", encoding="utf-8") as f:
            for entry in skipped_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Logged {n_skipped} skipped rows to {skipped_path}")

    if n_filtered:
        print(
            f"Filtered out {n_filtered} non-cohort row(s) "
            f"(kept categories={sorted(MMSB_COHORT_CATEGORIES)}, "
            f"_subset must contain '{MMSB_COHORT_SUBSET_SUBSTRING}')."
        )

    print(
        f"Prepared {n_written} request(s) across {len(written_paths)} chunk file(s) under {split_batches}:"
    )
    for p in written_paths:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name} ({size_mb:.2f} MB)")
    return written_paths


# ---------------------------- submit phase ----------------------------


def _state_path(split_batches: Path) -> Path:
    return split_batches / "state.json"


def _load_state(split_batches: Path) -> dict[str, Any]:
    sp = _state_path(split_batches)
    if not sp.is_file():
        return {"chunks": []}
    return json.loads(sp.read_text(encoding="utf-8"))


def _save_state(split_batches: Path, state: dict[str, Any]) -> None:
    sp = _state_path(split_batches)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _make_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai>=1.40.0 is required. Install with `pip install 'openai>=1.40.0'`."
        ) from exc
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY env var is not set. Export it before running this command."
        )
    return OpenAI(api_key=api_key)


def cmd_submit(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir: Path = args.dataset_dir
    split: str = args.split
    _, _, split_batches = _split_dirs(dataset_dir, split)

    inputs = sorted(split_batches.glob("input_*.jsonl"))
    if not inputs:
        raise SystemExit(
            f"No input chunks found under {split_batches}. Run `prepare` first."
        )

    client = _make_openai_client()
    state = _load_state(split_batches)
    existing_by_path = {c["input_path"]: c for c in state.get("chunks", [])}

    # Statuses where the existing batch is dead and a fresh submit is required.
    # Anything else (including in-progress and unknown) means we keep the
    # existing batch_id and let `wait` deal with it.
    _DEAD_STATUSES = {"failed", "expired", "cancelled"}

    new_chunks: list[dict[str, Any]] = []
    for path in inputs:
        rec = existing_by_path.get(str(path))
        existing_status = rec.get("status") if rec else None
        existing_bid = rec.get("batch_id") if rec else None
        is_dead = existing_status in _DEAD_STATUSES

        if existing_bid and not args.resubmit and not is_dead:
            print(
                f"Skipping {path.name}: already submitted as batch "
                f"{existing_bid} (status={existing_status})"
            )
            new_chunks.append(rec)
            continue
        if existing_bid and is_dead:
            print(
                f"Resubmitting {path.name}: previous batch {existing_bid} "
                f"is {existing_status}; creating a fresh one."
            )

        print(f"Uploading {path.name}...")
        with path.open("rb") as f:
            up = client.files.create(file=f, purpose="batch")
        print(f"  file id: {up.id}")

        print("  Creating batch...")
        batch = client.batches.create(
            input_file_id=up.id,
            endpoint=BATCH_ENDPOINT,
            completion_window="24h",
            metadata={"split": split, "chunk": path.name},
        )
        print(f"  batch id: {batch.id} (status: {batch.status})")

        new_chunks.append(
            {
                "input_path": str(path),
                "input_file_id": up.id,
                "batch_id": batch.id,
                "status": batch.status,
                "submitted_at": time.time(),
            }
        )

    state["chunks"] = new_chunks
    state["split"] = split
    _save_state(split_batches, state)
    print(f"Persisted state to {_state_path(split_batches)}")
    return state


# ---------------------------- wait phase ----------------------------


_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def _download_file(client, file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(file_id)
    text: str | bytes
    if hasattr(content, "text"):
        text = content.text
    elif hasattr(content, "read"):
        text = content.read()
    else:
        text = str(content)
    if isinstance(text, str):
        dest.write_text(text, encoding="utf-8")
    else:
        dest.write_bytes(text)


def _surface_batch_errors(
    batch: Any,
    chunk: dict[str, Any],
    split_batches: Path,
    idx: str,
) -> int:
    """Pull ``batch.errors`` off a (typically failed) batch object, print a
    human-readable summary, persist the structured form into ``chunk`` and into
    ``_batches/{split}/batch_errors_{idx}.jsonl`` for later inspection.

    Returns the number of error entries surfaced (0 if none).
    """
    errors_obj = getattr(batch, "errors", None)
    if not errors_obj:
        return 0
    raw_list = getattr(errors_obj, "data", None) or []
    if not raw_list:
        return 0

    serialized = [
        {
            "code": getattr(e, "code", None),
            "message": getattr(e, "message", None),
            "param": getattr(e, "param", None),
            "line": getattr(e, "line", None),
        }
        for e in raw_list
    ]
    chunk["batch_errors"] = serialized

    err_path = split_batches / f"batch_errors_{idx}.jsonl"
    err_path.parent.mkdir(parents=True, exist_ok=True)
    with err_path.open("w", encoding="utf-8") as f:
        for e in serialized:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(
        f"  batch {batch.id}: {len(serialized)} batch-level error(s) "
        f"(saved to {err_path}):"
    )
    head = 10
    for e in serialized[:head]:
        print(
            f"    line={e['line']} code={e['code']} "
            f"param={e['param']}: {e['message']}"
        )
    if len(serialized) > head:
        print(f"    ... and {len(serialized) - head} more (see {err_path})")
    return len(serialized)


def cmd_wait(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir: Path = args.dataset_dir
    split: str = args.split
    _, _, split_batches = _split_dirs(dataset_dir, split)
    state = _load_state(split_batches)
    chunks: list[dict[str, Any]] = state.get("chunks", [])
    if not chunks:
        raise SystemExit(f"No batches recorded under {_state_path(split_batches)}.")

    client = _make_openai_client()

    while True:
        all_terminal = True
        for chunk in chunks:
            bid = chunk.get("batch_id")
            if not bid:
                continue
            idx = Path(chunk["input_path"]).stem.replace("input_", "")
            status = chunk.get("status")

            # Re-fetch terminal-but-failed chunks once if we've never surfaced
            # their batch.errors yet (idempotent: skipped on subsequent reruns).
            terminal_needs_errors = (
                status in _TERMINAL_STATUSES
                and status != "completed"
                and "batch_errors" not in chunk
            )
            if status in _TERMINAL_STATUSES and not terminal_needs_errors:
                continue

            batch = client.batches.retrieve(bid)
            chunk["status"] = batch.status
            counts = getattr(batch, "request_counts", None)
            ctotal = getattr(counts, "total", None) if counts else None
            ccomplete = getattr(counts, "completed", None) if counts else None
            cfailed = getattr(counts, "failed", None) if counts else None
            print(
                f"  batch {bid}: status={batch.status} "
                f"completed={ccomplete}/{ctotal} failed={cfailed}"
            )

            if batch.status not in _TERMINAL_STATUSES:
                all_terminal = False
                continue

            # --- terminal: download per-row outputs/errors files ---
            output_id = getattr(batch, "output_file_id", None)
            error_id = getattr(batch, "error_file_id", None)
            chunk["output_file_id"] = output_id
            chunk["error_file_id"] = error_id
            if output_id:
                out_path = split_batches / f"output_{idx}.jsonl"
                print(f"  Downloading outputs -> {out_path}")
                _download_file(client, output_id, out_path)
                chunk["output_path"] = str(out_path)
            if error_id:
                err_path = split_batches / f"errors_{idx}.jsonl"
                print(f"  Downloading errors -> {err_path}")
                _download_file(client, error_id, err_path)
                chunk["error_path"] = str(err_path)

            # --- terminal: surface batch-level errors (validation/quota) ---
            # These live on batch.errors (separate from the per-row error file)
            # and were the silent failure mode this codepath used to hide.
            if batch.status != "completed":
                _surface_batch_errors(batch, chunk, split_batches, idx)

        _save_state(split_batches, state)
        if all_terminal:
            print("All batches terminal.")
            break
        print(f"Sleeping {args.poll_interval}s...")
        time.sleep(args.poll_interval)
    return state


# ---------------------------- collect phase ----------------------------


_RISK_RE = re.compile(r"<risk_factors>(.*?)</risk_factors>", re.DOTALL | re.IGNORECASE)
_LOGIC_RE = re.compile(r"<logic>(.*?)</logic>", re.DOTALL | re.IGNORECASE)
_RESPONSE_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)


def parse_method1_xml(text: str) -> dict[str, Any] | None:
    """Parse a Method 1 trace into {risk_factors: [...], logic, response}. Return None if any tag is missing."""
    m_risk = _RISK_RE.search(text)
    m_logic = _LOGIC_RE.search(text)
    m_resp = _RESPONSE_RE.search(text)
    if not (m_risk and m_logic and m_resp):
        return None
    risk_blob = m_risk.group(1).strip()
    logic = m_logic.group(1).strip()
    response = m_resp.group(1).strip()

    if risk_blob.lower() == "no risk":
        risk_factors = ["no risk"]
    else:
        risk_factors = [r.strip() for r in risk_blob.split(",") if r.strip()]
        if not risk_factors:
            risk_factors = ["no risk"]
    return {"risk_factors": risk_factors, "logic": logic, "response": response}


def _extract_response_text(output_row: dict[str, Any]) -> str | None:
    """Pull the assistant text content from a /v1/responses batch output row.

    The Responses API returns ``body.output`` as an array of typed items
    (``"reasoning"``, ``"message"``, possibly others). We concatenate every
    ``output_text`` (and tolerated legacy ``text``) part inside any assistant
    ``message`` block. ``body.output_text`` is a Python-SDK convenience that is
    NOT present in raw batch JSON, so we never rely on it.
    """
    response = output_row.get("response")
    if not isinstance(response, dict):
        return None
    body = response.get("body")
    if not isinstance(body, dict):
        return None
    output = body.get("output")
    if not isinstance(output, list):
        return None
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("output_text", "text"):
                txt = part.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
    if not parts:
        return None
    return "".join(parts)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


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


CSV_FIELDNAMES = [
    "image",
    "perturbed_image",
    "chosen_reasoning_trace",
    "rejected_reasoning_trace",
]


def cmd_collect(args: argparse.Namespace) -> None:
    """
    Parse OpenAI batch outputs into the trainer-ready Method 1 CSV.

    The CSV is written from scratch on each invocation (the source of truth for
    rows is ``_batches/{split}/output_*.jsonl``, which itself is downloaded by
    ``wait`` and persisted under ``_batches/`` for reproducibility).

    Rows are sorted by ``custom_id`` so the CSV order is stable across reruns,
    and dedup'd by ``custom_id`` (later occurrences win, mirroring how the
    OpenAI batch service treats duplicate ``custom_id`` rows).
    """
    dataset_dir: Path = args.dataset_dir
    split: str = args.split
    data_root: Path = args.data_root if args.data_root else dataset_dir
    metadata_path: Path = (
        args.metadata if args.metadata else dataset_dir / f"{split}_metadata.jsonl"
    )
    traces_dir, _, split_batches = _split_dirs(dataset_dir, split)
    csv_path = traces_dir / f"{split}_method1.csv"
    malformed_path = split_batches / "malformed.jsonl"

    metadata_by_id: dict[str, dict[str, Any]] = {}
    n_meta_filtered = 0
    for i, row in enumerate(_iter_metadata(metadata_path)):
        # MUST match the key derivation in cmd_prepare. The metadata row's "id"
        # field is intentionally ignored (not unique on MM-SafetyBench across
        # categories/subsets); we key off the file row index instead.
        rid = f"row{i}"
        if not _row_passes_mmsb_filter(row):
            n_meta_filtered += 1
            continue
        metadata_by_id[rid] = row

    output_files = sorted(split_batches.glob("output_*.jsonl"))
    if not output_files:
        raise SystemExit(
            f"No output_*.jsonl files under {split_batches}. Run `wait` first."
        )

    parsed_rows: dict[str, dict[str, str]] = {}
    malformed: list[dict[str, Any]] = []
    n_total_outputs = 0
    n_out_of_cohort = 0

    for of in output_files:
        for output_row in _read_jsonl(of):
            n_total_outputs += 1
            cid = output_row.get("custom_id")
            if not isinstance(cid, str) or "-" not in cid:
                malformed.append({"reason": "missing custom_id", "row": output_row})
                continue
            rid = cid.split("-", 1)[1]
            err = output_row.get("error")
            if err:
                malformed.append({"id": rid, "reason": "api error", "error": err})
                continue
            text = _extract_response_text(output_row)
            if not text:
                malformed.append({"id": rid, "reason": "no response text"})
                continue
            parsed = parse_method1_xml(text)
            if not parsed:
                malformed.append({"id": rid, "reason": "xml tags missing", "raw": text})
                continue

            # Only the cohort survives the metadata-side filter above. Any output
            # row whose source metadata is out-of-cohort (e.g. from a stale batch
            # built before this filter was added) gets dropped here so the CSV
            # is always cohort-pure.
            meta = metadata_by_id.get(rid)
            if meta is None:
                n_out_of_cohort += 1
                continue
            rel_image = meta.get("image")
            image_path_str = ""
            if rel_image:
                resolved = _resolve_image_path(str(rel_image), data_root)
                image_path_str = _csv_image_field(resolved)

            parsed_rows[rid] = {
                "image": image_path_str,
                "perturbed_image": "",
                "chosen_reasoning_trace": text.strip(),
                "rejected_reasoning_trace": "",
            }

    if malformed:
        with malformed_path.open("w", encoding="utf-8") as f:
            for m in malformed:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"Wrote {len(malformed)} malformed/error rows to {malformed_path}")

    sorted_csv_rows = [parsed_rows[rid] for rid in sorted(parsed_rows.keys())]
    _atomic_write_csv(csv_path, sorted_csv_rows, CSV_FIELDNAMES)
    print(f"Wrote trainer-ready CSV {csv_path} ({len(sorted_csv_rows)} rows).")

    if n_meta_filtered:
        print(
            f"Cohort filter: dropped {n_meta_filtered} metadata row(s) "
            f"(kept categories={sorted(MMSB_COHORT_CATEGORIES)}, "
            f"_subset must contain '{MMSB_COHORT_SUBSET_SUBSTRING}')."
        )
    if n_out_of_cohort:
        print(
            f"Cohort filter: dropped {n_out_of_cohort} output row(s) "
            "whose source metadata is out-of-cohort (stale batch?)."
        )

    print(
        f"Done. Outputs scanned: {n_total_outputs}, parsed: {len(parsed_rows)}, "
        f"malformed: {len(malformed)}, out-of-cohort: {n_out_of_cohort}"
    )


# ---------------------------- run (chain) ----------------------------


def cmd_run(args: argparse.Namespace) -> None:
    cmd_prepare(args)
    cmd_submit(args)
    cmd_wait(args)
    cmd_collect(args)


# ---------------------------- CLI ----------------------------


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "test"],
        help="MM-SafetyBench split to process.",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"MM-SafetyBench extracted_data root (default: {DEFAULT_DATASET_DIR}).",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Image root used when resolving the metadata `image` field. Default: --dataset-dir.",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Override metadata file path. Default: {dataset-dir}/{split}_metadata.jsonl.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m scripts.generate_method1_traces",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("prepare", help="Build OpenAI batch input JSONL from metadata.")
    _add_common_args(sp)
    sp.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of requests (for smoke tests).",
    )
    sp.add_argument(
        "--max-lines",
        type=int,
        default=DEFAULT_MAX_LINES_PER_BATCH,
        help=f"Max lines per batch input chunk (default {DEFAULT_MAX_LINES_PER_BATCH}).",
    )
    sp.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES_PER_BATCH,
        help=f"Max bytes per batch input chunk (default {DEFAULT_MAX_BYTES_PER_BATCH}).",
    )

    sp = sub.add_parser("submit", help="Upload + create OpenAI batch jobs for prepared chunks.")
    _add_common_args(sp)
    sp.add_argument(
        "--resubmit",
        action="store_true",
        help="Re-submit chunks already recorded in state.json (otherwise skipped).",
    )

    sp = sub.add_parser("wait", help="Poll batch status until terminal and download outputs.")
    _add_common_args(sp)
    sp.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between status polls (default 60).",
    )

    sp = sub.add_parser("collect", help="Parse outputs and write the trainer-ready CSV.")
    _add_common_args(sp)

    sp = sub.add_parser("run", help="prepare -> submit -> wait -> collect end to end.")
    _add_common_args(sp)
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES_PER_BATCH)
    sp.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES_PER_BATCH)
    sp.add_argument("--resubmit", action="store_true")
    sp.add_argument("--poll-interval", type=int, default=60)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "wait":
        cmd_wait(args)
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main(sys.argv[1:])
