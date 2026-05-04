"""Hugging Face ``Trainer`` subclass for bounding-box SFT: resilient checkpoint writes."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from transformers import Trainer


class BoundingBoxSFTTrainer(Trainer):
    """
    If a periodic HF checkpoint fails (disk full, NFS glitch, etc.), log and continue training
    while still mirroring PEFT weights to ``adapter_latest`` so work is not lost.
    """

    def __init__(
        self,
        *args,
        bbox_tokenizer: Any | None = None,
        bbox_peft_save: Callable[[Any, Any, Any, Path], None] | None = None,
        bbox_adapter_latest_dir: Path | str | None = None,
        bbox_processor: Any | None = None,
        **kwargs,
    ):
        # ``tokenizer=...`` support in ``Trainer.__init__`` varies across transformers releases.
        # Consume it here so callers on older/newer versions do not crash with unexpected kwargs.
        if bbox_tokenizer is None and "tokenizer" in kwargs:
            bbox_tokenizer = kwargs.pop("tokenizer")
        super().__init__(*args, **kwargs)
        self._bbox_tokenizer = bbox_tokenizer
        self._bbox_peft_save = bbox_peft_save
        self._bbox_adapter_latest = Path(bbox_adapter_latest_dir) if bbox_adapter_latest_dir else None
        self._bbox_processor = bbox_processor

    def _save_checkpoint(self, model, trial, metrics=None):
        try:
            return super()._save_checkpoint(model, trial, metrics=metrics)
        except (OSError, RuntimeError) as e:
            print(
                f"[bbox_sft] WARNING: checkpoint save failed; training continues. ({type(e).__name__}: {e})",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            self._try_mirror_peft_only(model)
            return None

    def _try_mirror_peft_only(self, model) -> None:
        if self._bbox_peft_save is None or self._bbox_adapter_latest is None:
            return
        tok = self._bbox_tokenizer
        if tok is None:
            print("[bbox_sft] WARNING: no tokenizer on trainer; skipping PEFT mirror.", file=sys.stderr, flush=True)
            return
        try:
            self._bbox_adapter_latest.mkdir(parents=True, exist_ok=True)
            self._bbox_peft_save(model, tok, self._bbox_processor, self._bbox_adapter_latest)
            print(
                f"[bbox_sft] Mirrored PEFT bundle to {self._bbox_adapter_latest} after failed checkpoint.",
                file=sys.stderr,
                flush=True,
            )
        except Exception as e2:
            print(
                f"[bbox_sft] ERROR: PEFT mirror after failed checkpoint also failed: {e2}",
                file=sys.stderr,
                flush=True,
            )
