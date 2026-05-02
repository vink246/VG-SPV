"""
Batch multimodal conversations for bounding-box SFT: mask loss on user/prompt tokens only.

Supports HF grounding rows and optional VG-SPV CSV rows (image + instruction + chosen trace).
"""

from __future__ import annotations

from typing import Any

from train.bounding_box_sft_dataset import hf_row_to_bbox_sft_sample, vgspv_csv_row_to_bbox_sft_sample


def _split_user_assistant(messages: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    if len(messages) < 2:
        raise ValueError("Expected user + assistant messages")
    return [messages[0]], messages


class BoundingBoxSFTCollator:
    def __init__(
        self,
        processor: Any,
        hf_dataset: Any,
        family: str,
        *,
        csv_rows: list[dict[str, Any]] | None = None,
        vgspv_prompt_instruction: str = "",
    ):
        self.processor = processor
        self.hf = hf_dataset
        self.family = family
        self.csv_rows = csv_rows
        self.vgspv_prompt_instruction = vgspv_prompt_instruction
        tok = getattr(processor, "tokenizer", processor)
        tok.padding_side = "right"

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_lens: list[int] = []
        full_texts: list[str] = []
        images: list[Any] = []

        for b in batch:
            src = b.get("source", "hf")
            idx = int(b["idx"])
            if src == "vgspv_csv":
                if not self.csv_rows:
                    raise ValueError("Collator received vgspv_csv batch but csv_rows is empty")
                sample = vgspv_csv_row_to_bbox_sft_sample(
                    self.csv_rows[idx],
                    self.vgspv_prompt_instruction,
                )
            else:
                sample = hf_row_to_bbox_sft_sample(self.hf[idx])
            messages = sample.messages
            user_only, full_msgs = _split_user_assistant(messages)
            text_prompt = self.processor.apply_chat_template(
                user_only,
                tokenize=False,
                add_generation_prompt=True,
            )
            text_full = self.processor.apply_chat_template(
                full_msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            img = None
            for part in messages[0]["content"]:
                if isinstance(part, dict) and part.get("type") == "image":
                    img = part.get("image")
                    break
            if img is None:
                raise ValueError("Missing image in user message")
            images.append(img)
            full_texts.append(text_full)

            enc_p = self.processor(
                images=[img],
                text=[text_prompt],
                padding=False,
                return_tensors="pt",
            )
            enc_f = self.processor(
                images=[img],
                text=[text_full],
                padding=False,
                return_tensors="pt",
            )
            fp = enc_p["input_ids"][0].tolist()
            ff = enc_f["input_ids"][0].tolist()
            pl = 0
            for tok_a, tok_b in zip(fp, ff):
                if tok_a != tok_b:
                    break
                pl += 1
            prompt_lens.append(pl)

        enc = self.processor(
            images=images,
            text=full_texts,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        labels = input_ids.clone()
        for i, pl in enumerate(prompt_lens):
            pl = min(pl, labels.shape[1])
            labels[i, :pl] = -100
        labels[input_ids == self.processor.tokenizer.pad_token_id] = -100
        out = dict(enc)
        out["labels"] = labels
        return out
