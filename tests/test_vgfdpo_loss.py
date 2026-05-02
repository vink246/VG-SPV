import re
import unittest

import torch
import torch.nn as nn

from train.dpo_trainer import VGSPVTrainer


class _DummyTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inv = {}
        self._next = 10

    def _tok(self, text: str):
        return re.findall(r"<[^>]+>|[^\s]+", text)

    def encode(self, text: str, add_special_tokens: bool = False):
        out = []
        for t in self._tok(text):
            if t not in self.vocab:
                self.vocab[t] = self._next
                self.inv[self._next] = t
                self._next += 1
            out.append(self.vocab[t])
        return out

    def decode(self, ids, skip_special_tokens: bool = False):
        return " ".join(self.inv.get(i, "<unk>") for i in ids)


class _ToyModel(nn.Module):
    def __init__(self, vocab_size: int = 2048, hidden: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids=None, attention_mask=None):
        h = self.emb(input_ids)
        logits = self.proj(h)
        return type("Out", (), {"logits": logits})()


def _build_trainer(mode: str = "semantic") -> VGSPVTrainer:
    t = VGSPVTrainer.__new__(VGSPVTrainer)
    t.tokenizer = _DummyTokenizer()
    t.alpha_vdpo = 0.2
    t.vdpo_margin_m = 0.1
    t.alpha_format = 12.0
    t.grounding_mode = mode
    t.alpha_sem = 1.0
    t.alpha_iou = 1.0
    t.s_fallback_value = 1.0
    t.strict_scaler_inputs = False
    t.beta = 0.1
    t.ref_model = None
    t.log = lambda metrics: None
    t._tag_ids = t._build_tag_ids()
    return t


class TestVGFDPOLoss(unittest.TestCase):
    def test_segment_masks_include_logic_and_response_tags(self):
        t = _build_trainer()
        text = (
            "<risk_factors> knife </risk_factors> "
            "<logic> unsafe object visible </logic> "
            "<response> I cannot help </response>"
        )
        ids = t.tokenizer.encode(text)
        labels = torch.tensor([-100] + ids, dtype=torch.long)
        spans = t._sample_spans(labels)
        self.assertTrue(spans.parsed_ok)

        kept_ids = [tok for tok, keep in zip(labels[1:][spans.valid].tolist(), spans.m_l.tolist()) if keep]
        kept_text = t.tokenizer.decode(kept_ids)
        self.assertIn("<logic>", kept_text)
        self.assertIn("</logic>", kept_text)
        self.assertIn("<response>", kept_text)
        self.assertIn("</response>", kept_text)

    def test_format_fail_when_required_tags_missing(self):
        t = _build_trainer()
        text = "<risk_factors> knife </risk_factors> <logic> unsafe </logic>"
        ids = t.tokenizer.encode(text)
        labels = torch.tensor([-100] + ids, dtype=torch.long)
        spans = t._sample_spans(labels)
        self.assertFalse(spans.parsed_ok)

    def test_semantic_and_spatial_scalers(self):
        t_sem = _build_trainer(mode="semantic")
        s_sem, invalid_sem = t_sem._compute_s(
            "<risk_factors> knife threat </risk_factors>",
            "<risk_factors> knife </risk_factors>",
        )
        self.assertFalse(invalid_sem)
        self.assertGreaterEqual(s_sem, 0.0)

        t_sp = _build_trainer(mode="spatial")
        s_sp, invalid_sp = t_sp._compute_s(
            "<risk_factors_with_boxes> knife [100, 100, 300, 300] </risk_factors_with_boxes>",
            "<risk_factors_with_boxes> knife [120, 120, 320, 320] </risk_factors_with_boxes>",
        )
        self.assertFalse(invalid_sp)
        self.assertGreaterEqual(s_sp, 0.0)

    def test_compute_loss_returns_total_with_vdpo_metrics(self):
        t = _build_trainer(mode="semantic")
        model = _ToyModel()

        chosen = (
            "<risk_factors> knife </risk_factors> "
            "<logic> unsafe </logic> <response> refuse </response>"
        )
        rejected = (
            "<risk_factors> none </risk_factors> "
            "<logic> safe </logic> <response> comply </response>"
        )

        chosen_ids = t.tokenizer.encode(chosen)
        rejected_ids = t.tokenizer.encode(rejected)

        def mk(ids):
            seq = [1] + ids
            labels = [-100] + ids
            return (
                torch.tensor([seq], dtype=torch.long),
                torch.tensor([[1] * len(seq)], dtype=torch.long),
                torch.tensor([labels], dtype=torch.long),
            )

        c_inp, c_attn, c_lbl = mk(chosen_ids)
        r_inp, r_attn, r_lbl = mk(rejected_ids)
        cp_inp, cp_attn, cp_lbl = mk(chosen_ids)

        inputs = {
            "chosen_input_ids": c_inp,
            "chosen_attention_mask": c_attn,
            "chosen_labels": c_lbl,
            "rejected_input_ids": r_inp,
            "rejected_attention_mask": r_attn,
            "rejected_labels": r_lbl,
            "chosen_perturbed_input_ids": cp_inp,
            "chosen_perturbed_attention_mask": cp_attn,
            "chosen_perturbed_labels": cp_lbl,
        }

        loss, outputs = t.compute_loss(model, inputs, return_outputs=True)
        self.assertTrue(torch.is_tensor(loss))
        self.assertIn("metrics", outputs)
        self.assertIn("loss_total", outputs["metrics"])
        self.assertIn("loss_vdpo", outputs["metrics"])


if __name__ == "__main__":
    unittest.main()
