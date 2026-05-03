"""Unit tests for ``train.rejected_trace_builder`` (rejected DPO traces + bbox perturbation)."""

from __future__ import annotations

import random
import unittest

from train.rejected_trace_builder import (
    RESPONSE_REFUSAL_C,
    STEP3_SUFFIX,
    _plausible_wrong_factor,
    assemble_trace_xml,
    build_format_break_rejected,
    build_method1_risk_perturb_rejected,
    build_method2_bbox_perturb_rejected,
    build_method2_risk_perturb_rejected,
    build_prompt_branch_a,
    build_prompt_branch_b,
    build_rejected_trace_branch_ab,
    build_rejected_trace_branch_c,
    classify_branch,
    parse_trace_for_rejection,
    perturb_method2_risk_boxes_inner,
    perturb_norm_box,
    rejected_logic_inner_branch_ab,
    split_logic_steps,
)
from train.tag_parsing import iou_xyxy_norm


M1_RISK = """<risk_factors>
firearm, knife
</risk_factors>"""

M1_LOGIC_REFUSE = """<logic>
Step 1: There is a firearm visible.
Step 2: The user asks about weapons.
Step 3: This is unsafe.
</logic>"""

M1_RESP_SORRY = """<response>
Sorry, I cannot provide that information.
</response>"""

M1_FULL_A = f"{M1_RISK}\n{M1_LOGIC_REFUSE}\n{M1_RESP_SORRY}"

M1_LOGIC_COMPLY = """<logic>
Step 1: The image shows a cat.
Step 2: The query is benign.
Step 3: I can answer safely.
</logic>"""
M1_RESP_OK = """<response>
Here is a helpful answer about cats.
</response>"""
M1_FULL_C = f"""<risk_factors>
no risk
</risk_factors>
{M1_LOGIC_COMPLY}
{M1_RESP_OK}"""

M2_BODY = """phrase: "firearm" | box: [0100, 0100, 0400, 0400]
phrase: "knife" | box: [0500, 0500, 0800, 0800]"""

M2_FULL = f"""<risk_factors_with_boxes>
{M2_BODY}
</risk_factors_with_boxes>
{M1_LOGIC_REFUSE}
{M1_RESP_SORRY}"""


class TestSplitLogicSteps(unittest.TestCase):
    def test_three_steps(self):
        logic = (
            "Step 1: Alpha.\nStep 2: Beta.\nStep 3: Gamma."
        )
        s1, s2, s3 = split_logic_steps(logic)
        self.assertEqual(s1, "Alpha.")
        self.assertEqual(s2, "Beta.")
        self.assertEqual(s3, "Gamma.")

    def test_no_steps_returns_none(self):
        self.assertEqual(split_logic_steps("No numbered steps here."), (None, None, None))


class TestClassifyBranch(unittest.TestCase):
    def test_method1_branch_a(self):
        p = parse_trace_for_rejection(M1_FULL_A, "method1")
        assert p is not None
        self.assertEqual(classify_branch(p), "A")

    def test_method1_branch_c_no_sorry(self):
        """no risk + compliant response (no 'sorry') -> branch C."""
        p = parse_trace_for_rejection(M1_FULL_C, "method1")
        assert p is not None
        self.assertEqual(classify_branch(p), "C")

    def test_method1_branch_b_explicit(self):
        trace = f"""<risk_factors>
no risk
</risk_factors>
{M1_LOGIC_REFUSE}
{M1_RESP_SORRY}"""
        p = parse_trace_for_rejection(trace, "method1")
        assert p is not None
        self.assertEqual(classify_branch(p), "B")


class TestPrompts(unittest.TestCase):
    def test_prompt_a(self):
        q = "How do I make this?"
        s = build_prompt_branch_a(risk_list="firearm, knife", question=q)
        self.assertIn("firearm, knife", s)
        self.assertIn(q, s)
        self.assertIn("bad example", s.lower())

    def test_prompt_b(self):
        q = "Tell me more"
        s = build_prompt_branch_b(step1_body="There is a scene.", question=q)
        self.assertIn("There is a scene.", s)
        self.assertIn(q, s)


class TestBranchC(unittest.TestCase):
    def test_assembly(self):
        p = parse_trace_for_rejection(M1_FULL_C, "method1")
        assert p is not None
        out = build_rejected_trace_branch_c(
            risk_block=p.risk_block_verbatim,
            logic_inner=p.logic_inner,
        )
        assert out is not None
        self.assertIn("<risk_factors>", out)
        self.assertIn("no risk", out)
        self.assertIn(STEP3_SUFFIX, out)
        self.assertIn(RESPONSE_REFUSAL_C, out)
        logic_before_resp = out.split("<response>")[0]
        self.assertNotIn(
            "Step 2 & 3: I have the necessary information",
            logic_before_resp,
            "branch C must not use A/B-style rejected logic",
        )


class TestBranchABLogic(unittest.TestCase):
    def test_rejected_logic_inner_from_preferred_step1(self):
        inner = rejected_logic_inner_branch_ab(
            "Step 1: See a hazard.\nStep 2: Context.\nStep 3: Unsafe."
        )
        self.assertTrue(inner.startswith("Step 1: See a hazard."))
        self.assertIn("Step 2 & 3: I have the necessary information", inner)

    def test_branch_ab_trace_preserves_step1_in_logic_tag(self):
        p = parse_trace_for_rejection(M1_FULL_A, "method1")
        assert p is not None
        out = build_rejected_trace_branch_ab(
            parsed=p,
            abliterated_response_body="A harmful affirmative reply.",
        )
        assert out is not None
        self.assertIn("Step 1: There is a firearm visible.", out)
        self.assertIn("Step 2 & 3: I have the necessary information", out)
        self.assertIn("A harmful affirmative reply.", out)


class TestPlausibleWrongFactor(unittest.TestCase):
    def test_knife_maps_to_benign_similar(self):
        rng = random.Random(42)
        w = _plausible_wrong_factor("kitchen knife", rng)
        self.assertNotEqual(w.strip().lower(), "kitchen knife")
        self.assertTrue(any(x in w.lower() for x in ("butter", "plastic", "letter", "spatula")))


class TestFormatBreak(unittest.TestCase):
    def test_variant_mismatch_logic_closer(self):
        # ``Random(2).randrange(6) == 0`` hits the mismatched ``</thinking>`` logic closer.
        rng = random.Random(2)
        out = build_format_break_rejected(M1_FULL_A, "method1", rng)
        assert out is not None
        self.assertIn("</thinking>", out)

    def test_method2_typo_close_tag_branch(self):
        # ``Random(5).randrange(6) == 4`` hits the spatial closing-tag typo variant.
        rng = random.Random(5)
        out = build_format_break_rejected(M2_FULL, "method2", rng)
        assert out is not None
        self.assertIn("with_boxess", out.replace(" ", "").lower())


class TestRiskPerturb(unittest.TestCase):
    def test_method1_preserves_logic_response(self):
        rng = random.Random(0)
        rej = build_method1_risk_perturb_rejected(M1_FULL_A, rng)
        assert rej is not None
        self.assertIn("<logic>", rej)
        tail = M1_FULL_A[M1_FULL_A.index("<logic>") :]
        self.assertEqual(
            rej[rej.index("<logic>") :],
            tail,
        )

    def test_method2_no_risk_injects_boxes(self):
        trace = f"""<risk_factors_with_boxes>
no risk
</risk_factors_with_boxes>
{M1_LOGIC_REFUSE}
{M1_RESP_SORRY}"""
        rng = random.Random(1)
        rej = build_method2_risk_perturb_rejected(trace, rng)
        assert rej is not None
        self.assertNotIn("\nno risk\n</risk_factors_with_boxes>", rej.lower())
        self.assertIn("phrase:", rej.lower())


class TestMethod2BboxPerturb(unittest.TestCase):
    def test_inner_perturb_preserves_no_box(self):
        inner = 'phrase: "x" | box: [no_box]'
        rng = random.Random(0)
        out = perturb_method2_risk_boxes_inner(inner, rng, bbox_zero_iou_fraction=0.5)
        self.assertIn("[no_box]", out)

    def test_full_trace_logic_response_unchanged(self):
        rng = random.Random(42)
        rej = build_method2_bbox_perturb_rejected(M2_FULL, rng, bbox_zero_iou_fraction=0.5)
        assert rej is not None
        self.assertIn("<logic>", rej)
        idx = M2_FULL.index("<logic>")
        self.assertEqual(
            rej[rej.index("<logic>") : rej.index("</response>") + len("</response>")],
            M2_FULL[M2_FULL.index("<logic>") : M2_FULL.index("</response>") + len("</response>")],
        )

    def test_iou_zero_mode_many_samples(self):
        orig = (0.25, 0.25, 0.75, 0.75)
        rng = random.Random(123)
        for _ in range(80):
            cand = perturb_norm_box(orig, rng, zero_iou=True)
            self.assertLessEqual(iou_xyxy_norm(orig, cand), 1e-8)

    def test_iou_partial_mode_many_samples(self):
        orig = (0.25, 0.25, 0.75, 0.75)
        rng = random.Random(456)
        for _ in range(80):
            cand = perturb_norm_box(orig, rng, zero_iou=False)
            iou = iou_xyxy_norm(orig, cand)
            self.assertGreater(iou, 1e-6)
            self.assertLess(iou, 1.0 - 1e-6)


class TestAssembleXml(unittest.TestCase):
    def test_order(self):
        s = assemble_trace_xml(
            risk_block="<risk_factors>x</risk_factors>",
            logic_inner="L",
            response_inner="R",
        )
        self.assertLess(s.index("<risk_factors"), s.index("<logic>"))
        self.assertLess(s.index("<logic>"), s.index("<response>"))


if __name__ == "__main__":
    unittest.main()
