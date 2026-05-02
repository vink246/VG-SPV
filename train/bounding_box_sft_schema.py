"""
Bounding-box SFT (LoRA) I/O contract: XML segments + normalized box coordinates.

Coordinates are emitted as integers in [0, 1000] per axis (Shikra / scaled-grid style) to avoid
awkward decimal subword tokenization in LLaVA / Llama tokenizers. Values map to [0,1] as v/1000.
"""

from __future__ import annotations

TAG_RISK_WITH_BOXES = "risk_factors_with_boxes"
TAG_LOGIC = "logic"
TAG_RESPONSE = "response"

# Discrete grid for box corners (inclusive). Matches common spatial-VLM practice (Shikra-style scaling).
BOX_COORD_SCALE = 1000

USER_INSTRUCTION_BBOX_SFT = (
    "You are a vision grounding assistant. Given the image and the referring expression, "
    "locate the described object or objects. Reply ONLY with the following XML structure (no markdown):\n"
    f"<{TAG_RISK_WITH_BOXES}>\n"
    "One line per instance (same phrase on each line if multiple instances match the expression): "
    "phrase: \"...\" | box: [x_min, y_min, x_max, y_max]\n"
    f"Each coordinate is an integer from 0 to {BOX_COORD_SCALE} inclusive, "
    f"where 0 is the image origin and {BOX_COORD_SCALE} is the opposite edge (same scale as width and height).\n"
    f"</{TAG_RISK_WITH_BOXES}>\n"
    f"<{TAG_LOGIC}>\nBrief justification.\n</{TAG_LOGIC}>\n"
    f"<{TAG_RESPONSE}>\nShort confirmation.\n</{TAG_RESPONSE}>"
)


def format_norm_box(x0: float, y0: float, x1: float, y1: float) -> str:
    """Format one box as integer corners on a 0..1000 grid (zero-padded to 4 digits for uniform length)."""
    bx0 = int(max(0.0, min(1.0, x0)) * BOX_COORD_SCALE)
    by0 = int(max(0.0, min(1.0, y0)) * BOX_COORD_SCALE)
    bx1 = int(max(0.0, min(1.0, x1)) * BOX_COORD_SCALE)
    by1 = int(max(0.0, min(1.0, y1)) * BOX_COORD_SCALE)
    return f"[{bx0:04d}, {by0:04d}, {bx1:04d}, {by1:04d}]"


def build_assistant_bbox_sft_multi(phrase: str, boxes: list[tuple[float, float, float, float]]) -> str:
    """Build assistant text with one `phrase: ... | box: ...` line per bounding box (same phrase repeated)."""
    if not boxes:
        raise ValueError("boxes must be non-empty")
    lines = "\n".join(f'phrase: "{phrase}" | box: {format_norm_box(*b)}' for b in boxes)
    boxes_joined = ", ".join(format_norm_box(*b) for b in boxes)
    n = len(boxes)
    logic = (
        f"The region(s) {boxes_joined} cover the pixels matching the description ({n} instance{'s' if n != 1 else ''}).\n"
        if n > 1
        else f"The region {format_norm_box(*boxes[0])} covers the pixels matching the description.\n"
    )
    resp = (
        f"Located {n} instance(s) for: {phrase}\n"
        if n > 1
        else f"Located the object described by: {phrase}\n"
    )
    return (
        f"<{TAG_RISK_WITH_BOXES}>\n"
        f"{lines}\n"
        f"</{TAG_RISK_WITH_BOXES}>\n"
        f"<{TAG_LOGIC}>\n"
        f"{logic}"
        f"</{TAG_LOGIC}>\n"
        f"<{TAG_RESPONSE}>\n"
        f"{resp}"
        f"</{TAG_RESPONSE}>"
    )


def build_assistant_bbox_sft(phrase: str, x0: float, y0: float, x1: float, y1: float) -> str:
    return build_assistant_bbox_sft_multi(phrase, [(x0, y0, x1, y1)])


def user_text_with_expression(expression: str) -> str:
    return f'{USER_INSTRUCTION_BBOX_SFT}\nReferring expression: "{expression}"'
